from lux.utils import direction_to
import sys
import numpy as np
import torch
from model import *

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

        # モデルの初期化
        self.imitator = GATActor()
        self.tile_embedder = TileEmbeddingCNN()

        # モデルの重みをロード
        imitator_path = 'models/20250109_213826/step_10000/gnn_actor.pth'
        tile_embedder_path = 'models/20250109_213826/step_10000/tile_embedder.pth'

        self.imitator.load_state_dict(torch.load(imitator_path, map_location=torch.device('cpu')))
        self.tile_embedder.load_state_dict(torch.load(tile_embedder_path, map_location=torch.device('cpu')))
    
    # グラフ構築関数
    def build_tile_graph(self, map_features, relic_nodes, units, team, tile_embedder, device='cpu'):
        tile_type = np.array(map_features['tile_type'])
        energy = np.array(map_features['energy'])
        tiles = np.concatenate([tile_type[..., np.newaxis], energy[..., np.newaxis]], axis=-1)

        for pos in relic_nodes:
            x, y = pos
            if x == -1 or y == -1: continue
            tiles[x, y, 0] = 3

        adversal_pos = np.zeros_like(tile_type, dtype=np.float32)
        for pos in units['position'][1 - team]:
            x, y = pos
            if x == -1 or y == -1: continue
            adversal_pos[x, y] += 1

        tiles = np.concatenate([tiles, adversal_pos[..., np.newaxis]], axis=-1)
        embed_tile = tile_embedder(torch.tensor(tiles, dtype=torch.float32, device=device))

        tile_features = []
        for pos in units['position'][team]:
            x, y = pos
            if x == -1 or y == -1: continue
            tile_features.append(embed_tile[x, y, :])

        return torch.stack(tile_features).to(device)

    def build_unit_graph(self, units, units_mask, team, device='cpu'):
        indices = np.where(units_mask[team])[0]
        team_positions = np.array(units['position'][team])[indices]
        team_energies = np.array(units['energy'][team])[indices]

        team_nodes = np.zeros((len(team_positions), 3), dtype=np.float32)
        team_nodes[:, :2] = team_positions
        team_nodes[:, 2] = team_energies

        units_nodes = torch.tensor(team_nodes, dtype=torch.float32, device=device)
        num_nodes = len(units_nodes)
        edge_index = torch.combinations(torch.arange(num_nodes, device=device), r=2, with_replacement=False)
        edge_index = torch.cat([edge_index, edge_index.flip(dims=(1,))], dim=0).T

        return units_nodes, edge_index

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        units_mask = obs['units_mask']
        units = obs['units']
        map_features = obs['map_features']
        relic_nodes = obs['relic_nodes']
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
    
        # 初期化: アクションサイズ固定 (max_units, 3)
        max_units = self.env_cfg["max_units"]
        actions = np.zeros((max_units, 3), dtype=int)
        # 行動可能なエージェントを特定
        mask = units_mask[self.team_id]
        if step < 50:
            # ids of units you can control at this timestep
            available_unit_ids = np.where(mask)[0]
            # visible relic nodes
            visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])

            for id in visible_relic_node_ids:
                if id not in self.discovered_relic_nodes_ids:
                    self.discovered_relic_nodes_ids.add(id)
                    self.relic_node_positions.append(observed_relic_node_positions[id])

            # unit ids range from 0 to max_units - 1
            for unit_id in available_unit_ids:
                unit_pos = unit_positions[unit_id]
                if len(self.relic_node_positions) > 0:
                    nearest_relic_node_position = self.relic_node_positions[0]
                    manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])
                    
                    # if close to the relic node we want to hover around it and hope to gain points
                    if manhattan_distance <= 4:
                        random_direction = np.random.randint(0, 5)
                        actions[unit_id] = [random_direction, 0, 0]
                    else:
                        # otherwise we want to move towards the relic node
                        actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
                else:
                    # randomly explore by picking a random location on the map and moving there for about 20 steps
                    if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                        rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                        self.unit_explore_locations[unit_id] = rand_loc
                    actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
        elif len(np.where(mask)[0]) > 0:
            # グラフ構築
            unit_nodes, units_edges = self.build_unit_graph(units, units_mask, self.team_id)
            tile_nodes = self.build_tile_graph(map_features, relic_nodes, units, self.team_id, self.tile_embedder)
            input_nodes = torch.cat([unit_nodes, tile_nodes], dim=-1)

            # モデルの予測
            action_probs, action_values = self.imitator.forward(input_nodes, units_edges)

            # アクション選択 (N, 3)
            selected_actions = torch.multinomial(action_probs, num_samples=1).unsqueeze(-1)
            selected_actions = torch.cat([selected_actions, torch.zeros(selected_actions.size(0), 2)], dim=-1)  # (N, 3)

            # 攻撃アクション (id=5) 用の設定
            sap_range = self.env_cfg['unit_sap_range']
            adversal_pos = torch.tensor(obs['units']['position'][1 - self.team_id])  # 敵の位置をテンソルに変換

            # 攻撃アクション (id=5) の処理
            attack_action_mask = selected_actions[:, 0] == 5  # 攻撃アクションを選択したエージェント
            if attack_action_mask.any():
                # 攻撃位置の相対座標を計算
                unit_positions = torch.tensor(units['position'][self.team_id], dtype=torch.float32)  # 自チームの位置をテンソルに変換
                relative_positions = adversal_pos - unit_positions  # 敵との相対座標

                # 攻撃範囲内に制限
                relative_positions = torch.clamp(relative_positions, min=-sap_range, max=sap_range)

                # 攻撃アクションのエージェントに相対座標を割り当て
                selected_actions[attack_action_mask, 1:3] = relative_positions[attack_action_mask].to(selected_actions.dtype)

            # 行動可能なエージェントのみにアクションを設定
            active_indices = torch.where(torch.tensor(mask))[0]  # 行動可能なユニットのインデックス
            for idx, active_idx in enumerate(active_indices):
                actions[active_idx] = selected_actions[idx].cpu().numpy()
                
        return actions


