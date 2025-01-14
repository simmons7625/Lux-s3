from lux.utils import direction_to
import sys
import numpy as np
import torch
from model import *

class Agent():
    def __init__(self, player: str, env_cfg, model_dir) -> None:
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
        imitator_path = model_dir + 'gnn_actor.pth'
        tile_embedder_path = model_dir + 'tile_embedder.pth'

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
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape: (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id])  # shape: (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"])  # shape: (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])  # shape: (max_relic_nodes,)
        match_step = obs['match_steps']
        
        # 初期化: アクションサイズ固定 (max_units, 3)
        max_units = self.env_cfg["max_units"]
        actions = np.zeros((max_units, 3), dtype=int)
        
        # 行動可能なエージェントを特定
        mask = units_mask[self.team_id]
        available_unit_ids = np.where(mask)[0]

        if match_step < 50:
            # 初期探索と収集フェーズ
            visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
            for id in visible_relic_node_ids:
                if id not in self.discovered_relic_nodes_ids:
                    self.discovered_relic_nodes_ids.add(id)
                    self.relic_node_positions.append(observed_relic_node_positions[id])

            for unit_id in available_unit_ids:
                unit_pos = unit_positions[unit_id]
                if self.relic_node_positions:
                    nearest_relic_node_position = self.relic_node_positions[0]
                    manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])
                    
                    if manhattan_distance <= 4:
                        # 近距離ならランダムに移動
                        random_direction = np.random.randint(0, 5)
                        actions[unit_id] = [random_direction, 0, 0]
                    else:
                        # 遠距離なら遺物に向かって移動
                        actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
                else:
                    # ランダム探索
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                    actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
        else:
            # 後半フェーズ: モデルを利用して行動を選択
            if available_unit_ids.size > 0:
                unit_nodes, units_edges = self.build_unit_graph(units, units_mask, self.team_id)
                tile_nodes = self.build_tile_graph(map_features, relic_nodes, units, self.team_id, self.tile_embedder)
                input_nodes = torch.cat([unit_nodes, tile_nodes], dim=-1)

                action_probs, action_values = self.imitator.forward(input_nodes, units_edges)
                selected_actions = torch.multinomial(action_probs, num_samples=1)
                selected_actions = torch.cat([selected_actions, torch.zeros(selected_actions.size(0), 2)], dim=-1)

                for idx, active_idx in enumerate(available_unit_ids):
                    actions[active_idx] = selected_actions[idx].cpu().numpy()

        # 攻撃アクション (id=5) の処理
        sap_range = self.env_cfg['unit_sap_range']
        adversal_pos = torch.tensor(obs['units']['position'][1 - self.team_id], dtype=torch.float32)  # 敵位置

        if adversal_pos.shape[0] > 0:  # 敵が存在する場合のみ処理
            attack_action_mask = actions[:, 0] == 5
            if attack_action_mask.any():
                unit_positions_tensor = torch.tensor(unit_positions, dtype=torch.float32)  # 自チームユニット位置
                relative_positions = adversal_pos.unsqueeze(1) - unit_positions_tensor.unsqueeze(0)  # shape: (num_enemies, num_units, 2)
                distances = torch.norm(relative_positions, dim=-1)  # shape: (num_enemies, num_units)
                nearest_enemy_indices = torch.argmin(distances, dim=0)  # 最近傍の敵

                nearest_relative_positions = relative_positions[nearest_enemy_indices, torch.arange(unit_positions_tensor.size(0))]  # shape: (num_units, 2)
                nearest_relative_positions = torch.clamp(nearest_relative_positions, min=-sap_range, max=sap_range)
                actions[attack_action_mask, 1:3] = nearest_relative_positions[attack_action_mask]

        return actions


