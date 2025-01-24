import os
import numpy as np
import torch
from model import *

class Agent():
    def __init__(self, player: str, env_cfg, model_dir) -> None:
        self.player = player
        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id
        self.env_cfg = env_cfg
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.relic_node_point_map = {}
        self.point = 0
        if self.team_id == 0:
            self.start = np.array([0, 0])
        else:
            self.start = np.array([self.env_cfg["map_width"] - 1, self.env_cfg["map_height"] - 1])

        # モデルの初期化
        self.imitator = GATActor()
        self.tile_embedder = TileEmbeddingCNN()

        # モデルの重みをロード
        if model_dir:
            self.model_dir = model_dir
            imitator_path = os.path.join(model_dir, 'gnn_actor.pth')
            tile_embedder_path = os.path.join(model_dir, 'tile_embedder.pth')
            self.imitator.load_state_dict(torch.load(imitator_path, map_location=torch.device('cpu'), weights_only=True))
            self.tile_embedder.load_state_dict(torch.load(tile_embedder_path, map_location=torch.device('cpu'), weights_only=True))
        else:
            self.model_dir = None

    def update_relic_node_points(self, point_delta, unit_positions):
        """
        リリックノード周囲のポイント増加量を更新
        """
        if point_delta <= 0:
            return
        for idx, relic_pos in enumerate(self.relic_node_positions):
            if idx not in self.relic_node_point_map.keys():
                self.relic_node_point_map[idx] = np.ones((5, 5), dtype=np.float32)
            
            center_x, center_y = relic_pos
            for unit_pos in unit_positions:
                dx, dy = unit_pos[0] - center_x, unit_pos[1] - center_y
                if -2 <= dx <= 2 and -2 <= dy <= 2:
                    self.relic_node_point_map[idx][dx + 2, dy + 2] += point_delta
    
    # グラフ構築関数
    def build_tile_graph(self, map_features, relic_nodes, units, team, tile_embedder, device='cpu'):
        # タイルタイプとエネルギーを取得
        tile_type = np.array(map_features['tile_type'])  # (24, 24)
        energy = np.array(map_features['energy'])  # (24, 24)

        # Relic Nodeをタイルタイプに反映
        for pos in relic_nodes:
            x, y = pos
            if x == -1 or y == -1: continue
            tile_type[x, y] = 3  # Relic Nodeのラベルを3に設定
            # 対称の位置にもrelic_nodeは存在
            tile_type[23-y, 23-x] = 3
            
        # タイルタイプをOne-Hotエンコード
        num_tile_types = 4  # タイルタイプの種類数（0: 空白, 1: 星雲, 2: 小惑星, 3: Relic Node）
        tile_type_onehot = np.eye(num_tile_types)[tile_type]  # (24, 24, num_tile_types)

        # 敵ユニットの位置情報を追加
        adversal_energy = np.zeros_like(tile_type, dtype=np.float32)
        for i, pos in enumerate(units['position'][1 - team]):  # 敵チームのユニット位置
            x, y = pos
            if x == -1 or y == -1: continue
            adversal_energy[x, y] += units['energy'][1 - team][i]  # 敵ユニット数をカウント

        # エネルギー情報と敵ユニット情報を結合
        tiles = np.concatenate([
            tile_type_onehot,  # (24, 24, num_tile_types)
            energy[..., np.newaxis],  # (24, 24, 1)
            adversal_energy[..., np.newaxis]  # (24, 24, 1)
        ], axis=-1)  # 最終形状: (24, 24, num_tile_types + 2)

        # タイル特徴量を埋め込み
        embed_tile = tile_embedder(torch.tensor(tiles, dtype=torch.float32, device=device))

        # チームのユニットごとにタイル特徴量を取得
        tile_features = []
        for pos in units['position'][team]:  # 自チームのユニット位置
            x, y = pos
            if x == -1 or y == -1: continue
            tile_features.append(embed_tile[x, y, :])  # 対応するタイルの特徴量を取得

        # 結果をスタックしてTensorで返す
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

    def _calculate_action_probabilities(self, unit_pos, target_pos, tile_types):
        """
        行動確率を計算
        """
        directions = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
        next_positions = unit_pos + directions

        valid_moves = [
            0 <= x < tile_types.shape[0] and 0 <= y < tile_types.shape[1] and tile_types[x, y] != 2
            for x, y in next_positions
        ]

        distances = np.abs(next_positions - target_pos).sum(axis=1).astype(float)
        exp_values = np.exp(-distances)
        exp_values[~np.array(valid_moves)] = 0
        if exp_values.sum() == 0:
            return np.ones_like(exp_values) / len(exp_values)
        return exp_values / exp_values.sum()
    
    def _determine_target(self, unit_pos):
        """
        ユニットの現在位置から次のターゲット位置を決定
        """
        if len(self.relic_node_positions) > 0:
            # 最も近いリリックノードを探索
            nearest_relic_node = min(
                self.relic_node_positions,
                key=lambda pos: np.linalg.norm(np.array(pos) - np.array(unit_pos), ord=1)
            )
            # リリックノードのインデックスを取得
            map_idx = next(
                (idx for idx, pos in enumerate(self.relic_node_positions)
                 if np.array_equal(pos, nearest_relic_node)),
                None
            )
            if map_idx is not None and map_idx in self.relic_node_point_map.keys():
                # リリックノード付近のポイントマップを参照
                if np.linalg.norm(np.array(nearest_relic_node) - np.array(unit_pos), ord=1) <= 2:
                    probs = self.relic_node_point_map[map_idx].flatten()
                    probs = probs / probs.sum()
                    chosen_index = np.random.choice(25, p=probs)
                    offset = np.array([chosen_index // 5, chosen_index % 5]) - 2
                    return np.array(nearest_relic_node) + offset
            return np.array(nearest_relic_node)
        else:
            # far from start
            return unit_pos + (unit_pos - self.start)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        アクションを選択
        """
        units_mask = obs['units_mask']
        units = obs['units']
        map_features = obs['map_features']
        match_step = obs['match_steps']
        observed_relic_node_positions = np.array(obs['relic_nodes'])
        observed_relic_nodes_mask = np.array(obs['relic_nodes_mask'])
        unit_positions = np.array(obs["units"]["position"][self.team_id])

        max_units = self.env_cfg["max_units"]
        actions = np.zeros((max_units, 3), dtype=int)

        # リリックノードの更新
        for idx in np.where(observed_relic_nodes_mask)[0]:
            if idx not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(idx)
                self.relic_node_positions.append(observed_relic_node_positions[idx])
        
        # ポイント増加量を更新
        point_delta = obs['team_points'][self.team_id] - self.point
        self.update_relic_node_points(point_delta, unit_positions)
        self.point = obs['team_points'][self.team_id]

        if self.model_dir and match_step >= 30:
        # if np.where(units_mask[self.team_id])[0].size > 10:
            # 後半フェーズ: モデルを利用して行動を選択
            unit_nodes, units_edges = self.build_unit_graph(units, units_mask, self.team_id)
            tile_nodes = self.build_tile_graph(map_features, self.relic_node_positions, units, self.team_id, self.tile_embedder)
            action_probs, _ = self.imitator.forward(unit_nodes, tile_nodes, units_edges)
            # Check if enemies are within attack range
            sap_range = self.env_cfg['unit_sap_range']
            adversal_pos = np.array(obs['units']['position'][1-self.team_id][np.where(units_mask[1-self.team_id])[0]])  # Enemy positions
            # Edit action_probs
            for idx, unit_id in enumerate(np.where(units_mask[self.team_id])[0]):
                unit_pos = unit_positions[unit_id]
                probs = action_probs[idx, :].detach().numpy()
                action = np.random.choice(6, p=probs)

                if action == 5 and adversal_pos.shape[0] > 0:  # Attack action and enemies exist
                    relative_positions = adversal_pos - unit_pos
                    distances = np.linalg.norm(relative_positions, axis=-1)

                    if np.any(distances <= sap_range):  # Enemy within attack range
                        nearest_enemy_idx = np.argmin(distances)
                        target_relative_position = relative_positions[nearest_enemy_idx]
                        target_relative_position = np.clip(target_relative_position, a_min=-sap_range, a_max=sap_range)
                        actions[unit_id, 0] = action
                        actions[unit_id, 1:3] = target_relative_position
                    else:  # No enemy in range, adjust probabilities
                        target_pos = self._determine_target(unit_pos)
                        weight = self._calculate_action_probabilities(unit_pos, target_pos, map_features['tile_type'])
                        weighted_probs = weight * probs[:5]
                        if weighted_probs.sum() == 0:
                            probs = np.ones_like(weighted_probs) / len(weighted_probs)
                        else:
                            probs = weighted_probs / weighted_probs.sum()
                        actions[unit_id, 0] = np.random.choice(5, p=probs)
                else:  # Non-attack action
                    target_pos = self._determine_target(unit_pos)
                    weight = self._calculate_action_probabilities(unit_pos, target_pos, map_features['tile_type'])
                    weighted_probs = weight * probs[:5]
                    if weighted_probs.sum() == 0:
                        probs = np.ones_like(weighted_probs) / len(weighted_probs)
                    else:
                        probs = weighted_probs / weighted_probs.sum()
                    actions[unit_id, 0] = np.random.choice(5, p=probs)
        else:
            for unit_id in np.where(units_mask[self.team_id])[0]:
                unit_pos = unit_positions[unit_id]
                target_pos = self._determine_target(unit_pos)
                action_probabilities = self._calculate_action_probabilities(unit_pos, target_pos, map_features['tile_type'])
                actions[unit_id, 0] = np.random.choice(5, p=action_probabilities)              
        return actions
