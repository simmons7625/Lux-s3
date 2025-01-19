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
            if idx not in self.relic_node_point_map:
                self.relic_node_point_map[idx] = np.zeros((5, 5), dtype=np.float32)
            
            center_x, center_y = relic_pos
            for unit_pos in unit_positions:
                dx, dy = unit_pos[0] - center_x, unit_pos[1] - center_y
                if -2 <= dx <= 2 and -2 <= dy <= 2:
                    self.relic_node_point_map[idx][dx + 2, dy + 2] += point_delta

    def build_tile_graph(self, map_features, relic_nodes, units, team, tile_embedder, device='cpu'):
        """
        マップ上のタイル情報から特徴量グラフを構築
        """
        tile_type = np.array(map_features['tile_type'])
        energy = np.array(map_features['energy'])
        tiles = np.concatenate([tile_type[..., np.newaxis], energy[..., np.newaxis]], axis=-1)

        for pos in relic_nodes:
            if pos[0] != -1 and pos[1] != -1:
                tiles[pos[0], pos[1], 0] = 3  # リリックノードを特別な値に設定

        adversal_pos = np.zeros_like(tile_type, dtype=np.float32)
        for pos in units['position'][1 - team]:
            if pos[0] != -1 and pos[1] != -1:
                adversal_pos[pos[0], pos[1]] += 1

        tiles = np.concatenate([tiles, adversal_pos[..., np.newaxis]], axis=-1)
        embed_tile = tile_embedder(torch.tensor(tiles, dtype=torch.float32, device=device))
        return embed_tile

    def build_unit_graph(self, units, units_mask, team, device='cpu'):
        """
        ユニット情報からグラフを構築
        """
        indices = np.where(units_mask[team])[0]
        team_positions = np.array(units['position'][team])[indices]
        team_energies = np.array(units['energy'][team])[indices]

        unit_nodes = np.zeros((len(team_positions), 3), dtype=np.float32)
        unit_nodes[:, :2] = team_positions
        unit_nodes[:, 2] = team_energies

        unit_nodes = torch.tensor(unit_nodes, dtype=torch.float32, device=device)
        num_nodes = len(unit_nodes)
        edge_index = torch.combinations(torch.arange(num_nodes, device=device), r=2, with_replacement=False)
        edge_index = torch.cat([edge_index, edge_index.flip(dims=(1,))], dim=0).T
        return unit_nodes, edge_index

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
        # print(exp_values)
        if exp_values.sum() == 0:
            return np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        return exp_values / exp_values.sum()
    
    def _determine_target(self, unit_pos):
        """
        ユニットの現在位置から次のターゲット位置を決定
        """
        if self.relic_node_positions:
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
            if map_idx is not None:
                # リリックノード付近のポイントマップを参照
                if np.linalg.norm(np.array(nearest_relic_node) - np.array(unit_pos), ord=1) <= 2:
                    probs = self.relic_node_point_map.get(map_idx, np.ones((5, 5)).flatten())
                    probs = probs / probs.sum()
                    chosen_index = np.random.choice(25, p=probs)
                    offset = np.array([chosen_index // 5, chosen_index % 5]) - 2
                    return np.array(nearest_relic_node) + offset
            return np.array(nearest_relic_node)
        else:
            # リリックノードが見つからない場合はユニットの重心方向
            unit_center = np.mean([unit_pos], axis=0)
            return unit_pos + (unit_pos - unit_center)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        アクションを選択
        """
        units_mask = obs['units_mask']
        units = obs['units']
        map_features = obs['map_features']
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

        # 各ユニットの行動を決定
        for unit_id in np.where(units_mask[self.team_id])[0]:
            unit_pos = unit_positions[unit_id]
            target_pos = self._determine_target(unit_pos)
            action_probabilities = self._calculate_action_probabilities(unit_pos, target_pos, map_features['tile_type'])
            actions[unit_id, 0] = np.random.choice(5, p=action_probabilities)

        # ポイント増加量を更新
        point_delta = obs['team_points'][self.team_id] - self.point
        self.update_relic_node_points(point_delta, unit_positions)
        self.point = obs['team_points'][self.team_id]

        return actions