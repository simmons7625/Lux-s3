import os
import numpy as np
import torch
from model import *
# 小隊に分割（同じ隊には同じ方策を割り当てる）
# 探索：スタート地点から

class Agent():
    def __init__(self, player: str, env_cfg, model_dir) -> None:
        self.player = player
        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id
        self.env_cfg = env_cfg
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.point_map = np.ones((self.env_cfg["map_width"] - 1, self.env_cfg["map_height"] - 1), dtype=np.float32)
        self.point = 0
        if self.team_id == 0:
            self.start = np.array([0, 0])
        else:
            self.start = np.array([self.env_cfg["map_width"] - 1, self.env_cfg["map_height"] - 1])

    def update_point_map(self, point_delta, unit_positions):
        if point_delta <= 0:
            return
        for relic_pos in self.relic_node_positions:
            center_x, center_y = relic_pos
            for unit_pos in unit_positions:
                dx, dy = unit_pos[0] - center_x, unit_pos[1] - center_y
                if -2 <= dx <= 2 and -2 <= dy <= 2:
                    self.point_map[dx + 2, dy + 2] += point_delta

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
            if map_idx is not None and map_idx in self.point_map.keys():
                # リリックノード付近のポイントマップを参照
                if np.linalg.norm(np.array(nearest_relic_node) - np.array(unit_pos), ord=1) <= 2:
                    probs = self.point_map[map_idx].flatten()
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
                # 対称の位置にも
        
        # ポイント増加量を更新
        point_delta = obs['team_points'][self.team_id] - self.point
        self.update_point_map(point_delta, unit_positions)
        self.point = obs['team_points'][self.team_id]
        for unit_id in np.where(units_mask[self.team_id])[0]:
                unit_pos = unit_positions[unit_id]
                target_pos = self._determine_target(unit_pos)
                action_probabilities = self._calculate_action_probabilities(unit_pos, target_pos, map_features['tile_type'])
                actions[unit_id, 0] = np.random.choice(5, p=action_probabilities)              
        return actions
