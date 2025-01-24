import numpy as np
from model import *

class Agent():
    def __init__(self, player: str, env_cfg, model_dir) -> None:
        self.player = player
        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id
        self.env_cfg = env_cfg
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.point_map = np.ones((self.env_cfg["map_width"], self.env_cfg["map_height"]), dtype=np.float32)
        self.point = 0
        if self.team_id == 0:
            self.start = np.array([0, 0])
        else:
            self.start = np.array([self.env_cfg["map_width"] - 1, self.env_cfg["map_height"] - 1])

    def update_point_map(self, point_delta, unit_positions, units_mask):
        if point_delta <= 0:
            return
        num_actives = len(np.where(units_mask[self.team_id])[0])
        for unit_pos in unit_positions:
            x, y = unit_pos
            self.point_map[x, y] += point_delta / num_actives

    def calculate_action_probabilities(self, unit_pos, target_pos, tile_types):
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
    
    def explore_behavior(self, unit_pos):
        # 探索行動：スタート地点からのベクトルを利用
        return unit_pos + (unit_pos - self.start)

    def exploit_behavior(self, unit_pos):
        # 最も近いリリックノードを探索
        nearest_relic_node = min(
            self.relic_node_positions,
            key=lambda pos: np.linalg.norm(np.array(pos) - np.array(unit_pos), ord=1)
        )
        # 80% の確率でリリックノードに向かう
        if np.random.random() < 0.8:
            return np.array(nearest_relic_node)
        else:
            # ポイントマップを平坦化して確率的にターゲットを選択
            probs = self.point_map.flatten()
            probs = probs / probs.sum()  # 確率分布を正規化
            chosen_index = np.random.choice(self.env_cfg["map_width"] * self.env_cfg["map_height"], p=probs)
            
            # 2D 座標に変換
            target = np.array([chosen_index // self.env_cfg["map_width"], chosen_index % self.env_cfg["map_height"]])
            
            # マップの範囲内に収めるための調整（必要に応じて）
            target = np.clip(target, 0, [self.env_cfg["map_width"] - 1, self.env_cfg["map_height"] - 1])
            
            return target
 
    def assign_task(self, unit_positions, units_mask):
        # 各リリックノードに対してユニットを割り当て
        task_mask = np.full_like(units_mask[self.team_id], fill_value=False)
        active_idx = np.where(units_mask[self.team_id])[0]
        if len(self.relic_node_positions) > 0:
            for relic in self.relic_node_positions:
                # スタート地点側のリリックノードに追跡者を割り当て
                if np.linalg.norm(relic - self.start) <= self.env_cfg["map_width"]:
                    # ユニット位置とリリックノードとの距離を計算
                    distances = np.linalg.norm(unit_positions[active_idx] - relic, axis=1)
                    sorted_idx = np.argsort(distances)
                    # 近い順にユニットを割り当て
                    for i in range(min(len(sorted_idx), 5)):
                        task_mask[active_idx[sorted_idx[i]]] = True
        return task_mask
        
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        units_mask = obs['units_mask']
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
                # 対称の位置にも存在
                self.relic_node_positions.append(
                    np.array(
                        [
                            self.env_cfg["map_width"] - observed_relic_node_positions[idx][0],
                            self.env_cfg["map_height"] - observed_relic_node_positions[idx][1]
                        ]
                    )
                )
        
        # ポイント増加量を更新
        point_delta = obs['team_points'][self.team_id] - self.point
        self.update_point_map(point_delta, unit_positions, units_mask)
        self.point = obs['team_points'][self.team_id]
        # 行動決定
        task_mask = self.assign_task(unit_positions, units_mask)
        for unit_id in np.where(units_mask[self.team_id])[0]:
                unit_pos = unit_positions[unit_id]
                if task_mask[unit_id]:
                    target_pos = self.exploit_behavior(unit_pos)
                else:
                    target_pos = self.explore_behavior(unit_pos)
                action_probabilities = self.calculate_action_probabilities(unit_pos, target_pos, map_features['tile_type'])
                actions[unit_id, 0] = np.random.choice(5, p=action_probabilities)              
        
        return actions
