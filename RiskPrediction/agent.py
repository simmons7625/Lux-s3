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
        self.sap_manager = RiskPredUNet()

        # モデルの重みをロード
        if model_dir:
            self.model_dir = model_dir
            sap_manager_path = os.path.join(model_dir, 'risk_predicter.pth')
            self.sap_manager.load_state_dict(torch.load(sap_manager_path, map_location=torch.device('cpu'), weights_only=True))
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
    
    def build_tile_graph(self, map_features, relic_nodes, units, team, device='cuda'):
        # タイルタイプとエネルギーを取得
        tile_type = np.array(map_features['tile_type'])  # (24, 24)
        energy = np.array(map_features['energy'])  # (24, 24)

        # Relic Nodeをタイルタイプに反映
        for pos in relic_nodes:
            x, y = pos
            if x == -1 or y == -1: continue
            tile_type[x, y] = 3  # Relic Nodeのラベルを3に設定

        # タイルタイプをOne-Hotエンコード
        num_tile_types = 4  # タイルタイプの種類数（0: 空白, 1: 星雲, 2: 小惑星, 3: Relic Node）
        tile_type_onehot = np.eye(num_tile_types)[tile_type]  # (24, 24, num_tile_types)
        
        # 敵ユニットの位置情報を追加
        team_energy = np.zeros_like(tile_type, dtype=np.float32)
        for i, pos in enumerate(units['position'][team]):  # 敵チームのユニット位置
            x, y = pos
            if x == -1 or y == -1: continue
            team_energy[x, y] += units['energy'][team][i]  

        # 敵ユニットの位置情報を追加
        enemy_energy = np.zeros_like(tile_type, dtype=np.float32)
        for i, pos in enumerate(units['position'][1-team]):  # 敵チームのユニット位置
            x, y = pos
            if x == -1 or y == -1: continue
            enemy_energy[x, y] += units['energy'][1-team][i]  
        
        # エネルギー情報と敵ユニット情報を結合
        tiles = np.concatenate([
            tile_type_onehot,  # (24, 24, num_tile_types)
            energy[..., np.newaxis],  # (24, 24, 1)
            team_energy[..., np.newaxis],  # (24, 24, 1)
            enemy_energy[..., np.newaxis]  # (24, 24, 1)
        ], axis=-1)

        return torch.tensor(tiles, dtype=torch.float32, device=device)

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
                    probs = self.relic_node_point_map.get(map_idx, np.ones((5, 5))).flatten()
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

        input_map = self.build_tile_graph(map_features, self.relic_node_positions, units, self.team_id)
        # pred_hazard_level
        hazard_map = self.sap_manager(input_map)
        # 攻撃アクション (id=5) の処理
        sap_range = self.env_cfg['unit_sap_range']
        enemy_pos = np.array(obs['units']['position'][1 - self.team_id])  # 敵位置

        # 各ユニットの行動を決定
        for unit_id in np.where(units_mask[self.team_id])[0]:
            unit_pos = unit_positions[unit_id]  # 自分のユニットの位置
            target_pos = self._determine_target(unit_pos)  # 目標位置の決定
            action_probabilities = self._calculate_action_probabilities(unit_pos, target_pos, map_features['tile_type'])
            actions[unit_id, 0] = np.random.choice(5, p=action_probabilities)

            # 敵が存在する場合のみ処理
            if enemy_pos.shape[0] > 0:
                for pos in enemy_pos:
                    x, y = pos
                    relevant_pos = pos - unit_pos
                    # マンハッタン距離の計算
                    dist = np.abs(relevant_pos).sum()
                    # 攻撃範囲内かつ hazard_map の条件を満たす場合、攻撃アクションを設定
                    if dist <= sap_range and hazard_map[x, y] < 0 and np.random.random() < abs(hazard_map[x, y]):
                        actions[unit_id, 0] = 5  # 攻撃アクション
                        actions[unit_id, 1:] = relevant_pos  # 攻撃対象の相対位置
                        break

        return actions
