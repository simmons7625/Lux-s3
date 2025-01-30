import os
import torch
import numpy as np
import json
import random
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from model import *

class Trainer:
    def __init__(self, config):
        self.gamma = config['gamma']
        self.num_learn = config['num_learn']
        self.batch_size = config['batch_size']
        self.num_steps = config['num_steps']
        self.data_dir = config['data_dir']
        self.model_dir = config['load_model_dir']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log_dir = config['log_dir']
        self.save_model_dir = config['save_model_dir']

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.imitator = GATActor().to(self.device)
        self.tile_embedder = TileEmbeddingCNN().to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.imitator.parameters()) + list(self.tile_embedder.parameters()),
            lr=1e-4
        )

        self.load_model_weights()

    def load_model_weights(self):
        imitator_path = os.path.join(self.model_dir, 'gnn_actor.pth')
        tile_embedder_path = os.path.join(self.model_dir, 'tile_embedder.pth')
        self.imitator.load_state_dict(torch.load(imitator_path, map_location=self.device, weights_only=True))
        self.tile_embedder.load_state_dict(torch.load(tile_embedder_path, map_location=self.device, weights_only=True))

    def generate_file_paths(self):
        return [os.path.join(self.data_dir, f"episode_{ep_id}.json") for ep_id in range(10)]

    def load_episode_json(self, file_paths):
        file_path = random.choice(file_paths)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f), file_path
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return None, None

    # グラフ構築関数
    def build_tile_graph(self, map_features, relic_nodes, units, units_mask, sensor_range, team):
        # タイルタイプとエネルギーを取得
        tile_type = np.array(map_features['tile_type'])  # (24, 24)
        energy = np.array(map_features['energy'])  # (24, 24)
        tile_type, energy = self.edit_map_features(
            np.array(map_features['tile_type']), 
            np.array(map_features['energy']),
            units,
            units_mask,
            sensor_range,
            team
        )

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
        enemy_map = np.zeros_like(tile_type, dtype=np.float32)
        indices = np.where(units_mask[1-team])[0]
        enemy_positions = np.array(units['position'][1-team])[indices]
        for pos in enemy_positions:  # 敵チームのユニット位置
            x, y = pos
            enemy_map[x, y] += 1

        # エネルギー情報と敵ユニット情報を結合
        tiles = np.concatenate([
            tile_type_onehot,  # (24, 24, num_tile_types)
            energy[..., np.newaxis],  # (24, 24, 1)
            enemy_map[..., np.newaxis]  # (24, 24, 1)
        ], axis=-1)  # 最終形状: (24, 24, num_tile_types + 2)

        # タイル特徴量を埋め込み
        embed_tile = self.tile_embedder(torch.tensor(tiles, dtype=torch.float32, device=self.device))

        # チームのユニットごとにタイル特徴量を取得
        tile_features = []
        indices = np.where(units_mask[team])[0]
        team_positions = np.array(units['position'][team])[indices]
        for pos in team_positions:  # 敵チームのユニット位置
            x, y = pos
            tile_features.append(embed_tile[x, y, :])  # 対応するタイルの特徴量を取得

        # 結果をスタックしてTensorで返す
        return torch.stack(tile_features).to(self.device)          

    def build_unit_graph(self, units, units_mask, team):
        indices = np.where(units_mask[team])[0]
        team_positions = np.array(units['position'][team])[indices]
        team_energies = np.array(units['energy'][team])[indices]

        team_nodes = np.zeros((len(team_positions), 3), dtype=np.float32)
        team_nodes[:, :2] = team_positions
        team_nodes[:, 2:3] = team_energies

        units_nodes = torch.tensor(team_nodes, dtype=torch.float32, device=self.device)
        num_nodes = len(units_nodes)
        edge_index = torch.combinations(torch.arange(num_nodes, device=self.device), r=2, with_replacement=False)
        edge_index = torch.cat([edge_index, edge_index.flip(dims=(1,))], dim=0).T

        return units_nodes, edge_index

    def edit_map_features(self, tile_type, energy, units, units_mask, sensor_range, team):
        indices = np.where(units_mask[team])[0]
        team_positions = np.array(units['position'][team])[indices]
        edited_tile_type = np.full_like(tile_type, -1)
        edited_energy = np.full_like(energy, -1)
        for pos in team_positions:
            for x in range(24):
                for y in range(24):
                    if abs(x - pos[0]) <= sensor_range and abs(y - pos[1]) <= sensor_range:
                        edited_tile_type[x, y] = tile_type[x, y]
                        edited_energy[x, y] = energy[x, y]

        return edited_tile_type, edited_energy

    def train(self):
        file_paths = self.generate_file_paths()
        progress_bar = tqdm(range(self.num_learn), desc="Training Progress")
        for learn_step in progress_bar:
            total_rl_loss = 0

            for _ in range(self.batch_size):
                data, file_path = self.load_episode_json(file_paths)
                if data is None: continue
                
                team = random.randint(0, 1)
                ep = random.randint(0, 4)
                step = random.randint(50, self.num_steps)
                sensor_range = data['params']['unit_sensor_range']

                current_match_result = np.array(data['observations'][ep * (self.num_steps + 1) + (self.num_steps + 1)]['team_wins'])
                previous_match_result = np.array(data['observations'][(ep - 1) * (self.num_steps + 1) + (self.num_steps + 1)]['team_wins'])
                win_flg = (current_match_result - previous_match_result)[team] == 1
                obs = data['observations'][ep * (self.num_steps + 1) + step]
                action = data['actions'][ep * (self.num_steps + 1) + step][f'player_{team}']

                units_mask = obs['units_mask']
                units = obs['units']
                map_features = obs['map_features']
                relic_nodes = obs['relic_nodes']

                sample_actions = np.array(action)[np.where(units_mask[team])[0]]
                sample_actions = torch.tensor(sample_actions, dtype=torch.long, device=self.device)
                if len(sample_actions) == 0:
                    continue

                unit_nodes, units_edges = self.build_unit_graph(units, units_mask, team)
                tile_nodes = self.build_tile_graph(map_features, relic_nodes, units, units_mask, sensor_range, team)
                
                action_probs, action_values = self.imitator.forward(unit_nodes, tile_nodes, units_edges)
                selected_action_probs = action_probs.gather(1, sample_actions[:, 0].unsqueeze(1)).squeeze(1)

                selected_action_values = action_values.gather(1, sample_actions[:, 0].unsqueeze(1)).squeeze(1)
                value = torch.dot(selected_action_probs, selected_action_values)
                pred = torch.tanh(value)

                if win_flg:
                    target = self.gamma ** (self.num_steps - step)
                else:
                    target = -self.gamma ** (self.num_steps - step)

                rl_loss = (target - pred).abs()
                total_rl_loss += rl_loss

            avg_rl_loss = total_rl_loss / self.batch_size

            self.optimizer.zero_grad()
            avg_rl_loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('Loss/RL', avg_rl_loss.item(), learn_step)

        os.makedirs(self.save_model_dir, exist_ok=True)
        torch.save(self.tile_embedder.state_dict(), os.path.join(self.save_model_dir, 'tile_embedder.pth'))
        torch.save(self.imitator.state_dict(), os.path.join(self.save_model_dir, 'gnn_actor.pth'))

        self.writer.close()
        return self.save_model_dir


if __name__ == "__main__":
    data_dir = 'dataset/train_rl'
    model_dir = 'models/20250119_203638/step_90000'
    save_model_dir = f"models/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(data_dir, exist_ok=True)

    # エピソードを10000step実行して保存
    # for ep in range(20):
    #     path = os.path.join(data_dir, f'episode_{ep}.json')
    #     # simulate
    #     ret_code = os.system(f'luxai-s3 test_agent.py test_agent.py --output={path}') 

    ep = 0
    for i in tqdm(range(1000), desc="Running Episodes", unit="episode"):
        # training
        config = {
            'gamma': 0.99,
            'num_learn': 100,
            'batch_size': 32,
            'num_steps': 100,
            'data_dir': data_dir,
            'load_model_dir': model_dir,
            'save_model_dir': save_model_dir,
            'log_dir': log_dir
        }
        trainer = Trainer(config)
        model_dir = trainer.train()

        # overwrite test_agent.py
        test_agent_path = "test_agent.py"
        with open(test_agent_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        with open(test_agent_path, "w", encoding="utf-8") as f:
            for line in lines:
                if "model_dir=" in line:
                    f.write(f'        agent_dict[player] = Agent(player, configurations["env_cfg"], model_dir="{model_dir}")\n')
                else:
                    f.write(line)

        path = os.path.join(data_dir, f'episode_{ep % 20}.json')
        ep += 1
        # simulate
        ret_code = os.system(f'luxai-s3 test_agent.py test_agent.py --output={path}') 
