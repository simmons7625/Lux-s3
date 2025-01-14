import os
import torch
import numpy as np
import json
import pandas as pd
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from model import *

# config
gamma_base = 0.95
NUM_LEARN = 10000
BATCH_SIZE = 64
NUM_STEPS = 100

# データ読み込み
data_dir = 'dataset/test/20250110_165639/'

# ファイルパスリストを生成
def generate_file_paths(data_dir):
    file_paths = []
    for ep_id in range(100):
        file_path = data_dir + f"episode_{ep_id}.json"
        file_paths.append(file_path)
    return file_paths

file_pathes = generate_file_paths(data_dir)

# エピソードデータの読み込み
def load_episode_json(file_pathes):
    file_path = random.choice(file_pathes)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data, file_path
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

# グラフ構築関数
def build_tile_graph(map_features, relic_nodes, units, units_mask, sensor_range, team, tile_embedder, device='cuda'):
    tile_type = np.array(map_features['tile_type'])
    energy = np.array(map_features['energy'])
    tile_type, energy = edit_map_features(tile_type, energy, units, units_mask, sensor_range, team)
    tiles = np.concatenate([tile_type[..., np.newaxis], energy[..., np.newaxis]], axis=-1)

    for pos in relic_nodes:
        x, y = pos
        if x == -1 or y == -1: continue
        tiles[x, y, 0] = 3

    adversal_map = np.zeros_like(tile_type, dtype=np.float32)
    adversal_indices = np.where(units_mask[1-team])[0]
    adversal_pos = np.array(units['position'][1-team])[adversal_indices]
    for pos in adversal_pos:
        x, y = pos
        if x == -1 or y == -1: continue
        adversal_map[x, y] += 1

    tiles = np.concatenate([tiles, adversal_map[..., np.newaxis]], axis=-1)
    embed_tile = tile_embedder(torch.tensor(tiles, dtype=torch.float32, device=device))

    tile_features = []
    team_indices = np.where(units_mask[team])[0]
    team_pos = np.array(units['position'][team])[team_indices]
    for pos in team_pos:
        x, y = pos
        if x == -1 or y == -1: continue
        tile_features.append(embed_tile[x, y, :])

    return torch.stack(tile_features).to(device)

def build_unit_graph(units, units_mask, team, device='cuda'):
    indices = np.where(units_mask[team])[0]
    team_positions = np.array(units['position'][team])[indices]
    team_energies = np.array(units['energy'][team])[indices]

    team_nodes = np.zeros((len(team_positions), 3), dtype=np.float32)
    team_nodes[:, :2] = team_positions
    team_nodes[:, 2:3] = team_energies

    units_nodes = torch.tensor(team_nodes, dtype=torch.float32, device=device)
    num_nodes = len(units_nodes)
    edge_index = torch.combinations(torch.arange(num_nodes, device=device), r=2, with_replacement=False)
    edge_index = torch.cat([edge_index, edge_index.flip(dims=(1,))], dim=0).T

    return units_nodes, edge_index

def edit_map_features(tile_type, energy, units, units_mask, sensor_range, team):
    indices = np.where(units_mask[team])[0]
    team_positions = np.array(units['position'][team])[indices]
    edited_tile_type = np.full_like(tile_type, -1)
    edited_energy = np.full_like(energy, -1)
    for pos in team_positions:
        for x in range(24):
            for y in range(24):
                if abs(x-pos[0])<=sensor_range and abs(y-pos[1])<=sensor_range:
                    edited_tile_type[x,y] = tile_type[x, y]
                    edited_energy[x,y] = energy[x, y]

    return edited_tile_type, edited_energy

# モデルとパラメータの初期化
device = 'cuda' if torch.cuda.is_available() else 'cpu'
imitator = GATActor().to(device)
tile_embedder = TileEmbeddingCNN().to(device)
# モデルの重みをロード
imitator_path = 'models/20250113_222631/step_10000/gnn_actor.pth'
tile_embedder_path = 'models/20250113_222631/step_10000/tile_embedder.pth'

imitator.load_state_dict(torch.load(imitator_path, map_location=device))
tile_embedder.load_state_dict(torch.load(tile_embedder_path, map_location=device))


optimizer = torch.optim.Adam(
    list(imitator.parameters()) + list(tile_embedder.parameters()),
    lr=1e-4
)

# ログディレクトリをdatetimeで一意に生成
log_dir = f'logs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
model_dir = f'models/{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
writer = SummaryWriter(log_dir=log_dir)

# 学習ループ
progress_bar = tqdm(range(NUM_LEARN), desc="Training Progress")
for learn_step in progress_bar:
    total_bc_loss = 0
    total_rl_loss = 0

    for _ in range(BATCH_SIZE):
        data, file_path = load_episode_json(file_pathes)
        if data is None: continue
        team = 0
        ep = random.randint(0, 4)
        step = random.randint(50, NUM_STEPS)
        sensor_range = data['params']['unit_sensor_range']

        win_flg = data['observations'][ep * (NUM_STEPS + 1) + (NUM_STEPS + 1)]['team_wins']
        obs = data['observations'][ep * (NUM_STEPS + 1) + step]
        action = data['actions'][ep * (NUM_STEPS + 1) + step][f'player_{team}']

        units_mask = obs['units_mask']
        units = obs['units']
        map_features = obs['map_features']
        relic_nodes = obs['relic_nodes']

        sample_actions = np.array(action)[np.where(units_mask[team])[0]]
        sample_actions = torch.tensor(sample_actions, dtype=torch.long, device=device)
        if len(sample_actions) == 0:
            continue

        unit_nodes, units_edges = build_unit_graph(units, units_mask, team, device=device)
        tile_nodes = build_tile_graph(map_features, relic_nodes, units, units_mask, sensor_range, team, tile_embedder, device=device)
        input_nodes = torch.cat([unit_nodes, tile_nodes], dim=-1)

        action_probs, action_values = imitator.forward(input_nodes, units_edges)
        selected_action_probs = action_probs.gather(1, sample_actions[:, 0].unsqueeze(1)).squeeze(1)

        current_point_diff = obs['team_points'][team] - obs['team_points'][1 - team]
        previous_obs = data['observations'][ep * (NUM_STEPS + 1) + step-1]
        previous_point_diff = 0 if step == 0 else previous_obs['team_points'][team] - previous_obs['team_points'][1 - team]
        reward = current_point_diff - previous_point_diff
        gamma = gamma_base ** (NUM_STEPS - step)
        selected_action_values = action_values.gather(1, sample_actions[:, 0].unsqueeze(1)).squeeze(1)
        value = torch.dot(selected_action_probs, selected_action_values)
        if win_flg:
            rl_loss = ((100 - torch.tensor(reward) + gamma * value) ** 2)
        else:
            rl_loss = ((-100 - torch.tensor(reward) + gamma * value) ** 2)

        total_rl_loss += rl_loss

    avg_rl_loss = total_rl_loss / BATCH_SIZE

    optimizer.zero_grad()
    avg_rl_loss.backward()
    optimizer.step()

    writer.add_scalar('Loss/RL', avg_rl_loss.item(), learn_step)

    # 1000回ごとにモデルを保存
    if (learn_step + 1) % 1000 == 0:
        save_dir = model_dir + f'step_{learn_step + 1}'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(tile_embedder.state_dict(), os.path.join(save_dir, 'tile_embedder.pth'))
        torch.save(imitator.state_dict(), os.path.join(save_dir, 'gnn_actor.pth'))
        print(f"Model checkpoint saved at step {learn_step + 1}")

writer.close()