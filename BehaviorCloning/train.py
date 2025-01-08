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
alpha = 1
beta = 1 - alpha
gamma_base = 0.95
NUM_LEARN = 10000
BATCH_SIZE = 32
NUM_STEPS = 100

# データ読み込み
data_dir = 'dataset/'
master_df = pd.read_csv(data_dir + 'episodes.csv')

# ファイルパスリストを生成
def generate_file_paths(master_df, data_dir):
    file_paths = []
    for sub_id, df in master_df.groupby("SubmissionId"):
        episode_ids = df["EpisodeId"].unique()
        for ep_id in episode_ids:
            file_path = data_dir + f"{sub_id}_{ep_id}.json"
            file_paths.append(file_path)
    return file_paths

file_pathes = generate_file_paths(master_df, data_dir)

# エピソードデータの読み込み
def load_episode_json(file_pathes):
    file_path = random.choice(file_pathes)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

# グラフ構築関数
def build_tile_graph(map_features, relic_nodes, units, team, tile_embedder, device='cuda'):
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

def build_unit_graph(units, units_mask, team, device='cuda'):
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

# モデルとパラメータの初期化
device = 'cuda' if torch.cuda.is_available() else 'cpu'
imitator = GATActor().to(device)
tile_embedder = TileEmbedding().to(device)

optimizer = torch.optim.Adam(
    list(imitator.parameters()) + list(tile_embedder.parameters()),
    lr=0.001
)

# ログディレクトリをdatetimeで一意に生成
log_dir = f'logs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
writer = SummaryWriter(log_dir=log_dir)

# 学習ループ
progress_bar = tqdm(range(NUM_LEARN), desc="Training Progress")
for learn_step in progress_bar:
    total_bc_loss = 0
    total_rl_loss = 0

    for _ in range(BATCH_SIZE):
        data = load_episode_json(file_pathes)
        if data is None: continue

        winner = np.argmax(data['rewards'])
        ep = random.randint(0, 4)
        step = random.randint(1, NUM_STEPS)

        step_log = data['steps'][ep * (NUM_STEPS + 1) + step][winner]
        obs = json.loads(step_log['observation']['obs'])

        units_mask = obs['units_mask']
        units = obs['units']
        map_features = obs['map_features']
        relic_nodes = obs['relic_nodes']

        sample_actions = np.array(step_log['action'])[np.where(units_mask[winner])[0]]
        sample_actions = torch.tensor(sample_actions, dtype=torch.long, device=device)
        if len(sample_actions) == 0:
            continue

        unit_nodes, units_edges = build_unit_graph(units, units_mask, winner, device=device)
        tile_nodes = build_tile_graph(map_features, relic_nodes, units, winner, tile_embedder, device=device)
        input_nodes = torch.cat([unit_nodes, tile_nodes], dim=-1)

        action_probs, action_values = imitator.forward(input_nodes, units_edges)
        selected_action_probs = action_probs.gather(1, sample_actions[:, 0].unsqueeze(1)).squeeze(1)
        bc_loss = -torch.log(selected_action_probs + 1e-8).mean()

        current_point_diff = obs['team_points'][winner] - obs['team_points'][1 - winner]
        previous_obs = json.loads(data['steps'][ep * (NUM_STEPS + 1) + step][winner]['observation']['obs'])
        previous_point_diff = 0 if step == 0 else previous_obs['team_points'][winner] - previous_obs['team_points'][1 - winner]
        reward = current_point_diff - previous_point_diff
        gamma = gamma_base ** (NUM_STEPS - step)
        selected_action_values = action_values.gather(1, sample_actions[:, 0].unsqueeze(1)).squeeze(1)
        q = selected_action_values.sum()
        rl_loss = ((1 - torch.tanh(torch.tensor(reward) + gamma * q)) ** 2).mean()

        total_bc_loss += bc_loss
        total_rl_loss += rl_loss

    avg_bc_loss = total_bc_loss / BATCH_SIZE
    avg_rl_loss = total_rl_loss / BATCH_SIZE
    avg_loss = alpha * avg_bc_loss + beta * avg_rl_loss

    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()

    writer.add_scalar('Loss/Total', avg_loss.item(), learn_step)
    writer.add_scalar('Loss/BC', avg_bc_loss.item(), learn_step)
    writer.add_scalar('Loss/RL', avg_rl_loss.item(), learn_step)

    # 1000回ごとにモデルを保存
    if (learn_step + 1) % 1000 == 0:
        save_dir = f'models/{datetime.now().strftime("%Y%m%d_%H%M%S")}_step_{learn_step + 1}'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(tile_embedder.state_dict(), os.path.join(save_dir, 'tile_embedder.pth'))
        torch.save(imitator.state_dict(), os.path.join(save_dir, 'gnn_actor.pth'))
        print(f"Model checkpoint saved at step {learn_step + 1}")

writer.close()