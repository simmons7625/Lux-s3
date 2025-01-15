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
NUM_LEARN = 10000
BATCH_SIZE = 64
NUM_STEPS = 100

submission_map = {
    '41862933':'ry_andy_', 
    '41863713':'ry_andy_', 
    # '41789980':"aDg4b"
}

# データ読み込み
data_dir = 'dataset/'
master_df = pd.read_csv(data_dir + 'episodes.csv')

# ファイルパスリストを生成
def generate_file_paths(master_df, data_dir):
    file_paths = []
    for sub_id, df in master_df.groupby("SubmissionId"):
        if sub_id in [41862933, 41863713]:
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
        return data, file_path
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

    for _ in range(BATCH_SIZE):
        data, file_path = load_episode_json(file_pathes)
        if data is None: continue
        
        for id, name in submission_map.items():
            if file_path[8:16] == id:
                team_name = name
        team = 0 if data['info']['TeamNames'][0] == team_name else 1
        win_flg = np.argmax(data['rewards']) == team
        ep = random.randint(0, 4)
        step = random.randint(50, NUM_STEPS)

        step_log = data['steps'][ep * (NUM_STEPS + 1) + step][team]
        obs = json.loads(step_log['observation']['obs'])

        units_mask = obs['units_mask']
        units = obs['units']
        map_features = obs['map_features']
        relic_nodes = obs['relic_nodes']

        sample_actions = np.array(step_log['action'])[np.where(units_mask[team])[0]]
        sample_actions = torch.tensor(sample_actions, dtype=torch.long, device=device)
        if len(sample_actions) == 0:
            continue

        unit_nodes, units_edges = build_unit_graph(units, units_mask, team, device=device)
        tile_nodes = build_tile_graph(map_features, relic_nodes, units, team, tile_embedder, device=device)
        input_nodes = torch.cat([unit_nodes, tile_nodes], dim=-1)

        action_probs, action_values = imitator.forward(input_nodes, units_edges)
        selected_action_probs = action_probs.gather(1, sample_actions[:, 0].unsqueeze(1)).squeeze(1)
        bc_loss = -(torch.log(selected_action_probs + 1e-8).sum())

        total_bc_loss += bc_loss

    avg_bc_loss = total_bc_loss / BATCH_SIZE

    optimizer.zero_grad()
    avg_bc_loss.backward()
    optimizer.step()

    writer.add_scalar('Loss/BC', avg_bc_loss.item(), learn_step)

    # 1000回ごとにモデルを保存
    if (learn_step + 1) % 1000 == 0:
        save_dir = model_dir + f'step_{learn_step + 1}'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(tile_embedder.state_dict(), os.path.join(save_dir, 'tile_embedder.pth'))
        torch.save(imitator.state_dict(), os.path.join(save_dir, 'gnn_actor.pth'))
        print(f"Model checkpoint saved at step {learn_step + 1}")

writer.close()