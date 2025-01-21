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
NUM_LEARN = 100000
BATCH_SIZE = 32
NUM_STEPS = 100
gamma = 0.99

submission_map = {
    '41862933':'ry_andy_', 
    '41863713':'ry_andy_', 
}

# データ読み込み
data_dir = 'dataset/train_bc/'
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
    
def build_tile_graph(map_features, relic_nodes, units, team, device='cuda'):
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
    adversal_energy = np.zeros_like(tile_type, dtype=np.float32)
    for i, pos in enumerate(units['position'][1-team]):  # 敵チームのユニット位置
        x, y = pos
        if x == -1 or y == -1: continue
        adversal_energy[x, y] += units['energy'][1-team][i]  
    
    # エネルギー情報と敵ユニット情報を結合
    tiles = np.concatenate([
        tile_type_onehot,  # (24, 24, num_tile_types)
        energy[..., np.newaxis],  # (24, 24, 1)
        team_energy[..., np.newaxis],  # (24, 24, 1)
        adversal_energy[..., np.newaxis]  # (24, 24, 1)
    ], axis=-1)
    return torch.tensor(tiles, dtype=torch.float32, device=device)

# モデルとパラメータの初期化
device = 'cuda' if torch.cuda.is_available() else 'cpu'
risk_predicter = RiskPredUNet().to(device)
optimizer = torch.optim.Adam(risk_predicter.parameters(), lr=1e-4)

# ログディレクトリをdatetimeで一意に生成
log_dir = f'logs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
model_dir = f'models/{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
writer = SummaryWriter(log_dir=log_dir)

# 学習ループ
progress_bar = tqdm(range(NUM_LEARN), desc="Training Progress")
for learn_step in progress_bar:
    total_loss = 0

    for _ in range(BATCH_SIZE):
        data, file_path = load_episode_json(file_pathes)
        if data is None: continue
        
        for id, name in submission_map.items():
            if file_path[17:25] == id:
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

        input_map = build_tile_graph(map_features, relic_nodes, units, team, device=device)
        hazard_map = risk_predicter(input_map)
        
        # 敵座標から hazard_map の値を抽出
        enemy_team = 1 - team  # 敵チームのインデックス
        enemy_positions = units['position'][enemy_team]  # 敵チームの座標リスト

        # 敵座標から hazard_map の値を取得
        extracted_hazard_values = []
        for pos in enemy_positions:
            x, y = pos
            if x != -1 and y != -1:  # 有効な座標のみを対象
                extracted_hazard_values.append(hazard_map[x, y])
        if not extracted_hazard_values:  # 敵がいない場合はスキップ
            continue

        # 抽出した値をテンソル化
        extracted_hazard_values = torch.stack(extracted_hazard_values)  # Tensor化

        # ターゲット値を計算
        target = gamma ** (NUM_STEPS - step) if win_flg else -(gamma ** (NUM_STEPS - step))
        target_tensor = torch.full_like(extracted_hazard_values, target)

        # 損失計算
        loss = F.mse_loss(extracted_hazard_values, target_tensor)
        total_loss += loss

    avg_loss = total_loss / BATCH_SIZE
    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()

    writer.add_scalar('Loss', avg_loss.item(), learn_step)

    # 1000回ごとにモデルを保存
    if (learn_step + 1) % 10000 == 0:
        save_dir = model_dir + f'step_{learn_step + 1}'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(risk_predicter.state_dict(), os.path.join(save_dir, 'risk_predicter.pth'))
        print(f"Model checkpoint saved at step {learn_step + 1}")

writer.close()