import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

def build_unit_graph(units, units_mask, team_points):
    # 現時点で勝利しているチーム (0 or 1)
    win_team = np.argmax(team_points)

    # 各チームでグラフを構成
    units_nodes = []

    for i in range(2):  # 2チーム
        mask = units_mask[i]
        team_positions = units['position'][i][mask]
        team_energies = units['energy'][i][mask]

        # ノード情報: [勝利フラグ, x座標, y座標, エネルギー]
        team_nodes = np.zeros((len(team_positions), 4))
        team_nodes[:, 0] = 1 if i == win_team else 0
        team_nodes[:, 1:3] = team_positions
        team_nodes[:, 3] = team_energies
        units_nodes.append(team_nodes)
        
    return units_nodes

class TileEmbedding(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super(TileEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

    def forward(self, tile):
        tile = F.relu(self.fc(tile))
        output, _ = self.attention(tile, tile, tile)
        return F.relu(output)

def build_tile_graph(tile, units, units_mask):
    tile_embedder = TileEmbedding()

    # タイル埋め込みの取得
    embed_tile = tile_embedder(tile)

    # ユニットの現在地をインデキシング
    units_positions = units['position']
    tile_features = []

    for i in range(2):  # チームごと
        mask = units_mask[i]
        positions = units_positions[i][mask]

        for pos in positions:
            x, y = int(pos[0]), int(pos[1])
            tile_features.append(embed_tile[:, x, y])

    return torch.stack(tile_features)

class GATActor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=5):
        super(GATActor, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, concat=True, add_self_loops=True)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True, add_self_loops=True)
        self.fc = nn.Linear(hidden_dim * 4, output_dim)

    def forward(self, x):
        x = F.relu(self.gat1(x))
        x = F.relu(self.gat2(x))
        x = self.fc(x)
        return x
