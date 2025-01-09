import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TileEmbedding(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=3, num_heads=4):
        super(TileEmbedding, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tile):
        tile = F.relu(self.fc_in(tile))  # 初期線形変換
        for attention in self.attention_layers:
            tile, _ = attention(tile, tile, tile)  # 自己注意
            tile = F.relu(tile)  # 非線形変換
        tile = F.relu(self.fc_out(tile))  # 出力線形変換
        return tile


class GATActor(nn.Module):
    def __init__(self, input_dim=3 + 128, hidden_dim=256, output_dim=6, num_layers=3, heads=4):
        super(GATActor, self).__init__()
        self.gat_layers = nn.ModuleList()

        # 初期 GAT レイヤー
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True, add_self_loops=True))

        # 中間 GAT レイヤー
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, add_self_loops=True))

        # 出力層
        self.fc = nn.Linear(hidden_dim * heads, output_dim)

    def forward(self, x, edge_index):
        for gat in self.gat_layers:
            x = F.relu(gat(x, edge_index))  # GAT レイヤー
            x = F.dropout(x, p=0.1, training=self.training)  # Dropout を追加

        x = self.fc(x)  # 最終出力
        return F.softmax(x, dim=1), x

class PositionalTileEmbedding(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=3, num_heads=4):
        super(PositionalTileEmbedding, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

        # Positional Encoding の初期化
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, hidden_dim))  # 長さ100のシーケンスを仮定
        nn.init.xavier_uniform_(self.positional_encoding)  # 重みをXavier初期化

    def forward(self, tile):
        batch_size, seq_len, _ = tile.size()

        # Positional Encoding のサイズを入力タイルのサイズに合わせる
        pos_enc = self.positional_encoding[:, :seq_len, :].repeat(batch_size, 1, 1)

        # 入力に Positional Encoding を加算
        tile = F.relu(self.fc_in(tile)) + pos_enc

        # Attention 層を適用
        for attention in self.attention_layers:
            tile, _ = attention(tile, tile, tile)  # 自己注意
            tile = F.relu(tile)  # 非線形変換

        # 出力層
        tile = F.relu(self.fc_out(tile))  # 出力線形変換
        return tile
