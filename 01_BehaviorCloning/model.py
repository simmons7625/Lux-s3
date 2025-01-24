import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TileEmbeddingCNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, num_layers=3, kernel_size=7):
        super(TileEmbeddingCNN, self).__init__()
        self.conv_in = nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.layer_norm_in = nn.LayerNorm(hidden_dim)
        
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        self.conv_out = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.layer_norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, tile):
        # (B, H, W, C) -> (B, C, H, W)
        tile = tile.unsqueeze(0).permute(0, 3, 1, 2)
        # 初期畳み込み + LayerNorm
        tile = self.conv_in(tile)
        tile = tile.permute(0, 2, 3, 1)  # (B, H, W, C) -> (B, H, W, C) for LayerNorm
        tile = F.relu(self.layer_norm_in(tile))
        tile = tile.permute(0, 3, 1, 2)

        # 畳み込み層 + 残差接続 + LayerNorm
        for conv, layer_norm in zip(self.conv_layers, self.layer_norms):
            residual = tile
            tile = conv(tile)
            tile = tile.permute(0, 2, 3, 1)  # (B, H, W, C) -> (B, H, W, C) for LayerNorm
            tile = F.relu(layer_norm(tile))
            tile = tile.permute(0, 3, 1, 2)
            tile = tile + residual  # 残差接続

        # 出力畳み込み + LayerNorm
        tile = self.conv_out(tile)
        tile = tile.permute(0, 2, 3, 1)  # (B, H, W, C) -> (B, H, W, C) for LayerNorm
        tile = F.relu(self.layer_norm_out(tile))
        tile = tile.squeeze(0)
        return tile

class GATActor(nn.Module):
    def __init__(self, unit_dim=3, tile_dim = 32, hidden_dim=128, output_dim=6, num_layers=2, heads=4):
        super(GATActor, self).__init__()
        self.unit_gat_layers = nn.ModuleList()
        # unit GAT レイヤー
        self.unit_gat_layers.append(GATConv(unit_dim, tile_dim, heads=heads, concat=False))
        for _ in range(num_layers - 1):
            self.unit_gat_layers.append(GATConv(tile_dim, tile_dim, heads=heads, concat=False))

        self.grobal_gat_layers = nn.ModuleList()
        self.grobal_gat_layers.append(GATConv(tile_dim*2, hidden_dim, heads=heads, concat=False))
        self.grobal_gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False))
        # 出力層
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, unit_nodes, tile_nodes, edge_index):
        # GAT レイヤー + 残差接続 + LayerNorm
        for i, gat in enumerate(self.unit_gat_layers):
            residual = unit_nodes
            unit_nodes = gat(unit_nodes, edge_index)
            if i > 0:
                unit_nodes = F.relu(unit_nodes + residual)
        
        # グローバル GAT レイヤー + 残差接続 + LayerNorm
        x = torch.cat([unit_nodes, tile_nodes], dim=1)
        for i, gat in enumerate(self.grobal_gat_layers):
            residual = x
            x = gat(x, edge_index)
            if i > 0:
                x = F.relu(x + residual)

        # 出力層
        x = self.fc(x)
        return F.softmax(x, dim=1), x
