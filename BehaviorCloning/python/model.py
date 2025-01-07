import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TileEmbedding(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super(TileEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

    def forward(self, tile):
        tile = F.relu(self.fc(tile))
        output, _ = self.attention(tile, tile, tile)
        return F.relu(output)

class GATActor(nn.Module):
    def __init__(self, input_dim=3+128, hidden_dim=256, output_dim=5):
        super(GATActor, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, concat=True, add_self_loops=True)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True, add_self_loops=True)
        self.fc = nn.Linear(hidden_dim * 4, output_dim)

    def forward(self, x):
        x = F.relu(self.gat1(x))
        x = F.relu(self.gat2(x))
        x = self.fc(x)
        return F.softmax(x, dim=1), x
