"""
modules stuff
"""
import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Two-layer graph convolutional network for node classification."""

    def __init__(self, input_dim, hidden_dim=64, num_classes=4, dropout=0.5):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # First layer combines node features with graph neighbours.
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        # Second layer produces one class score per node.
        x = self.conv2(x, edge_index)
        return x


class MLP(nn.Module):
    """Feature-only baseline without graph message passing."""

    def __init__(self, input_dim, hidden_dim=64, num_classes=4, dropout=0.5):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Small synthetic graph for checking model output dimensions.
    x = torch.randn(10, 4714)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    ], dtype=torch.long)

    gcn = GCN(input_dim=4714)
    mlp = MLP(input_dim=4714)

    gcn.eval()
    mlp.eval()

    with torch.no_grad():
        gcn_output = gcn(x, edge_index)
        mlp_output = mlp(x)

    assert gcn_output.shape == (10, 4)
    assert mlp_output.shape == (10, 4)

    print("GCN output shape:", gcn_output.shape)
    print("MLP output shape:", mlp_output.shape)
    print("GCN parameters:", sum(p.numel() for p in gcn.parameters()))
    print("MLP parameters:", sum(p.numel() for p in mlp.parameters()))
    print("Model smoke tests passed.")
