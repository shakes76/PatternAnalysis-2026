"""MLP baseline using node features."""

import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Three linear layers with two equal-width hidden layers."""

    def __init__(self, input_dim=128, hidden_dim=64, num_classes=4, dropout=0.2):
        """Create three linear layers."""
        super().__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """Return class scores from node features."""
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc3(x)
