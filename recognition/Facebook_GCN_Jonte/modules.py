"""PyTorch model components for the Facebook node-classification project.

GCN architecture: Kipf & Welling (2017), https://arxiv.org/abs/1609.02907.
Layer implementation: PyG 2.6.1 GCNConv public API,
https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.GCNConv.html.
"""

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class MLP(nn.Module):
    """Feature-only baseline: linear -> ReLU -> dropout -> linear logits.

    Each row is classified independently. No edges or labels enter forward().
    Return unnormalised logits because CrossEntropyLoss applies log-softmax.
    """

    def __init__(
        self, in_features: int = 128, hidden_features: int = 64,
        num_classes: int = 4, dropout: float = 0.5,
    ):
        super().__init__()
        if min(in_features, hidden_features) <= 0 or num_classes < 2:
            raise ValueError("Feature dimensions must be positive; need at least two classes")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map an [N, in_features] feature tensor to [N, num_classes] logits."""
        return self.layers(features)


class GCN(nn.Module):
    """Two-layer transductive GCN; only features and edges enter forward().

    Cached normalization is valid only for this fixed graph. PyG removes the
    supplied loops and adds one per node for unweighted COO input. Neither the
    raw duplicate loops nor graph tensors are saved in the state dictionary.
    """

    def __init__(
        self, in_features: int = 128, hidden_features: int = 64,
        num_classes: int = 4, dropout: float = 0.5,
    ):
        super().__init__()
        if min(in_features, hidden_features) <= 0 or num_classes < 2:
            raise ValueError("Feature dimensions must be positive; need at least two classes")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        self.conv1 = GCNConv(in_features, hidden_features, cached=True,
                             normalize=True, add_self_loops=True)
        self.conv2 = GCNConv(hidden_features, num_classes, cached=True,
                             normalize=True, add_self_loops=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return [N, num_classes] logits using two hops of fixed-graph context."""
        hidden = self.dropout(torch.relu(self.conv1(features, edge_index)))
        return self.conv2(hidden, edge_index)
