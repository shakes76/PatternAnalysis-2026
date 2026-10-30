"""
Model components: an MLP baseline and a multi-layer Graph Convolutional Network.

Both models share the same interface, ``forward(x, edge_index) -> logits`` and
``embed(x, edge_index) -> hidden features``, so the same training, evaluation
and visualisation code can run on either. The MLP simply ignores ``edge_index``,
which makes it the "features only, no graph" baseline.

GCN reference: T. N. Kipf and M. Welling, "Semi-Supervised Classification with
Graph Convolutional Networks", ICLR 2017.
"""

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class MLP(nn.Module):
    """Baseline: classifies each page from its own 128-dim features only."""

    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=3, dropout=0.5):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(dims[-1], num_classes)
        self.dropout = nn.Dropout(dropout)

    def embed(self, x, edge_index=None):
        """Return the last hidden layer's activations (used for t-SNE plots)."""
        for layer in self.hidden:
            x = self.dropout(torch.relu(layer(x)))
        return x

    def forward(self, x, edge_index=None):
        return self.out(self.embed(x, edge_index))


class GCN(nn.Module):
    """Multi-layer GCN: each layer averages a node's features with its neighbours'.

    One GCNConv computes H' = D^-1/2 (A + I) D^-1/2 H W, i.e. a degree-normalised
    sum over the node and its neighbours followed by a linear map. Stacking
    ``num_layers`` of them lets each prediction see the ``num_layers``-hop
    neighbourhood, which exploits the graph's high homophily (~0.89 of edges
    join pages of the same class).
    """

    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=3, dropout=0.5):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        self.hidden = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = GCNConv(dims[-1], num_classes)
        self.dropout = nn.Dropout(dropout)

    def embed(self, x, edge_index):
        """Return the last hidden layer's node embeddings (used for t-SNE plots)."""
        for conv in self.hidden:
            x = self.dropout(torch.relu(conv(x, edge_index)))
        return x

    def forward(self, x, edge_index):
        return self.out(self.embed(x, edge_index), edge_index)


MODELS = {"mlp": MLP, "gcn": GCN}


def build_model(name, in_dim, num_classes, hidden_dim=64, num_layers=3, dropout=0.5):
    """Construct a model by name ('mlp' or 'gcn') with shared hyperparameters."""
    return MODELS[name](in_dim, hidden_dim, num_classes, num_layers, dropout)


def count_parameters(model):
    """Number of trainable parameters (reported in the resource profiling table)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
