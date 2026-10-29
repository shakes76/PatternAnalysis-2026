"""Pure-PyTorch MLP and two-layer GCN; no labels enter either forward method.

GCN normalization follows Kipf and Welling, ICLR 2017:
https://arxiv.org/abs/1609.02907
"""

import torch
from torch import nn
from torch.nn import functional as F


def normalized_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Construct D^-1/2 (A+I) D^-1/2 with binary edges and one loop per node."""
    edges = edge_index[:, edge_index[0] != edge_index[1]]
    bidirectional = torch.cat((edges, edges.flip(0)), dim=1)
    unique = torch.sparse_coo_tensor(
        bidirectional, torch.ones(bidirectional.shape[1], device=edges.device),
        (num_nodes, num_nodes), check_invariants=True,
    ).coalesce().indices()
    nodes = torch.arange(num_nodes, device=edges.device)
    indices = torch.cat((unique, torch.stack((nodes, nodes))), dim=1)
    degree = torch.bincount(indices[0], minlength=num_nodes).float()
    inverse = degree.rsqrt()
    values = inverse[indices[0]] * inverse[indices[1]]
    return torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_nodes), check_invariants=True,
    ).coalesce()


class MLP(nn.Module):
    """Feature-only baseline with the same widths as the GCN."""

    def __init__(self, inputs=128, hidden=64, classes=4, dropout=0.5):
        super().__init__()
        self.first = nn.Linear(inputs, hidden)
        self.second = nn.Linear(hidden, classes)
        self.dropout = dropout

    def forward(self, x, adjacency=None):
        embedding = F.relu(self.first(x))
        logits = self.second(F.dropout(embedding, p=self.dropout, training=self.training))
        return logits, embedding


class GCN(nn.Module):
    """Two graph-convolution layers, with biases added after graph aggregation."""

    def __init__(self, inputs=128, hidden=64, classes=4, dropout=0.5):
        super().__init__()
        self.first = nn.Linear(inputs, hidden, bias=False)
        self.second = nn.Linear(hidden, classes, bias=False)
        self.first_bias = nn.Parameter(torch.zeros(hidden))
        self.second_bias = nn.Parameter(torch.zeros(classes))
        self.dropout = dropout

    def forward(self, x, adjacency):
        embedding = F.relu(torch.sparse.mm(adjacency, self.first(x)) + self.first_bias)
        hidden = F.dropout(embedding, p=self.dropout, training=self.training)
        logits = torch.sparse.mm(adjacency, self.second(hidden)) + self.second_bias
        return logits, embedding


def make_model(name, inputs=128, hidden=64, classes=4, dropout=0.5):
    constructors = {"mlp": MLP, "gcn": GCN}
    if name not in constructors:
        raise ValueError(f"Unknown model: {name}")
    return constructors[name](inputs, hidden, classes, dropout)
