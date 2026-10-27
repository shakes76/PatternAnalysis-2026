"""Models: an MLP baseline and a multi-layer GCN (plain PyTorch).

Both take the same inputs, forward(x, adj), so one training loop can run
either. The MLP ignores adj; the GCN uses it to average over neighbours.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """One graph convolution, H' = A_hat @ (H W) + b"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj):
        return torch.sparse.mm(adj, self.weight(x)) + self.bias


class GCN(nn.Module):
    """Multi-layer GCN. The final layer outputs raw class scores (logits)."""
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = dropout

    def forward(self, x, adj, return_embedding=False):
        for layer in self.layers[:-1]:
            x = F.dropout(x, self.dropout, self.training)
            x = F.relu(layer(x, adj))
        # last hidden layer, for t-SNE/UMAP
        emb = x
        x = F.dropout(x, self.dropout, self.training)
        out = self.layers[-1](x, adj)
        return (out, emb) if return_embedding else out


class MLP(nn.Module):
    """Baseline with the same sizes and dropout as the GCN, but no graph:
    each node is classified from its own features only."""

    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = dropout

    def forward(self, x, adj=None, return_embedding=False):
        for layer in self.layers[:-1]:
            x = F.dropout(x, self.dropout, self.training)
            x = F.relu(layer(x))
        emb = x
        x = F.dropout(x, self.dropout, self.training)
        out = self.layers[-1](x)
        return (out, emb) if return_embedding else out