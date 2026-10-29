"""
Model definitions: a multi-layer MLP (baseline) and a multi-layer GCN (main
model), each with a configurable number of layers.

The two models are identical except that the GCN averages each node with its
neighbours in every layer, so any difference in results comes from the graph.
Depth was chosen on the validation set by a grid search over layers, weight
decay, dropout and hidden size: 3 layers for the MLP, 4 for the GCN.

Each GCN layer computes  H' = Â · H · W , where
    H · W   is an ordinary linear layer applied to every node, and
    Â       averages each node with its neighbours, using the adjacency
            matrix with self-loops added and degree normalisation:
            Â = D^(-1/2) (A + I) D^(-1/2).
Reference: T. N. Kipf and M. Welling, "Semi-supervised classification with
graph convolutional networks", ICLR 2017.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def layer_sizes(in_dim, hidden, n_classes, num_layers):
    """Input/output size of every layer, e.g. 3 layers: 128->64, 64->64, 64->4."""
    dims = [in_dim] + [hidden] * (num_layers - 1) + [n_classes]
    return list(zip(dims[:-1], dims[1:]))


class MLP(torch.nn.Module):
    """Multi-layer perceptron baseline: classifies each node from its own
    features only, ignoring all edges.

    Input:  x [N, in_dim] node features, edge_index (accepted but ignored, so
            both models can be called the same way in train.py)
    Output: [N, n_classes] raw class scores (logits) for every node
    """

    def __init__(self, in_dim=128, hidden=64, n_classes=4, num_layers=3,
                 dropout=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(d_in, d_out)
            for d_in, d_out in layer_sizes(in_dim, hidden, n_classes, num_layers))
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        # Same steps as the GCN below, minus the neighbour averaging
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)


class GCN(torch.nn.Module):
    """Multi-layer graph convolutional network for node classification.

    Input:  x [N, in_dim] node features, edge_index [2, E] edges
    Output: [N, n_classes] raw class scores (logits) for every node
    """

    def __init__(self, in_dim=128, hidden=64, n_classes=4, num_layers=4,
                 dropout=0.5):
        super().__init__()
        # cached=True: the graph never changes (transductive setting), so the
        # normalised adjacency Â is computed once and reused every epoch
        self.layers = torch.nn.ModuleList(
            GCNConv(d_in, d_out, cached=True)
            for d_in, d_out in layer_sizes(in_dim, hidden, n_classes, num_layers))
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Each hidden layer: mix every page with its neighbours (one more hop
        # each time), apply ReLU (the "bend"), then dropout (training only)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Last layer: mix once more and produce one score per class
        return self.layers[-1](x, edge_index)
