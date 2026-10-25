"""modules.py: Node classifiers for Facebook Page-Page: an MLP baseline and a multi-layer GCN.

Author: 49250523 (COMP3710 Pattern Recognition project, 2026)

Both models share one interface, ``forward(x, edge_index) -> logits``, so the
training and prediction scripts can use either without knowing which it is.
Only PyTorch and PyTorch Geometric are used here (no NumPy), as the spec requires.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class NodeClassifier(nn.Module):
    """Common interface for every model in this project.

    Subclasses implement ``embed``. The final layer is shared logic: it maps the
    embedding to one score (logit) per class.
    """

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return the hidden representation before the final layer.

        Args:
            x: Node features, shape ``(num_nodes, in_dim)``.
            edge_index: Undirected edges, shape ``(2, num_edges)``.

        Returns:
            Node embeddings, shape ``(num_nodes, hidden_dim)``.
        """
        raise NotImplementedError

    def classify(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Map embeddings to class logits.

        Args:
            h: Node embeddings from ``embed``.
            edge_index: Undirected edges, shape ``(2, num_edges)``.

        Returns:
            Logits, shape ``(num_nodes, num_classes)``.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return class logits (apply softmax outside to get probabilities).

        Args:
            x: Node features, shape ``(num_nodes, in_dim)``.
            edge_index: Undirected edges, shape ``(2, num_edges)``.

        Returns:
            Logits, shape ``(num_nodes, num_classes)``.
        """
        return self.classify(self.embed(x, edge_index), edge_index)


class MLP(NodeClassifier):
    """Baseline: classifies each page from its own features and ignores the graph.

    Same depth, width and dropout as the 2-layer GCN, so any gain of the GCN
    over this model comes from the edges.

    Args:
        in_dim: Number of input features per node.
        hidden_dim: Width of the hidden layer.
        num_classes: Number of output classes.
        dropout: Probability of zeroing each hidden unit during training.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.5):
        """Create the two linear layers."""
        super().__init__()
        self.hidden = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return the hidden layer output; ``edge_index`` is not used."""
        return F.relu(self.hidden(x))

    def classify(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return logits from the output layer; ``edge_index`` is not used."""
        return self.out(F.dropout(h, self.dropout, self.training))


class GCN(NodeClassifier):
    """Multi-layer graph convolutional network (Kipf & Welling, 2017).

    Each layer computes ``ReLU(D^-1/2 (A + I) D^-1/2 H W)``: every page takes a
    degree-weighted average of its own and its neighbours' features, then
    applies a shared weight matrix. ``L`` layers reach ``L`` hops.

    Layers between hidden layers of equal width add a residual connection
    (``h = h + conv(h)``), so a deeper model can keep each page's own signal
    instead of averaging it away (over-smoothing). A 2-layer GCN has no such
    layer, so it matches the feasibility smoke test exactly.

    Args:
        in_dim: Number of input features per node.
        hidden_dim: Width of every hidden layer.
        num_classes: Number of output classes.
        num_layers: Number of GCN layers (at least 2).
        dropout: Probability of zeroing each hidden unit during training.

    Raises:
        ValueError: If ``num_layers`` is less than 2.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        """Create the GCN layers."""
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be at least 2, got {num_layers}")
        self.first = GCNConv(in_dim, hidden_dim)
        self.middle = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2))
        self.last = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return the output of the last hidden layer."""
        h = F.relu(self.first(x, edge_index))
        for conv in self.middle:
            h = h + F.relu(conv(F.dropout(h, self.dropout, self.training), edge_index))
        return h

    def classify(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return logits from the final GCN layer."""
        return self.last(F.dropout(h, self.dropout, self.training), edge_index)


MODELS: dict[str, type[NodeClassifier]] = {"mlp": MLP, "gcn": GCN}


def build_model(name: str, in_dim: int, num_classes: int, **kwargs: object) -> NodeClassifier:
    """Create a model by name.

    Args:
        name: ``"mlp"`` or ``"gcn"``.
        in_dim: Number of input features per node.
        num_classes: Number of output classes.
        **kwargs: Passed to the model, e.g. ``hidden_dim``, ``num_layers``, ``dropout``.

    Returns:
        The model, in training mode.

    Raises:
        ValueError: If ``name`` is not a known model.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    if name == "mlp":
        kwargs.pop("num_layers", None)  # the baseline always has two layers
    kwargs.setdefault("hidden_dim", 64)
    return MODELS[name](in_dim=in_dim, num_classes=num_classes, **kwargs)
