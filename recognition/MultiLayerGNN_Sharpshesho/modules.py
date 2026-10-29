"""Neural-network models for Facebook page classification.

This module contains a feature-only MLP baseline and a two-layer Graph
Convolutional Network. Both models return raw class scores for use with
CrossEntropyLoss.

References:
    Architecture: T. N. Kipf and M. Welling, Semi-Supervised Classification
    with Graph Convolutional Networks, ICLR 2017.
    https://arxiv.org/abs/1609.02907
    Graph layers use the PyTorch Geometric GCNConv library operator:
    https://pytorch-geometric.readthedocs.io/en/2.8.1/generated/torch_geometric.nn.conv.GCNConv.html

AI assistance:
    OpenAI Codex assisted in generating, revising and debugging this code.
    The README section "Artificial Intelligence Usage Disclosure" documents
    the scope of assistance and the data audits, smoke runs, complete training
    runs and saved-output checks used to verify the implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class MLPBaseline(nn.Module):
    """Classify each Facebook page using only its node features."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(
            in_channels,
            hidden_channels,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(
            p=dropout_p,
        )
        self.linear2 = nn.Linear(
            hidden_channels,
            out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one class-score vector for every page."""

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class TwoLayerGCN(nn.Module):
    """Classify pages using node features and graph connections."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(
            p=dropout_p,
        )
        self.conv2 = GCNConv(
            hidden_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return one class-score vector for every graph node."""

        x = self.conv1(
            x,
            edge_index,
        )
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(
            x,
            edge_index,
        )

        return x
