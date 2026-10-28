
import torch
from torch import nn

from torch_geometric.nn import GCNConv

class MLP(nn.Module):
    """
    Multi-Layer Perceptron baseline for node classification.

    The MLP uses only node features and does not consider
    graph connections.
    """

    def __init__(self, input_dim=128, hidden_dim=32,
                 output_dim=4, dropout=0.5):
        """
        Initialise the MLP architecture.

        Args:
            input_dim: Number of input node features.
            hidden_dim: Number of neurons in the hidden layer.
            output_dim: Number of classification categories.
            dropout: Probability of dropping hidden activations.
        """
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Perform a forward pass through the MLP.

        Args:
            x: Node features [num_nodes, input_dim].

        Returns:
            Class logits [num_nodes, output_dim].
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x



class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network for node classification.

    The GCN uses node features and graph connections to
    classify Facebook pages into four categories.
    """

    def __init__(self, input_dim=128, hidden_dim=32,
                 output_dim=4, dropout=0.5):
        """
        Initialise the GCN architecture.

        Args:
            input_dim: Number of input node features.
            hidden_dim: Number of hidden features.
            output_dim: Number of classification categories.
            dropout: Probability of dropping hidden activations.
        """
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """
        Perform a forward pass through the GCN.

        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Graph connections [2, num_edges].

        Returns:
            Class logits [num_nodes, output_dim].
        """
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return x

    def get_embeddings(self, x, edge_index):
        """
        Return the learned hidden node representations.

        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Graph connections [2, num_edges].

        Returns:
            Hidden node embeddings [num_nodes, hidden_dim].
        """
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        return x