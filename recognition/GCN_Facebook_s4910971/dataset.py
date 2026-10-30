"""Load the graph and create node masks."""

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


DATA_FILE = Path(__file__).resolve().parent / "data" / "facebook.npz"


def make_mask(node_ids, total_nodes):
    """Mark nodes in one split."""
    mask = torch.zeros(total_nodes, dtype=torch.bool)
    mask[node_ids] = True
    return mask


def load_graph(data_path=DATA_FILE, seed=42):
    """Load the graph with a 70/10/20 split."""
    with np.load(data_path, allow_pickle=False) as file:
        x = torch.tensor(file["X"], dtype=torch.float32)
        y = torch.tensor(file["y"], dtype=torch.long)
        edges = torch.tensor(np.stack([file["edges_x"], file["edges_y"]]), dtype=torch.long)

    node_ids = np.arange(len(y))
    labels = y.numpy()

    # Keep the fixed test partition.
    train_ids, remaining_ids = train_test_split(
        node_ids, test_size=0.4, stratify=labels, random_state=seed
    )
    old_val_ids, test_ids = train_test_split(
        remaining_ids, test_size=0.5, stratify=labels[remaining_ids], random_state=seed
    )

    # Complete the 70/10/20 split.
    old_val_ids = np.sort(old_val_ids)
    extra_train_ids, val_ids = train_test_split(
        old_val_ids, test_size=0.5, stratify=labels[old_val_ids], random_state=seed
    )
    train_ids = np.concatenate([train_ids, extra_train_ids])

    # Keep all edges; use masks to select training labels.
    return Data(
        x=x,
        edge_index=edges,
        y=y,
        train_mask=make_mask(train_ids, len(y)),
        val_mask=make_mask(val_ids, len(y)),
        test_mask=make_mask(test_ids, len(y)),
    )
