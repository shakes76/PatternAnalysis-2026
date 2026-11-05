import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.utils import remove_self_loops

NPZ_PATH = "data/FacebookPagePage/raw/facebook.npz"
CLASS_NAMES = ["tvshow", "government", "company", "politician"]


def load_graph(path=NPZ_PATH):
    """Load the Facebook Page-Page graph from the .npz file."""
    npz = np.load(path, allow_pickle=True) # load npz archive

    edge_index = torch.tensor(np.vstack([npz["edges_x"], npz["edges_y"]]), dtype=torch.long)  # format edge components into 2xN edge index
    edge_index, _ = remove_self_loops(edge_index) # drop existing self-loops; GCNLayer adds exactly one per node

    return Data( # build and return the primary PyG Data object
        x=torch.tensor(npz["X"], dtype=torch.float32), # each row represents a fb page
        edge_index=edge_index,
        y=torch.tensor(npz["y"], dtype=torch.long), # each col represents the class of the corresponding fb page
    )


def make_masks(data, train=0.1, val=0.1, seed=42):
    """Stratified random node split (transductive setting)."""
    idx = np.arange(data.num_nodes) # list of node numbers; 0 to 
    y = data.y.numpy()


    train_idx, rest = train_test_split(idx, train_size=train, stratify=y, random_state=seed) # take 10% of nodes for training, keep the other 90% in rest.

    val_idx, test_idx = train_test_split( # split rest in half into validation and test (20% of all nodes each)
        rest, train_size=val / (1 - train), stratify=y[rest], random_state=seed
    )

    # create boolean masks for train, validation, and test sets
    for name, ids in [("train_mask", train_idx), ("val_mask", val_idx), ("test_mask", test_idx)]:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[ids] = True
        setattr(data, name, mask)
    return data


if __name__ == "__main__":
    # Runs only with 'python dataset.py': prints the data audit for the feasibility page
    data = make_masks(load_graph())
    print(data)
    print("Nodes:", data.num_nodes, "| Edges (directed):", data.edge_index.size(1))
    print("Feature dim:", data.x.size(1))
    # how many pages are in each category

    counts = torch.bincount(data.y)
    for i, c in enumerate(counts):
        print(f"Class {i} ({CLASS_NAMES[i]}): {c.item()} ({100 * c.item() / data.num_nodes:.1f}%)")

    # how many nodes are connected (non-isolated), and isolated (not the source of any link)
    connected = torch.unique(data.edge_index[0]).numel()
    print("Isolated nodes:", data.num_nodes - connected)

    # Number of nodes in each split
    print("Train/val/test:", data.train_mask.sum().item(), data.val_mask.sum().item(), data.test_mask.sum().item())