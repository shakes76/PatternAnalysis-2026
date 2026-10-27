"""Data loading, graph preprocessing and leakage-free splitting."""
import os
from types import SimpleNamespace
import numpy as np
import torch


def load_graph(root="data"):
    """Load the Facebook Page-Page graph from facebook.npz."""
    path = os.path.join(root, "FacebookPagePage", "raw", "facebook.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Place facebook.npz at {path}")
    raw = np.load(path)  # the pickled 'page_name' array is never read
    x = torch.from_numpy(raw["X"]).float()                    # (22470, 128)
    y = torch.from_numpy(raw["y"]).long()                     # (22470,)
    edge_index = torch.from_numpy(
        np.stack([raw["edges_x"], raw["edges_y"]])
    ).long()                                                  # (2, 342004), both directions
    data = SimpleNamespace(
        x=x, y=y, edge_index=edge_index,
        num_nodes=x.size(0), num_node_features=x.size(1),
    )
    return data, int(y.max()) + 1


def normalized_adjacency(edge_index, num_nodes):
    """Return D^-1/2 (A + I) D^-1/2 as a sparse tensor (Kipf & Welling, 2017)."""
    loops = torch.arange(num_nodes)
    row = torch.cat([edge_index[0], loops])
    col = torch.cat([edge_index[1], loops])
    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]), torch.ones(row.size(0)), (num_nodes, num_nodes)
    ).coalesce()
    idx = adj.indices()
    vals = torch.ones(idx.size(1))              # binary weights even if edges repeat
    deg = torch.zeros(num_nodes).scatter_add_(0, idx[0], vals)
    d_inv_sqrt = deg.pow(-0.5)
    vals = d_inv_sqrt[idx[0]] * vals * d_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes)).coalesce()


def stratified_split(y, train_frac=0.6, val_frac=0.2, seed=0):
    """Random node split that keeps class proportions the same in each part.

    Returns three disjoint index tensors (train, val, test).
    """
    g = torch.Generator().manual_seed(seed)
    train, val, test = [], [], []
    for c in torch.unique(y):
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        n_train = int(train_frac * idx.numel())
        n_val = int(val_frac * idx.numel())
        train.append(idx[:n_train])
        val.append(idx[n_train:n_train + n_val])
        test.append(idx[n_train + n_val:])
    return torch.cat(train), torch.cat(val), torch.cat(test)


def get_data(root="data", train_frac=0.6, val_frac=0.2, seed=0):
    """Everything the training script needs, in one dict."""
    data, num_classes = load_graph(root)
    adj = normalized_adjacency(data.edge_index, data.num_nodes)
    train_idx, val_idx, test_idx = stratified_split(data.y, train_frac, val_frac, seed)
    return {
        "x": data.x, "y": data.y, "adj": adj, "edge_index": data.edge_index,
        "num_classes": num_classes,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
    }


if __name__ == "__main__":
    d = get_data()
    print("nodes:", d["x"].size(0), "| features:", d["x"].size(1),
          "| classes:", d["num_classes"])
    print("train/val/test:", len(d["train_idx"]), len(d["val_idx"]), len(d["test_idx"]))