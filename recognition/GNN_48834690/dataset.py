"""
Data loading and splitting for the Facebook Large Page-Page Network.

The graph is downloaded as the course-provided ``facebook.npz`` (a rebuild of the
GraphMining.ai file that PyG's FacebookPagePage loader used to fetch):
https://github.com/gayanku/facebook_page-page_dataset

Setting: transductive semi-supervised node classification. The whole graph
(all node features and edges) is visible during training, but only the labels
of training nodes are ever used in the loss. Validation/test labels are hidden
behind boolean masks and only read when computing metrics.
"""

import csv
import os
import urllib.request

import numpy as np
import torch
from torch_geometric.data import Data

NPZ_URL = "https://raw.githubusercontent.com/gayanku/facebook_page-page_dataset/main/facebook.npz"
TARGET_CSV_URL = ("https://raw.githubusercontent.com/gayanku/facebook_page-page_dataset/main/"
                  "facebook_large/musae_facebook_target.csv")
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# Label order is alphabetical. The dataset README lists a different order, but the
# per-label counts (6495, 6880, 5768, 3327) match SNAP's published class sizes
# for company, government, politician, tvshow respectively.
CLASS_NAMES = ["company", "government", "politician", "tvshow"]


def download_npz(root=DEFAULT_ROOT):
    """Download facebook.npz into ``root`` if it is not already there. Returns its path."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "facebook.npz")
    if not os.path.exists(path):
        print(f"Downloading {NPZ_URL} -> {path}")
        urllib.request.urlretrieve(NPZ_URL, path)
    return path


def load_page_info(root=DEFAULT_ROOT):
    """Return (page_names, page_types) for every node, from the raw MUSAE target CSV.

    Used for failure-case analysis. The npz also stores page names, but as a
    pickled object array; reading the plain CSV avoids unpickling a downloaded
    file. Rows are ordered by node id, matching the npz.
    """
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "musae_facebook_target.csv")
    if not os.path.exists(path):
        print(f"Downloading {TARGET_CSV_URL} -> {path}")
        urllib.request.urlretrieve(TARGET_CSV_URL, path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["id"]))
    return [r["page_name"] for r in rows], [r["page_type"] for r in rows]


def stratified_split(y, train_frac=0.7, val_frac=0.15, seed=42):
    """Split node indices into train/val/test boolean masks, stratified by class.

    Stratifying keeps the class proportions the same in every split, so
    per-class metrics on the test set are not skewed by an unlucky draw.
    """
    gen = torch.Generator().manual_seed(seed)
    n = y.size(0)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    for c in torch.unique(y):
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=gen)]
        n_train = int(train_frac * idx.numel())
        n_val = int(val_frac * idx.numel())
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True

    return train_mask, val_mask, test_mask


def load_graph(root=DEFAULT_ROOT, train_frac=0.7, val_frac=0.15, seed=42, standardise=True):
    """Load the graph as a PyG ``Data`` object with train/val/test masks attached.

    Features are standardised with the mean/std of the *training* nodes only,
    so no statistics from validation/test nodes leak into the inputs.
    """
    with np.load(download_npz(root), allow_pickle=False) as npz:
        x = torch.from_numpy(npz["X"]).float()
        y = torch.from_numpy(npz["y"]).long()
        edge_index = torch.from_numpy(np.vstack([npz["edges_x"], npz["edges_y"]])).long()

    train_mask, val_mask, test_mask = stratified_split(y, train_frac, val_frac, seed)

    if standardise:
        mean = x[train_mask].mean(dim=0)
        std = x[train_mask].std(dim=0) + 1e-8
        x = (x - mean) / std

    return Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


if __name__ == "__main__":
    # Quick data audit: sizes, class balance per split, and graph homophily.
    data = load_graph()
    print(data)
    print(f"Nodes: {data.num_nodes}, directed edges: {data.num_edges}, features: {data.num_node_features}")
    for name, mask in [("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)]:
        counts = torch.bincount(data.y[mask], minlength=len(CLASS_NAMES))
        print(f"{name:>5}: {int(mask.sum()):>6} nodes | " +
              ", ".join(f"{c}={int(k)}" for c, k in zip(CLASS_NAMES, counts)))
    src, dst = data.edge_index
    homophily = (data.y[src] == data.y[dst]).float().mean().item()
    print(f"Edge homophily (fraction of edges joining same-class pages): {homophily:.3f}")

    # Verify the label mapping and node ordering against the raw CSV's text labels.
    names, types = load_page_info()
    matches = sum(CLASS_NAMES[int(label)] == t for label, t in zip(data.y, types))
    print(f"Labels matching raw CSV page_type: {matches}/{data.num_nodes}")
