
import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data


def load_facebook(data_dir="data/facebook_uci/facebook_large", seed=42):
    path = Path(data_dir)

    edges = pd.read_csv(path / "musae_facebook_edges.csv")
    targets = pd.read_csv(path / "musae_facebook_target.csv")

    with open(path / "musae_facebook_features.json") as f:
        features = json.load(f)

    # Map original node IDs to contiguous tensor indices.
    node_ids = targets["id"].astype(int).tolist()
    id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    # Build binary node-feature matrix.
    feature_dim = max(
        feature_id
        for values in features.values()
        for feature_id in values
    ) + 1

    x = torch.zeros((len(node_ids), feature_dim), dtype=torch.float32)

    for node_id, feature_ids in features.items():
        i = id_to_idx[int(node_id)]
        x[i, feature_ids] = 1.0

    # Encode the four page categories.
    classes = sorted(targets["page_type"].unique())
    class_to_idx = {name: i for i, name in enumerate(classes)}

    y = torch.tensor(
        [class_to_idx[name] for name in targets["page_type"]],
        dtype=torch.long
    )

    # Build graph edges using the same node index mapping.
    edge_pairs = edges.iloc[:, :2].to_numpy()

    edge_index = torch.tensor(
        [
            [id_to_idx[int(u)], id_to_idx[int(v)]]
            for u, v in edge_pairs
        ],
        dtype=torch.long
    ).T.contiguous()

    # Convert the undirected graph to bidirectional edges.
    edge_index = torch.cat(
        [edge_index, edge_index.flip(0)], dim=1
    )

    # Reproducible 60/20/20 split.
    n = len(node_ids)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n, generator=generator)

    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[permutation[:n_train]] = True
    val_mask[permutation[n_train:n_train + n_val]] = True
    test_mask[permutation[n_train + n_val:]] = True

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )


if __name__ == "__main__":
    graph = load_facebook()

    print(graph)
    print("Nodes:", graph.num_nodes)
    print("Edges:", graph.num_edges)
    print("Features:", graph.num_node_features)
    print("Classes:", graph.y.unique().tolist())
    print("Training nodes:", int(graph.train_mask.sum()))
    print("Validation nodes:", int(graph.val_mask.sum()))
    print("Testing nodes:", int(graph.test_mask.sum()))

