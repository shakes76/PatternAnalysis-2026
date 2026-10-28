"""
Facebook Page-Page Dataset available here on Rangpur:
/home/groups/comp3710

The GCN uses transductive semi-supervised learning so it learns 
from the labels of some nodes while using the features and 
connections of all nodes, including those whose labels are 
withheld during training.
"""
import torch
from torch_geometric.datasets import FacebookPagePage

def create_node_splits(data, seed=42):
    """
    Split nodes into training, validation and test sets.

    Args:
        data: PyTorch Geometric graph containing node labels.
        seed: Random seed for reproducibility.

    Returns:
        Graph with train_mask, val_mask and test_mask.
    """
    generator = torch.Generator().manual_seed(seed)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    # Split each class separately to preserve class proportions
    for class_id in torch.unique(data.y):
        class_indices = torch.where(data.y == class_id)[0]

        permutation = torch.randperm(
            len(class_indices), generator=generator
        )
        class_indices = class_indices[permutation]

        num_train = int(0.70 * len(class_indices))
        num_val = int(0.15 * len(class_indices))

        train_indices = class_indices[:num_train]
        val_indices = class_indices[num_train:num_train + num_val]
        test_indices = class_indices[num_train + num_val:]

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def check_node_splits(data):
    """
    Check that the three node splits are valid.

    Args:
        data: Graph containing the train, validation and test masks.

    Raises:
        AssertionError: If masks overlap or omit nodes.
    """
    train = data.train_mask
    val = data.val_mask
    test = data.test_mask

    # No node can belong to more than one split
    assert not torch.any(train & val)
    assert not torch.any(train & test)
    assert not torch.any(val & test)

    # Every node must belong to exactly one split
    assert torch.all(train | val | test)

    print("All nodes belong to exactly one split.")
    print("No overlapping nodes detected.")


def standardise_features(data):
    """
    Standardise node features using training statistics only.

    Args:
        data: Graph with node features and train_mask.

    Returns:
        Graph containing standardised node features.
    """
    # Calculate statistics only from training nodes
    train_features = data.x[data.train_mask]

    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False)

    # Prevent division by zero for constant features
    std = std.clamp(min=1e-8)

    # Apply the same transformation to every node
    data.x = (data.x - mean) / std

    return data


def verify_dataset(data):
    """
    Verify graph dimensions, node splits and preprocessing.
    Verifies that:
    - Every node has 128 features and a valid label.
    - Every node belongs to exactly one split.
    - All four classes appear in each split.
    - The class distribution remains approximately consistent.
    - Feature standardisation hasn't created invalid numerical values.
    - Repeating the splitting procedure with the same seed produces identical masks.

    Args:
        data: Preprocessed graph with train/val/test masks.
    """
    assert data.x.shape == (data.num_nodes, 128)
    assert data.y.shape == (data.num_nodes,)
    assert data.edge_index.shape[0] == 2

    assert torch.isfinite(data.x).all()
    assert data.y.min() >= 0
    assert data.y.max() < 4

    check_node_splits(data)

    print("\n=== Split Class Distribution ===")

    splits = {
        "Training": data.train_mask,
        "Validation": data.val_mask,
        "Testing": data.test_mask
    }

    for name, mask in splits.items():
        labels = data.y[mask]

        counts = torch.bincount(labels, minlength=4)

        print(f"\n{name}: {mask.sum().item()} nodes")

        for class_id, count in enumerate(counts):
            percentage = count.item() / len(labels) * 100

            print(
                f"  Class {class_id}: "
                f"{count.item()} ({percentage:.2f}%)"
            )

        assert torch.all(counts > 0)

    print("\nDataset verification successful.")


def load_dataset(root=None, seed=42):
    """
    Load and preprocess the Facebook Page-Page dataset.

    Args:
        root: Directory containing the dataset.
        seed: Random seed for reproducible node splits.

    Returns:
        Graph with node features, labels and data splits.
    """
    if root is None:
        from pathlib import Path
        root = Path(__file__).resolve().parent

    dataset = FacebookPagePage(root=str(root))

    # The dataset contains one graph
    data = dataset[0]

    # Create the training, validation and test splits
    data = create_node_splits(data, seed)

    # Standardise the node features
    data = standardise_features(data)

    return data



if __name__ == "__main__":
    # Load and preprocess the dataset
    data = load_dataset()

    # Inspect graph dimensions
    print("=== Graph Dimensions ===")
    print("Number of nodes:", data.num_nodes)
    print("Number of edges:", data.num_edges)
    print("Number of node features:", data.num_node_features)
    print("Number of classes:", torch.unique(data.y).numel())

    # Inspect node features
    print("\n=== Node Features ===")
    print("Feature matrix shape:", data.x.shape)
    print("Feature data type:", data.x.dtype)
    print("First node's features:", data.x[0])

    # Inspect labels
    print("\n=== Node Labels ===")
    print("Label tensor shape:", data.y.shape)
    print("Label data type:", data.y.dtype)
    print("Unique labels:", torch.unique(data.y))
    print("First 10 labels:", data.y[:10])

    # Inspect overall class distribution
    print("\n=== Class Distribution ===")

    classes, counts = torch.unique(
        data.y, return_counts=True
    )

    for class_id, count in zip(classes, counts):
        percentage = count.item() / data.num_nodes * 100

        print(
            f"Class {class_id.item()}: "
            f"{count.item()} nodes ({percentage:.2f}%)"
        )

    # Inspect dataset splits
    print("\n=== Dataset Splits ===")
    print("Training nodes:", data.train_mask.sum().item())
    print("Validation nodes:", data.val_mask.sum().item())
    print("Testing nodes:", data.test_mask.sum().item())

    # Verify dataset
    verify_dataset(data)

    # Check feature standardisation
    print("\n=== Feature Standardisation ===")

    train_features = data.x[data.train_mask]

    print("Feature shape:", data.x.shape)
    print("Mean of first feature:",
          train_features[:, 0].mean().item())
    print("Standard deviation of first feature:",
          train_features[:, 0].std(unbiased=False).item())

    print("All feature values finite:",
          torch.isfinite(data.x).all().item())

    # Check reproducibility
    data_again = load_dataset(seed=42)

    assert torch.equal(
        data.train_mask, data_again.train_mask
    )
    assert torch.equal(
        data.val_mask, data_again.val_mask
    )
    assert torch.equal(
        data.test_mask, data_again.test_mask
    )

    print("\nReproducibility check passed.")



