
import torch
import torch.nn.functional as F
from collections import Counter

from dataset import load_dataset
from modules import MLP, GCN

SEED = 42

# Five representative failure cases
NODE_IDS = [16052, 9254, 9708, 2924, 12241]

CLASS_NAMES = {
    0: "Television Shows",
    1: "Governmental Organisations",
    2: "Companies",
    3: "Politicians"
}

data = load_dataset(seed=SEED)

models = {}

for name, Model in [("mlp", MLP), ("gcn", GCN)]:
    model = Model(
        input_dim=128,
        hidden_dim=32,
        output_dim=4,
        dropout=0.5
    )

    model.load_state_dict(
        torch.load(
            f"{name}_best.pt",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()
    models[name] = model

# Generate predictions from both models
with torch.inference_mode():
    mlp_probs = F.softmax(
        models["mlp"](data.x), dim=1
    )

    gcn_probs = F.softmax(
        models["gcn"](data.x, data.edge_index),
        dim=1
    )

# Build undirected neighbourhoods
adjacency = [set() for _ in range(data.num_nodes)]

for source, target in data.edge_index.t().tolist():
    adjacency[source].add(target)
    adjacency[target].add(source)


def get_distribution(node_ids):
    """
    Count the ground-truth classes of selected nodes.

    Args:
        node_ids: List of node IDs to analyse.

    Returns:
        Counter containing the number of nodes in each class.
    """
    labels = [
        int(data.y[node].item())
        for node in node_ids
    ]
    return Counter(labels)


def print_distribution(distribution, total):
    """
    Print the class distribution as counts and percentages.

    Args:
        distribution: Number of nodes belonging to each class.
        total: Total number of nodes in the distribution.

    Returns:
        None
    """
    for class_id in range(4):
        count = distribution.get(class_id, 0)
        percentage = (
            100 * count / total if total else 0
        )

        print(
            f"  Class {class_id} "
            f"({CLASS_NAMES[class_id]}): "
            f"{count} ({percentage:.2f}%)"
        )


def inspect_node(node_id):
    """
    Analyse a selected node and its surrounding graph structure.

    Compares MLP and GCN predictions, direct and two-hop
    neighbour class distributions, and node feature similarity.

    Args:
        node_id: ID of the node to analyse.

    Returns:
        None
    """
    true_class = int(data.y[node_id].item())

    neighbours = sorted(adjacency[node_id])

    two_hop = set()
    for neighbour in neighbours:
        two_hop.update(adjacency[neighbour])

    # Exclude target and immediate neighbours
    two_hop.discard(node_id)
    two_hop.difference_update(neighbours)
    two_hop = sorted(two_hop)

    print("\n" + "=" * 65)
    print(f"NODE {node_id}")
    print("=" * 65)

    print(
        f"Actual Class: {true_class} "
        f"({CLASS_NAMES[true_class]})"
    )
    print(f"Test Node: {bool(data.test_mask[node_id])}")

    # Compare predictions
    for name, probs in [
        ("MLP", mlp_probs),
        ("GCN", gcn_probs)
    ]:
        prediction = int(probs[node_id].argmax())
        confidence = float(probs[node_id].max())

        print(f"\n{name} Prediction:")
        print(
            f"  Predicted Class: {prediction} "
            f"({CLASS_NAMES[prediction]})"
        )
        print(f"  Confidence: {confidence * 100:.2f}%")
        print(f"  Correct: {prediction == true_class}")

        print("  Class Probabilities:")
        for class_id in range(4):
            print(
                f"    Class {class_id}: "
                f"{probs[node_id, class_id].item() * 100:.2f}%"
            )

    # Direct neighbours
    print(f"\nDirect Neighbours: {len(neighbours)}")
    print("Neighbour Class Distribution:")

    neighbour_distribution = get_distribution(neighbours)
    print_distribution(
        neighbour_distribution, len(neighbours)
    )

    # Two-hop neighbours
    print(f"\nTwo-Hop Neighbours: {len(two_hop)}")
    print("Two-Hop Class Distribution:")

    two_hop_distribution = get_distribution(two_hop)
    print_distribution(
        two_hop_distribution, len(two_hop)
    )

    # Neighbour feature similarities
    print("\nFeature Similarity:")

    if neighbours:
        node_features = data.x[node_id].unsqueeze(0)
        neighbour_features = data.x[neighbours]

        similarities = F.cosine_similarity(
            node_features,
            neighbour_features,
            dim=1
        )

        print(
            "  Mean cosine similarity to direct neighbours:",
            f"{similarities.mean().item():.4f}"
        )

        print(
            "  Minimum similarity:",
            f"{similarities.min().item():.4f}"
        )

        print(
            "  Maximum similarity:",
            f"{similarities.max().item():.4f}"
        )

        print("\nExample Direct Neighbours:")

        for neighbour in neighbours[:10]:
            print(
                f"  Node {neighbour}: "
                f"Class {int(data.y[neighbour])}"
            )

    print("\nFirst 10 Standardised Features:")
    print([
        round(value, 4)
        for value in data.x[node_id, :10].tolist()
    ])


for node_id in NODE_IDS:
    inspect_node(node_id)
