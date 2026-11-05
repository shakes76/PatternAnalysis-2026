"""Compare test accuracy across graph degree and neighbour-label agreement"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import load_graph
from utils import neighbour_agreement, node_degrees


PREDICTION_DIR = Path("outputs")
RESULTS_PATH = Path("results/graph_context_accuracy.csv")
PLOT_PATH = Path("figures/graph_context_accuracy.png")
MODELS = ("mlp", "gcn")


def degree_groups(degree: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Split test nodes into approximately equal-sized degree groups"""
    # use degree quantiles so each group has a similar number of nodes
    quantiles = np.unique(np.quantile(degree, np.linspace(0, 1, 6)))
    if quantiles.size < 2:
        return np.zeros(degree.size, dtype=int), [f"degree {int(quantiles[0])}"]

    # assign each degree to the interval between quantile boundaries
    groups = np.digitize(degree, quantiles[1:-1], right=True)
    labels = [f"≤{edge:.0f}" for edge in quantiles[1:]]
    labels[0] = f"≤{quantiles[1]:.0f}"
    for index in range(1, len(labels)):
        labels[index] = f">{quantiles[index]:.0f}–≤{quantiles[index + 1]:.0f}"
    return groups, labels


def agreement_groups(agreement: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Group nodes by the fraction of neighbours sharing their true class"""
    # use fixed ranges to make neighbour agreement easy to interpret
    boundaries = np.array([0.0, 0.5, 0.75, 0.9, 0.99, 1.0])
    groups = np.digitize(agreement, boundaries[1:-1], right=True)
    labels = [
        "≤0.50",
        ">0.50–≤0.75",
        ">0.75–≤0.90",
        ">0.90–≤0.99",
        ">0.99–1.00",
    ]
    return groups, labels


def summarise_groups(
    grouping: str,
    group_ids: np.ndarray,
    group_labels: list[str],
    correct_by_model: dict[str, np.ndarray],
) -> list[dict[str, str | int | float]]:
    """Calculate test accuracy and support for each group and model"""
    rows = []
    for group_id, label in enumerate(group_labels):
        # skip ranges that contain no test nodes
        selected = group_ids == group_id
        if not selected.any():
            continue
        for model_name, correct in correct_by_model.items():
            rows.append(
                {
                    "grouping": grouping,
                    "group": label,
                    "model": model_name,
                    "nodes": int(selected.sum()),
                    "accuracy": float(correct[selected].mean()),
                }
            )
    return rows


def main() -> None:
    """Load saved predictions, compare test groups, and save results"""
    data = load_graph()
    test_mask = None
    predictions = {}

    # load both models' predictions and verify they use the same test split
    for model_name in MODELS:
        path = PREDICTION_DIR / f"predictions_{model_name}_seed42_five-seed.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; generate predictions with predict.py first")
        with np.load(path) as saved:
            model_test_mask = saved["test_mask"].astype(bool)
            if test_mask is None:
                test_mask = model_test_mask
            elif not np.array_equal(test_mask, model_test_mask):
                raise ValueError("MLP and GCN predictions use different test splits")
            predictions[model_name] = saved["predictions"]

    if test_mask is None or any(pred.shape != (data.num_nodes,) for pred in predictions.values()):
        raise ValueError("Prediction arrays must contain one prediction per graph node")

    # measure graph properties and compare predictions only on held-out test nodes
    test_indices = np.flatnonzero(test_mask)
    degree = node_degrees(data.edge_index, data.num_nodes).cpu().numpy()[test_mask]
    # this uses ground-truth neighbour labels, so it is retrospective analysis only
    agreement = neighbour_agreement(data.edge_index, data.y, data.num_nodes).cpu().numpy()[test_mask]
    labels = data.y.cpu().numpy()[test_mask]
    correct = {
        model_name: predictions[model_name][test_mask] == labels
        for model_name in MODELS
    }

    degree_ids, degree_labels = degree_groups(degree)
    agreement_ids, agreement_labels = agreement_groups(agreement)
    rows = summarise_groups("degree", degree_ids, degree_labels, correct)
    rows.extend(summarise_groups("neighbour_agreement", agreement_ids, agreement_labels, correct))

    # save exact group counts and accuracies as a csv
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["grouping", "group", "model", "nodes", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # plot the two accuracy comparisons side by side
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, grouping, group_ids, group_labels in (
        (axes[0], "degree", degree_ids, degree_labels),
        (axes[1], "neighbour_agreement", agreement_ids, agreement_labels),
    ):
        # omit empty groups and show each plotted group's sample count
        active_groups = [group_id for group_id in range(len(group_labels)) if (group_ids == group_id).any()]
        tick_labels = [
            f"{group_labels[group_id]}\nn={(group_ids == group_id).sum()}"
            for group_id in active_groups
        ]
        for model_name in MODELS:
            accuracies = [
                correct[model_name][group_ids == group_id].mean()
                for group_id in active_groups
            ]
            axis.plot(range(len(active_groups)), accuracies, marker="o", label=model_name.upper())
        axis.set_xticks(range(len(active_groups)), tick_labels)
        axis.set_ylim(0, 1)
        axis.set_ylabel("test accuracy")
        axis.set_xlabel(
            "in-degree"
            if grouping == "degree"
            else "fraction of neighbours with the same true class"
        )
        axis.set_title(
            "accuracy by node degree"
            if grouping == "degree"
            else "accuracy by neighbour-label agreement"
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    fig.suptitle("Test accuracy by graph context")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)

    print(f"Test nodes analysed: {len(test_indices)}")
    print("Neighbour agreement uses true labels and is retrospective only")
    print(f"Saved grouped results to {RESULTS_PATH}")
    print(f"Saved plot to {PLOT_PATH}")
    for row in rows:
        print(
            f"{row['grouping']:22s} {row['group']:18s} {row['model'].upper():3s} "
            f"n={row['nodes']:5d} accuracy={row['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
