"""Sensitivity study: stratified 60/20/20 versus 70/10/20 node splits.

The original 20% test mask is held fixed and is NEVER evaluated here.
The 70/10/20 setting promotes half of the ORIGINAL validation nodes
into training; its other half remains a COMMON validation subset used
for both settings' side-by-side comparison.

This experiment only saves separate CSV/PNG files in figures/.
It NEVER loads, modifies or saves checkpoints in checkpoints/.

Run from repository root:
    python recognition/GCN_49832853/split_ablation.py
"""

import argparse
import copy
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from dataset import load_facebook, split_nodes
from modules import GCN, MLP
from train import evaluate, set_seed, train_one_epoch


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "figures"
DEFAULT_SEED = 42


def make_masks(data, seed):
    """Reproduce original split, then build NESTED 70/10/20 masks.

    Comparison validation = 10% of all nodes, common to both settings.
    Both train masks are disjoint from this common validation mask.
    """
    original = split_nodes(data, seed=seed)
    train60 = original.train_mask.clone()
    val20 = original.val_mask.clone()
    test20 = original.test_mask.clone()

    val_indices = np.flatnonzero(val20.cpu().numpy())
    y = original.y.cpu().numpy()
    common_ids, promoted_ids = train_test_split(
        val_indices,
        test_size=0.5,
        random_state=seed,
        stratify=y[val_indices],
    )

    common10 = torch.zeros(original.num_nodes, dtype=torch.bool)
    common10[common_ids] = True
    promoted10 = torch.zeros(original.num_nodes, dtype=torch.bool)
    promoted10[promoted_ids] = True

    train70 = train60 | promoted10
    val10 = common10

    assert torch.equal(val20, common10 | promoted10)
    assert not (train60 & val20).any()
    assert not (train70 & val10).any()
    assert not (train70 & test20).any()
    assert not (val10 & test20).any()
    assert torch.equal(train70 | val10 | test20,
                       torch.ones(original.num_nodes, dtype=torch.bool))

    n = original.num_nodes
    assert (train60.sum().item(), val20.sum().item(), test20.sum().item()) == (
        13482, 4494, 4494
    ), "Unexpected data dimensions: revise expected counts."
    assert (train70.sum().item(), val10.sum().item(), test20.sum().item()) == (
        15729, 2247, 4494
    ), "Unexpected nested split sizes."

    print(f"Nodes: {n}; fixed test nodes (NOT evaluated): {test20.sum().item()}")
    print(f"Shared validation nodes for comparison: {common10.sum().item()}")
    print("60/20/20: train=13482, val=4494, test=4494")
    print("70/10/20: train=15729, val=2247, test=4494")

    return {
        "60/20/20": (train60, val20),
        "70/10/20": (train70, val10),
    }, common10, test20


def fit_one(name, model, data, common_mask, epochs, patience, split_name):
    """Match train.py: Adam 0.01, weight decay 5e-4, macro-F1 selection."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_epoch = 0
    best_weights = None
    stale = 0
    epoch_history = []

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, data, optimizer, criterion)
        val_acc, val_f1 = evaluate(model, data, data.val_mask)
        epoch_history.append({
            "split": split_name,
            "model": name,
            "epoch": epoch,
            "train_loss": loss,
            "native_val_accuracy": val_acc,
            "native_val_macro_f1": val_f1,
        })

        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  epoch {epoch:03d}: loss={loss:.4f}, "
                f"native validation accuracy={val_acc:.4f}, macro-F1={val_f1:.4f}"
            )
        if stale >= patience:
            print(f"  early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_weights)
    native_acc, native_f1 = evaluate(model, data, data.val_mask)
    common_acc, common_f1 = evaluate(model, data, common_mask)
    print(
        f"  best_epoch={best_epoch}; "
        f"shared-10%-validation acc={common_acc:.4f}, "
        f"macro-F1={common_f1:.4f}"
    )

    return {
        "split": split_name,
        "model": name,
        "train_nodes": int(data.train_mask.sum().item()),
        "native_val_nodes": int(data.val_mask.sum().item()),
        "common_val_nodes": int(common_mask.sum().item()),
        "best_epoch": best_epoch,
        "native_val_accuracy": native_acc,
        "native_val_macro_f1": native_f1,
        "common_val_accuracy": common_acc,
        "common_val_macro_f1": common_f1,
    }, epoch_history


def write_csv(filepath, records, fields):
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print("Saved:", filepath)


def plot_comparison(results, output):
    names = ["MLP", "GCN"]
    rows_by_key = {(r["split"], r["model"]): r for r in results}
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    w = 0.35
    for offset, split in [(-w / 2, "60/20/20"), (w / 2, "70/10/20")]:
        y = [rows_by_key[(split, name)]["common_val_macro_f1"] for name in names]
        bars = ax.bar(x + offset, y, width=w, label=split)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Macro-F1 on identical shared 10% validation nodes")
    ax.set_title("Split ratio sensitivity (seed 42; test mask held fixed)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print("Saved:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    data = load_facebook()
    masks, common_mask, fixed_test_mask = make_masks(data, args.seed)
    # fixed_test_mask exists only to verify set membership and disjointness.
    # No predictions on test nodes are evaluated or reported here.
    assert fixed_test_mask.sum().item() == 4494

    data = data.to(device)
    common_mask = common_mask.to(device)
    feature_count = data.num_node_features
    class_count = int(data.y.max().item()) + 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, history = [], []

    for split_name, (training_mask, validation_mask) in masks.items():
        data.train_mask = training_mask.to(device)
        data.val_mask = validation_mask.to(device)
        # Note: no code path creates a test metric.
        for name, constructor in (("MLP", MLP), ("GCN", GCN)):
            print(f"\n===== {split_name}: {name} =====")
            set_seed(args.seed)  # Same model initialization seed for each run
            model = constructor(feature_count, class_count).to(device)
            one_row, one_history = fit_one(
                name, model, data, common_mask,
                args.epochs, args.patience, split_name
            )
            summary.append(one_row)
            history.extend(one_history)
            del model

    summary_fields = [
        "split", "model", "train_nodes", "native_val_nodes", "common_val_nodes",
        "best_epoch", "native_val_accuracy", "native_val_macro_f1",
        "common_val_accuracy", "common_val_macro_f1"
    ]
    write_csv(OUT_DIR / "split_ablation_summary.csv", summary, summary_fields)
    write_csv(
        OUT_DIR / "split_ablation_epoch_history.csv", history,
        ["split", "model", "epoch", "train_loss", "native_val_accuracy", "native_val_macro_f1"]
    )
    if args.seed == DEFAULT_SEED:
        plot_comparison(summary, OUT_DIR / "split_ablation_common_val_f1.png")
    print("\n===== SHARED VALIDATION (same 2247 nodes) =====")
    for row in summary:
        print(
            f'{row["split"]} {row["model"]}: '
            f'accuracy={row["common_val_accuracy"]:.4f}, '
            f'macro-F1={row["common_val_macro_f1"]:.4f}'
        )
    print("\nNo test labels were evaluated. No existing .pt checkpoints were changed.")
    print("Exploratory experiment only: prior test results were already inspected.")


if __name__ == "__main__":
    main()
