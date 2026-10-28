"""Evaluate frozen MLP and GCN checkpoints on validation or final test nodes.

Place beside dataset.py, modules.py and checkpoints/.
Usage:
    python final_test.py --split val
    python final_test.py --split test --confirm-test

Use the test split only after architecture and hyperparameters are frozen.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from dataset import load_facebook, split_nodes
from modules import GCN, MLP

BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ["company", "government", "politician", "tvshow"]
SEED = 42


def expected_calibration_error(probs, labels, n_bins=10):
    """Top-label ECE using equal-width confidence bins."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        in_bin = (confidence >= lower) & (
            (confidence <= upper) if i == n_bins - 1 else (confidence < upper)
        )
        if np.any(in_bin):
            ece += np.mean(in_bin) * abs(
                np.mean(correct[in_bin]) - np.mean(confidence[in_bin])
            )
    return float(ece)


@torch.inference_mode()
def evaluate_model(name, model, data, split, threshold):
    checkpoint = BASE_DIR / "checkpoints" / f"{name.lower()}_best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    logits = model(data.x, data.edge_index)
    mask = getattr(data, f"{split}_mask")
    y_true = data.y[mask].cpu().numpy()
    probs = torch.softmax(logits[mask], dim=1).cpu().numpy()
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (y_pred == y_true)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auroc = roc_auc_score(
        y_true, probs, labels=np.arange(len(CLASS_NAMES)),
        multi_class="ovr", average="macro"
    )
    ece = expected_calibration_error(probs, y_true)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accepted = confidence >= threshold
    coverage = float(np.mean(accepted))
    accepted_acc = float(np.mean(correct[accepted])) if np.any(accepted) else float("nan")
    total_errors = int(np.sum(~correct))
    errors_flagged = (
        float(np.sum((~correct) & (~accepted)) / total_errors)
        if total_errors else float("nan")
    )

    print(f"\n===== {name} / {split.upper()} =====")
    print(f"Parameters: {n_params}")
    print(f"Evaluated nodes: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print(f"Macro-AUROC (OvR): {auroc:.4f}")
    print(f"ECE (10 bins): {ece:.4f}")
    print(f"Misclassified: {total_errors}")
    print(f"At validation-selected threshold {threshold:.2f}:")
    print(f"  Coverage: {coverage:.2%}")
    print(f"  Accepted accuracy: {accepted_acc:.2%}")
    print(f"  Errors flagged: {errors_flagged:.2%}")
    print("\nPer-class classification report:")
    print(classification_report(
        y_true, y_pred, labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(values_format="d", xticks_rotation=30, cmap="Blues")
    plt.title(f"{name} {split.title()} Confusion Matrix")
    plt.tight_layout()
    fig_dir = BASE_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    output = fig_dir / f"{name.lower()}_{split}_confusion_matrix.png"
    plt.savefig(output, dpi=170)
    plt.close()
    print("Saved figure:", output)

    return {"accuracy": acc, "macro_f1": f1, "auroc": auroc, "ece": ece}


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved models without retraining")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--confirm-test", action="store_true", help="Acknowledge this is a final test evaluation")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Confidence threshold chosen on validation data (default: 0.90)")
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_test:
        parser.error("To evaluate the held-out test split, pass --confirm-test after freezing all choices.")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1].")

    data = split_nodes(load_facebook(), seed=SEED)
    # Data split sanity checks: disjoint, complete, and reproducible.
    assert (data.train_mask | data.val_mask | data.test_mask).all()
    assert not (data.train_mask & data.val_mask).any()
    assert not (data.train_mask & data.test_mask).any()
    assert not (data.val_mask & data.test_mask).any()

    num_features = data.num_node_features
    num_classes = int(data.y.max().item()) + 1
    results = {}
    for name, cls in [("MLP", MLP), ("GCN", GCN)]:
        model = cls(num_features, num_classes)
        results[name] = evaluate_model(name, model, data, args.split, args.threshold)

    print(f"\n===== {args.split.upper()} SUMMARY =====")
    for name, values in results.items():
        print(f"{name}: " + ", ".join(f"{key}={value:.4f}" for key, value in values.items()))


if __name__ == "__main__":
    main()
