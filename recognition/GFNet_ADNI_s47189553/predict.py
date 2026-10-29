"""
predict.py — use a trained checkpoint to make predictions (metrics, confusion matrix, learning curve).

Usage:
    # Test-set evaluation (defaults to runs/gfnet/best.pt)
    python predict.py --root /path/to/dataset
    python predict.py --ckpt runs/gfnet/best.pt --root /path/to/dataset

    python predict.py --ckpt "runs/gfnet_cropflip/cv_best.pt" 
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataset import CLASSES, DEFAULT_ROOT, get_dataloaders
from modules import IMG_SIZE, IN_CHANS, GFNet, SimpleCNN
from train import evaluate, get_device


# --------------------------------------------------------------------------- #
# Constants — edit these instead of passing flags every run
# --------------------------------------------------------------------------- #
DATA_ROOT = DEFAULT_ROOT              # dataset root; set a path here to override dataset.py
CKPT_PATH = "runs/gfnet/best.pt"      # checkpoint to evaluate

BATCH_SIZE = 64
NUM_WORKERS = 8
THRESHOLD = 0.5                       # P(AD) above this = predicted AD
METRICS = {                           # key -> printed name
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall (AD sensitivity)",
    "specificity": "specificity (NC sensitivity)",
    "f1": "f1",
    "auc": "auc",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_model(ckpt_path: str, device):
    """Rebuild the model and load trained weights. Returns (model, ckpt)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    # Pick the architecture from the saved weights (SimpleCNN has conv "features.*" keys)
    is_cnn = any(k.startswith("features.") for k in ckpt["model"])
    model = (SimpleCNN() if is_cnn else GFNet()).to(device)   # same settings as training
    print(f"Model: {type(model).__name__}")
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch')}, val acc {ckpt.get('val_acc', 0):.4f})")
    return model, ckpt


def load_history(ckpt, ckpt_path: str):
    """Training history: stored inside cv_best.pt, otherwise history.json next to the checkpoint."""
    if "history" in ckpt:
        return ckpt["history"]
    path = os.path.join(os.path.dirname(ckpt_path), "history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# --------------------------------------------------------------------------- #
# Metrics 
# --------------------------------------------------------------------------- #
def binary_metrics(labels, probs, threshold=THRESHOLD) -> dict:
    """AD = positive class. Returns accuracy, precision, recall, specificity, F1, confusion matrix."""
    preds = [int(p > threshold) for p in probs]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0          # sensitivity
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision, "recall": recall, "specificity": specificity, "f1": f1,
        "auc": roc_auc(labels, probs),
        "confusion": [[tn, fp], [fn, tp]],                # rows = true NC/AD, cols = predicted NC/AD
    }


def roc_auc(labels, probs) -> float:
    """AUC = probability a random AD sample scores higher than a random NC sample."""
    pos = [p for y, p in zip(labels, probs) if y == 1]
    neg = [p for y, p in zip(labels, probs) if y == 0]
    if not pos or not neg:
        return float("nan")
    ranked = sorted([(p, 1) for p in pos] + [(p, 0) for p in neg])
    # Mann–Whitney U with average ranks for ties
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2                         # ranks are 1-based
        rank_sum += avg_rank * sum(1 for k in range(i, j) if ranked[k][1] == 1)
        i = j
    u = rank_sum - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def patient_level(dataset, probs):
    """Average slice probabilities per patient. Returns (labels, probs)."""
    per_patient, labels = defaultdict(list), {}
    for pid, (_, y), p in zip(dataset.patients, dataset.samples, probs.tolist()):
        per_patient[pid].append(p)
        labels[pid] = y
    ids = sorted(per_patient)
    return [labels[i] for i in ids], [sum(per_patient[i]) / len(per_patient[i]) for i in ids]


def pct(value) -> str:
    """0.7473 -> '74.73'"""
    return f"{100 * value:.2f}"


def print_confusion(name, m):
    """Confusion matrix with counts, % of each true class, and accuracy %."""
    print(f"\n  {name:10s}" + "".join(f"{'pred ' + c:>16s}" for c in CLASSES))
    for c, row in zip(CLASSES, m["confusion"]):
        total = sum(row) or 1
        print(f"  {'true ' + c:10s}" + "".join(f"{f'{n} ({100 * n / total:.1f}%)':>16s}" for n in row))
    print(f"  accuracy: {pct(m['accuracy'])} %")


# --------------------------------------------------------------------------- #
# Learning curve
# --------------------------------------------------------------------------- #
def plot_learning_curve(history, name, out_path):
    """Train vs val loss (left) and accuracy (right) per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(axes, ["loss", "acc"], ["Loss", "Accuracy"]):
        for split in ["train", "val"]:
            values = history.get(f"{split}_{metric}")
            if isinstance(values, list):                   # fold history.json has val_acc as one number
                ax.plot(range(1, len(values) + 1), values, label=split)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True)
        ax.legend()
    fig.suptitle(f"{name} learning curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved learning curve to {out_path}")


# --------------------------------------------------------------------------- #
# Test-set evaluation
# --------------------------------------------------------------------------- #
def evaluate_test_set(model, args, device):
    _, _, test_loader = get_dataloaders(
        args.root, batch_size=args.batch_size, img_size=IMG_SIZE, in_chans=IN_CHANS,
        num_workers=args.workers,
    )
    ds = test_loader.dataset
    print(f"Test set: {len(ds)} slices, {len(set(ds.patients))} patients")

    _, _, probs = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    slice_labels = [y for _, y in ds.samples]
    s = binary_metrics(slice_labels, probs.tolist())
    p_labels, p_probs = patient_level(ds, probs)
    p = binary_metrics(p_labels, p_probs)

    print("\nMetrics in %")
    print(f"{'':30s}{'Slice':>10s}{'Patient':>10s}")
    for k, name in METRICS.items():
        print(f"{name:30s}{pct(s[k]):>10s}{pct(p[k]):>10s}")
    print("\nConfusion matrix (rows = true class, cols = predicted; % of the true class)")
    print_confusion("slice", s)
    print_confusion("patient", p)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained ADNI model on the test set")
    ap.add_argument("--ckpt", default=CKPT_PATH)
    ap.add_argument("--root", default=DATA_ROOT, help="dataset root")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    model, ckpt = load_model(args.ckpt, device)

    history = load_history(ckpt, args.ckpt)
    if history:
        name = type(model).__name__
        out_path = os.path.join(os.path.dirname(args.ckpt), f"learning_curve_{name}.png")
        plot_learning_curve(history, name, out_path)
    else:
        print("No training history found, skipping learning curve")

    evaluate_test_set(model, args, device)


if __name__ == "__main__":
    main()