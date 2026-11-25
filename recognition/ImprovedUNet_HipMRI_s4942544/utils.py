"""
utils.py -- Metrics, profiling and plotting helpers shared by train.py and predict.py.

Metrics are accumulated over a whole split (dataset-level Dice/IoU) rather than
averaged per slice. Per-slice averaging is ill-defined for the prostate: about 64%
of slices contain no prostate at all (data audit), and a slice where both the
prediction and the ground truth are empty would otherwise count as a perfect 1.0,
inflating the score. Dataset-level accumulation counts every pixel exactly once.
"""
import json
import random
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless plotting on Rangpur
import matplotlib.pyplot as plt

from dataset import CLASS_NAMES, NUM_CLASSES, PROSTATE


def set_seed(seed):
    """Seed every RNG in play so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SegMetrics:
    """Accumulates per-class intersection / prediction / target pixel counts over a split."""

    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.inter = torch.zeros(self.num_classes, dtype=torch.float64)
        self.pred = torch.zeros(self.num_classes, dtype=torch.float64)
        self.true = torch.zeros(self.num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, target):
        pred = logits.argmax(dim=1)
        for c in range(self.num_classes):
            p, t = pred == c, target == c
            self.inter[c] += (p & t).sum().item()
            self.pred[c] += p.sum().item()
            self.true[c] += t.sum().item()

    def dice(self):
        return (2 * self.inter / (self.pred + self.true).clamp(min=1)).tolist()

    def iou(self):
        return (self.inter / (self.pred + self.true - self.inter).clamp(min=1)).tolist()

    def summary(self):
        d, i = self.dice(), self.iou()
        return {
            "dice": dict(zip(CLASS_NAMES, d)),
            "iou": dict(zip(CLASS_NAMES, i)),
            "prostate_dice": d[PROSTATE],
            "mean_fg_dice": float(np.mean(d[1:])),  # foreground classes only
        }


@torch.no_grad()
def measure_latency(model, device, shape=(1, 1, 256, 128), warmup=10, runs=50):
    """Mean per-slice inference time in milliseconds (batch of 1, as in a clinical viewer)."""
    model.eval()
    x = torch.randn(*shape, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / runs


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_history(history, path, title):
    """Loss curves and per-class validation Dice, side by side."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(epochs, history["train_loss"], "o-", label="train")
    ax1.plot(epochs, history["val_loss"], "s-", label="validation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Dice + CE loss")
    ax1.set_title("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    for c in range(1, NUM_CLASSES):
        name = CLASS_NAMES[c]
        values = [h[name] for h in history["val_dice"]]
        ax2.plot(epochs, values, "o-", lw=2.5 if c == PROSTATE else 1.2,
                 label=name + (" (target)" if c == PROSTATE else ""))
    ax2.axhline(0.75, color="red", ls="--", lw=1, label="0.75 prostate target")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation Dice")
    ax2.set_ylim(0, 1)
    ax2.set_title("Per-class validation Dice")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  # whole epochs only
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
