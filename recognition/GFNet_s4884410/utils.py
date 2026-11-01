"""
Shared helpers for training, evaluation and later analysis.

- set_seed:               reproducible Python / NumPy / PyTorch randomness
- predict:                run a model over a slice loader and collect per-slice outputs
- aggregate_patients:     average slice probabilities into one prediction per patient
- classification_metrics: accuracy, per-class precision/recall, macro-F1, AUROC
- evaluate:               predict + metrics at slice and patient level, in one call
- save_rows_csv:          write a list of dicts to CSV
- plot_history:           training curves (loss and accuracy per epoch)

The same evaluate() is used for validation during training and for the final
test evaluation, so every model and split is scored identically.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def set_seed(seed):
    """Seed every random number generator the pipeline uses."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, amp=True):
    """Run the model over a slice loader in eval mode.

    Returns:
        (mean cross-entropy loss, rows), with one row per slice:
        {"pid", "slice_pos", "label", "logit_ad_minus_nc", "prob_ad"}.

    The logit difference (AD minus NC) is kept so that temperature scaling can be
    fitted later without rerunning the model. Logits are cast to float32 before
    the softmax so that mixed precision does not round the probabilities.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    use_amp = amp and device.type == "cuda"
    total_loss, n = 0.0, 0
    rows = []
    for x, y, pids, slice_pos in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            logits = model(x)
        logits = logits.float()
        total_loss += loss_fn(logits, y).item()
        n += y.numel()
        prob_ad = torch.softmax(logits, dim=1)[:, 1]
        logit_diff = logits[:, 1] - logits[:, 0]
        for pid, pos, label, z, p in zip(
            pids, slice_pos.tolist(), y.tolist(), logit_diff.tolist(), prob_ad.tolist()
        ):
            rows.append(
                {
                    "pid": pid,
                    "slice_pos": pos,
                    "label": label,
                    "logit_ad_minus_nc": z,
                    "prob_ad": p,
                }
            )
    return total_loss / n, rows


def aggregate_patients(slice_rows):
    """One row per patient: mean AD probability over its slices.

    Returns rows {"pid", "label", "n_slices", "prob_ad", "slice_prob_std"}, sorted
    by patient ID. slice_prob_std (the standard deviation of the slice
    probabilities) measures how much the patient's slices disagree.
    """
    by_pid = defaultdict(list)
    labels = {}
    for r in slice_rows:
        by_pid[r["pid"]].append(r["prob_ad"])
        labels[r["pid"]] = r["label"]
    rows = []
    for pid in sorted(by_pid):
        probs = np.asarray(by_pid[pid])
        rows.append(
            {
                "pid": pid,
                "label": labels[pid],
                "n_slices": len(probs),
                "prob_ad": float(probs.mean()),
                "slice_prob_std": float(probs.std()),
            }
        )
    return rows


def classification_metrics(labels, probs, threshold=0.5):
    """Accuracy, per-class precision and recall, macro-F1 and AUROC.

    AD (label 1) is the positive class, so recall_ad is sensitivity and
    recall_nc is specificity. Predictions are AD when prob >= threshold. AUROC
    uses the probabilities directly, and is NaN if only one class is present.
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    pred = (p >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0
    )
    auroc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    return {
        "acc": float((pred == y).mean()),
        "precision_nc": float(precision[0]),
        "precision_ad": float(precision[1]),
        "recall_nc": float(recall[0]),  # specificity
        "recall_ad": float(recall[1]),  # sensitivity
        "macro_f1": float(f1.mean()),
        "auroc": float(auroc),
    }


def evaluate(model, loader, device, amp=True):
    """Loss plus slice- and patient-level metrics, and the per-slice / per-patient rows.

    Returns:
        {"loss", "slice": metrics, "patient": metrics, "slice_rows", "patient_rows"}
    """
    loss, slice_rows = predict(model, loader, device, amp)
    patient_rows = aggregate_patients(slice_rows)
    return {
        "loss": loss,
        "slice": classification_metrics(
            [r["label"] for r in slice_rows], [r["prob_ad"] for r in slice_rows]
        ),
        "patient": classification_metrics(
            [r["label"] for r in patient_rows], [r["prob_ad"] for r in patient_rows]
        ),
        "slice_rows": slice_rows,
        "patient_rows": patient_rows,
    }


def save_rows_csv(rows, path):
    """Write a list of dicts (all with the same keys) to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(history, out_path, title=""):
    """Plot train/val loss and train/val accuracy against epoch, and save to out_path.

    history: list of per-epoch dicts as written by train.py (history.csv rows).
    """
    import matplotlib

    matplotlib.use("Agg")  # no display on Rangpur
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax1.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax2.plot(epochs, [h["train_acc"] for h in history], label="train (slice)")
    ax2.plot(epochs, [h["val_slice_acc"] for h in history], label="validation (slice)")
    ax2.plot(
        epochs, [h["val_patient_acc"] for h in history], label="validation (patient)"
    )
    ax2.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
