"""Classification, calibration and selective-prediction evaluation helpers."""

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score,
)

from dataset import CLASS_NAMES


def split_validation_roles(y, validation_mask, seed=43):
    """Separate model selection from post-training calibration/threshold fitting."""
    random = np.random.default_rng(seed)
    labels = y.cpu().numpy()
    validation_ids = torch.where(validation_mask)[0].cpu().numpy()
    selection = torch.zeros_like(validation_mask)
    calibration = torch.zeros_like(validation_mask)
    for label in range(len(CLASS_NAMES)):
        ids = random.permutation(validation_ids[labels[validation_ids] == label])
        cut = len(ids) // 2
        if cut == 0 or len(ids) == cut:
            raise ValueError("Validation classes must support two nonempty roles")
        selection[torch.from_numpy(ids[:cut])] = True
        calibration[torch.from_numpy(ids[cut:])] = True
    return selection, calibration


def fit_temperature(logits, labels):
    """Fit a positive scalar on calibration nodes only, independently of training."""
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.))
    if not np.isfinite(temperature):
        raise ValueError("Temperature fitting produced a nonfinite value")
    # Do not keep a calibration setting that worsens its fitting objective.
    if F.cross_entropy(logits / temperature, labels) > F.cross_entropy(logits, labels):
        return 1.0
    return temperature


def reliability(probabilities, labels, bins=10):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    results = []
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        chosen = (confidence >= lower) & (confidence < upper if index < bins-1 else confidence <= upper)
        count = int(chosen.sum())
        accuracy = float(correct[chosen].mean()) if count else None
        average = float(confidence[chosen].mean()) if count else None
        if count:
            error += count / len(labels) * abs(accuracy - average)
        results.append({"lower": lower, "upper": upper, "count": count,
                        "accuracy": accuracy, "confidence": average})
    return float(error), results


def classification_metrics(logits, labels, temperature=1.0):
    probabilities = (logits.detach().cpu() / temperature).softmax(dim=1).numpy()
    truth = labels.detach().cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=list(range(len(CLASS_NAMES))), zero_division=0,
    )
    ece, calibration_bins = reliability(probabilities, truth)
    auc = roc_auc_score(truth, probabilities, multi_class="ovr", average="macro")
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1.mean()),
        "macro_ovr_auroc": float(auc),
        "negative_log_likelihood": float(F.cross_entropy(logits.detach().cpu() / temperature, labels.cpu())),
        "ece_10_bins": ece,
        "reliability_bins": calibration_bins,
        "confusion_matrix": confusion_matrix(truth, predictions, labels=range(len(CLASS_NAMES))).tolist(),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(CLASS_NAMES)
        },
    }


def choose_threshold(logits, labels, temperature, target_accuracy=0.9):
    """Maximize calibration coverage among grid thresholds meeting a target.

    This is an empirical fitting rule, not a guarantee of held-out accuracy.
    A threshold greater than one means that all predictions require review.
    """
    probabilities = (logits.detach().cpu() / temperature).softmax(dim=1).numpy()
    truth = labels.cpu().numpy()
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == truth
    candidates = []
    for threshold in np.linspace(0., 1., 101):
        chosen = confidence >= threshold
        if chosen.sum() >= max(1, int(0.05 * len(truth))):
            if correct[chosen].mean() >= target_accuracy:
                candidates.append((float(chosen.mean()), float(threshold)))
    if not candidates:
        return 1.01
    return max(candidates, key=lambda item: (item[0], -item[1]))[1]


def selective_metrics(logits, labels, temperature, threshold):
    probabilities = (logits.detach().cpu() / temperature).softmax(dim=1).numpy()
    truth = labels.cpu().numpy()
    chosen = probabilities.max(axis=1) >= threshold
    accepted = int(chosen.sum())
    accuracy = float((probabilities.argmax(axis=1)[chosen] == truth[chosen]).mean()) if accepted else None
    return {"threshold": float(threshold), "accepted": accepted, "total": len(truth),
            "coverage": float(chosen.mean()), "accepted_accuracy": accuracy,
            "accepted_error_rate": 1.-accuracy if accuracy is not None else None}


def plot_history(history, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Training")
    axes[0].plot(epochs, [row["selection_loss"] for row in history], label="Model-selection validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend(fontsize=8)
    axes[1].plot(epochs, [row["selection_macro_f1"] for row in history])
    axes[1].set(xlabel="Epoch", ylabel="Model-selection macro-F1", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
