"""Losses and explicit empty-class metrics for 2D segmentation.

Class zero is background. Reported Dice/IoU use pooled pixel counts; they are
not patient-averaged scores. A class absent from both truth and prediction is
undefined (``None``), never a perfect score. Boundary F1 measures distance in
the supplied mask grid's pixels, not millimetres or clinical contour accuracy.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


_LABEL_DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _labels(tensor: Tensor, name: str, num_classes: int) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype not in _LABEL_DTYPES:
        raise TypeError(f"{name} must contain integer class labels")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if bool((tensor < 0).any()) or bool((tensor >= num_classes).any()):
        raise ValueError(f"{name} labels must be in [0, {num_classes - 1}]")


def _logits(logits: Tensor, targets: Tensor, num_classes: int, channel_dim: int) -> None:
    if not torch.is_floating_point(logits):
        raise TypeError("Logits must be floating-point tensors")
    expected = list(targets.shape)
    expected.insert(channel_dim, num_classes)
    if list(logits.shape) != expected:
        raise ValueError(f"Expected logits shape {tuple(expected)}, got {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Logits must be finite")


def segmentation_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross entropy plus foreground soft Dice loss, with epsilon=1e-6.

    ``logits`` are [B,C,H,W] and integer ``targets`` are [B,H,W]. Dice sums
    over batch and spatial dimensions, then averages all foreground classes.
    Empty truth classes are retained, so foreground false positives are
    penalised. Half precision logits are evaluated in float32 for stability.
    """
    if not isinstance(logits, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("logits and targets must be torch.Tensor instances")
    if logits.ndim != 4 or targets.ndim != 3:
        raise ValueError("Expected logits [B,C,H,W] and targets [B,H,W]")
    if logits.shape[1] < 2:
        raise ValueError("At least two classes, including background, are required")
    num_classes = logits.shape[1]
    _labels(targets, "targets", num_classes)
    _logits(logits, targets, num_classes, 1)
    if logits.device != targets.device:
        raise ValueError("logits and targets must be on the same device")
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    truth = targets.long()
    probabilities = work.softmax(dim=1)
    one_hot = F.one_hot(truth, num_classes).permute(0, 3, 1, 2).to(work.dtype)
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = (probabilities + one_hot).sum(dim=(0, 2, 3))
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return F.cross_entropy(work, truth) + (1 - dice[1:]).mean()


@torch.no_grad()
def confusion_matrix(logits_or_predictions: Tensor, targets: Tensor,
                     num_classes: int) -> Tensor:
    """Return int64 [C,C] counts with rows=true labels, columns=predictions.

    Targets/predictions may be [H,W] or [B,H,W]. Alternatively supply logits
    [C,H,W] or [B,C,H,W] respectively; argmax selects the predicted class.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if not isinstance(logits_or_predictions, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("predictions and targets must be torch.Tensor instances")
    if targets.ndim not in (2, 3):
        raise ValueError("Targets must have shape [H,W] or [B,H,W]")
    _labels(targets, "targets", num_classes)
    predictions = logits_or_predictions
    if predictions.device != targets.device:
        raise ValueError("predictions and targets must be on the same device")
    if predictions.ndim == targets.ndim + 1:
        channel_dim = targets.ndim - 2
        _logits(predictions, targets, num_classes, channel_dim)
        predictions = predictions.argmax(dim=channel_dim)
    elif predictions.shape != targets.shape:
        raise ValueError("Prediction and target mask shapes must match")
    _labels(predictions, "predictions", num_classes)
    indices = targets.long().reshape(-1) * num_classes + predictions.long().reshape(-1)
    return torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def scores_from_confusion(matrix: Tensor) -> dict:
    """Convert counts to JSON-safe scores; macro excludes background/undefined classes."""
    if not isinstance(matrix, Tensor):
        raise TypeError("matrix must be a torch.Tensor")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("Confusion matrix must be a nonempty square [C,C] tensor")
    if matrix.dtype not in _LABEL_DTYPES or matrix.dtype == torch.bool:
        raise TypeError("Confusion matrix must contain integer counts")
    if bool((matrix < 0).any()):
        raise ValueError("Confusion matrix counts cannot be negative")
    counts = matrix.detach().cpu().tolist()
    per_class = []
    for class_index, row in enumerate(counts):
        true_positive = row[class_index]
        support = sum(row)
        predicted = sum(other[class_index] for other in counts)
        denominator = support + predicted
        union = denominator - true_positive
        per_class.append({
            "class_index": class_index,
            "dice": 2 * true_positive / denominator if denominator else None,
            "iou": true_positive / union if union else None,
            "support_pixels": support,
        })
    foreground = [entry["dice"] for entry in per_class[1:] if entry["dice"] is not None]
    return {
        "per_class": per_class,
        "macro_foreground_dice": sum(foreground) / len(foreground) if foreground else None,
    }


@torch.no_grad()
def boundary_f1(prediction: Tensor, target: Tensor, tolerance: int = 2) -> float | None:
    """Mean binary boundary F1 with a Chebyshev tolerance in mask-grid pixels.

    Inputs are integer/bool [H,W] or [B,H,W] masks. Boundaries are foreground
    pixels touching background in an 8-neighbourhood; outside-image pixels are
    background. A tolerance of 2 permits a square 5x5 neighbourhood. Evaluate
    both models on the same grid: resized-grid pixels are not physical mm.
    Both-empty samples are undefined and excluded; one-empty samples score 0.
    """
    if isinstance(tolerance, bool) or not isinstance(tolerance, int) or tolerance < 0:
        raise ValueError("tolerance must be a nonnegative integer")
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be torch.Tensor instances")
    if prediction.ndim not in (2, 3) or prediction.shape != target.shape:
        raise ValueError("Binary masks must have matching [H,W] or [B,H,W] shapes")
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    _labels(prediction, "prediction", 2)
    _labels(target, "target", 2)

    def contours(mask: Tensor) -> Tensor:
        values = mask.reshape(-1, 1, *mask.shape[-2:]).float()
        padded = F.pad(values, (1, 1, 1, 1), value=0)
        eroded = -F.max_pool2d(-padded, kernel_size=3, stride=1)
        return (values - eroded).bool()

    predicted_boundary, true_boundary = contours(prediction), contours(target)
    kernel = 2 * tolerance + 1
    near_prediction = F.max_pool2d(predicted_boundary.float(), kernel, 1, tolerance).bool()
    near_truth = F.max_pool2d(true_boundary.float(), kernel, 1, tolerance).bool()
    values = []
    for predicted, truth, predicted_region, true_region in zip(
            predicted_boundary, true_boundary, near_prediction, near_truth):
        predicted_count, true_count = int(predicted.sum()), int(truth.sum())
        if predicted_count == 0 and true_count == 0:
            continue
        if predicted_count == 0 or true_count == 0:
            values.append(0.0)
            continue
        precision = int((predicted & true_region).sum()) / predicted_count
        recall = int((truth & predicted_region).sum()) / true_count
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(values) / len(values) if values else None
