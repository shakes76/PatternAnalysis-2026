"""Evaluation metrics computed from confusion matrices.

Dice for class c from a confusion matrix (rows = ground truth, cols = prediction):

    D_c = 2 TP / (2 TP + FP + FN)

* Undefined (NaN) when the class is absent from both ground truth and
  prediction; reported in ``undefined_counts`` and excluded from means.
* 0.0 when the class is absent from ground truth but predicted.

Global metrics use one confusion matrix summed over all evaluated volumes.
Per-volume metrics compute Dice per volume, then mean and sample SD (ddof=1)
across volumes. Mean foreground Dice averages classes 1..C-1.
"""

from __future__ import annotations

import numpy as np
import torch

CLASS_NAMES: tuple[str, ...] = ("background", "bladder", "body", "bone", "rectum", "prostate")


def confusion_matrix(
    target: torch.Tensor, prediction: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """[C, C] int64 confusion matrix on CPU. Rows: ground truth, columns: prediction."""
    target = target.reshape(-1).long()
    prediction = prediction.reshape(-1).to(target.device).long()
    if target.numel() != prediction.numel():
        raise ValueError("target and prediction contain different numbers of elements")
    for name, values in (("target", target), ("prediction", prediction)):
        if values.numel() and (values.min() < 0 or values.max() >= num_classes):
            raise ValueError(f"{name} contains labels outside [0, {num_classes - 1}]")
    counts = torch.bincount(target * num_classes + prediction, minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes).cpu()


def dice_from_confusion(matrix: np.ndarray | torch.Tensor) -> np.ndarray:
    """Per-class Dice; NaN where 2TP + FP + FN == 0."""
    matrix = np.asarray(matrix, dtype=np.float64)
    true_positive = np.diag(matrix)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    return np.divide(
        2 * true_positive,
        denominator,
        out=np.full(len(matrix), np.nan),
        where=denominator > 0,
    )


def _nan_to_none(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def global_metrics(matrix: np.ndarray | torch.Tensor, class_names=CLASS_NAMES) -> dict:
    """Dice per class, mean foreground Dice and pixel accuracy from one matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    dice = dice_from_confusion(matrix)
    foreground = dice[1:]
    total = matrix.sum()
    return {
        "dice_per_class": {
            name: _nan_to_none(d) for name, d in zip(class_names, dice, strict=True)
        },
        "mean_foreground_dice": (
            float(np.nanmean(foreground)) if np.isfinite(foreground).any() else None
        ),
        "pixel_accuracy": float(np.trace(matrix) / total) if total else None,
        "undefined_classes": [
            name for name, d in zip(class_names, dice, strict=True) if np.isnan(d)
        ],
    }


def per_volume_metrics(matrices: list[np.ndarray | torch.Tensor], class_names=CLASS_NAMES) -> dict:
    """Per-volume Dice summarised as mean and sample SD (ddof=1) across volumes."""
    if not matrices:
        raise ValueError("No volumes to summarise")
    dice = np.stack([dice_from_confusion(m) for m in matrices])  # [V, C]
    foreground = dice[:, 1:]
    with np.errstate(all="ignore"):
        volume_fg_mean = np.array(
            [np.nanmean(row) if np.isfinite(row).any() else np.nan for row in foreground]
        )

    def mean_sd(values: np.ndarray) -> dict:
        valid = values[np.isfinite(values)]
        return {
            "mean": float(valid.mean()) if valid.size else None,
            "sd": float(valid.std(ddof=1)) if valid.size > 1 else None,
            "n": int(valid.size),
        }

    return {
        "mean_foreground_dice": mean_sd(volume_fg_mean),
        "dice_per_class": {name: mean_sd(dice[:, i]) for i, name in enumerate(class_names)},
        "undefined_counts": {
            name: int(np.isnan(dice[:, i]).sum()) for i, name in enumerate(class_names)
        },
        "volumes": len(matrices),
    }
