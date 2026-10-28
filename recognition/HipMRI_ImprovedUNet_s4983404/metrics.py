"""Exclusive multiclass Dice/IoU, with explicit empty-mask conventions."""
import torch
from torch.nn import functional as F


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target.sum(1) > 0
    probabilities = logits.softmax(1) * valid.unsqueeze(1)
    intersection = (probabilities * target).sum((0, 2, 3))
    denominator = (probabilities + target).sum((0, 2, 3))
    dice = (2*intersection + 1e-6) / (denominator + 1e-6)
    cross_entropy = F.cross_entropy(logits, target.argmax(1), reduction='none')
    return cross_entropy[valid].mean() + 1 - dice[1:].mean()


def confusion_counts(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target.sum(1) > 0
    predicted = logits.argmax(1)[valid]
    actual = target.argmax(1)[valid]
    return torch.bincount(6*actual + predicted, minlength=36).reshape(6, 6)


def scores(confusion: torch.Tensor) -> dict[str, list[float | None]]:
    confusion = confusion.double()
    intersection = confusion.diag()
    denominator = confusion.sum(0) + confusion.sum(1)
    union = denominator - intersection
    # Both empty: undefined (None), excluded from macro averages; one empty: zero.
    return {name: [float(value) if count > 0 else None
                   for value, count in zip(numerator / divisor.clamp_min(1), divisor)]
            for name, numerator, divisor in (('dice', 2*intersection, denominator),
                                             ('iou', intersection, union))}
