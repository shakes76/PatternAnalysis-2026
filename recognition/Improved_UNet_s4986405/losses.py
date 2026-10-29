"""Shared loss for mutually exclusive multiclass segmentation."""
import math

import torch
from torch.nn import functional as F


def dice_ce_loss(logits, targets, smooth=1e-5):
    """Return mean pixel cross-entropy plus batch-pooled macro soft Dice loss.

    Inputs have shape [B, C, H, W]. Targets must be float one-hot masks.
    All channels contribute equally to Dice, including channel 0.
    Half-precision inputs are promoted to float32 for the calculation.
    This differentiable loss is separate from hard-label reporting metrics.
    """
    if logits.ndim != 4 or targets.shape != logits.shape:
        raise ValueError('Expected matching logits and targets with shape [B, C, H, W]')
    if any(size == 0 for size in logits.shape) or logits.shape[1] < 2:
        raise ValueError('Inputs must be non-empty with at least two classes')
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise ValueError('Logits and one-hot targets must be floating point')
    if logits.device != targets.device:
        raise ValueError('Logits and targets must be on the same device')
    if not math.isfinite(smooth) or smooth <= 0:
        raise ValueError('smooth must be finite and positive')
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError('Logits and targets must be finite')
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError('Targets must contain binary one-hot values')
    if not torch.all(targets.sum(dim=1) == 1):
        raise ValueError('Each target pixel must have exactly one active channel')

    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    log_probabilities = F.log_softmax(logits.to(dtype=dtype), dim=1)
    probabilities = log_probabilities.exp()
    targets = targets.to(dtype=dtype)
    cross_entropy = -(targets * log_probabilities).sum(dim=1).mean()

    axes = (0, 2, 3)
    intersection = (probabilities * targets).sum(dim=axes)
    denominator = probabilities.sum(dim=axes) + targets.sum(dim=axes)
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return cross_entropy + (1 - dice.mean())
