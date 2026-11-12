"""Batch-level soft Dice + cross-entropy loss, computed in FP32.

L = dice_weight * L_Dice + ce_weight * L_CE        (default 0.5 / 0.5)

Soft Dice is computed per foreground class c = 1..C-1, summing over the whole
batch (every sample and every pixel):

    I_c = sum(p_c * g_c)      P_c = sum(p_c)      G_c = sum(g_c)
    D_c = (2 * I_c + eps) / (P_c + G_c + eps)

    p = softmax(logits), g = one-hot(target), eps = 1e-5.

* Background (c = 0) is excluded from Dice; cross-entropy supervises it.
* A class is "present" when G_c > 0 anywhere in the batch. Only present
  classes are averaged:  L_Dice = mean over present c of (1 - D_c).
* Absent classes are excluded, so false positives on them are penalised by
  cross-entropy only.
* If no foreground class is present, L_Dice = 0 (still attached to the graph)
  and the loss reduces to ce_weight * L_CE.

Cross-entropy is the unweighted mean over all pixels and all classes.

The 0.5/0.5 weighting follows the author's earlier COMP3710-Project
``losses.DiceCrossEntropyLoss``. That version averaged Dice per sample; this
version computes it per batch, as decided for this experiment.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class LossTerms(NamedTuple):
    total: torch.Tensor
    dice: torch.Tensor
    ce: torch.Tensor


def batch_dice_per_class(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (D_c, present_c) for foreground classes 1..num_classes-1.

    probabilities: [B, C, *spatial] (already softmaxed); target: [B, *spatial] int64.
    """
    one_hot = F.one_hot(target, num_classes).movedim(-1, 1).to(probabilities.dtype)
    sum_dims = (0, *range(2, probabilities.ndim))  # batch + spatial dims
    intersection = (probabilities * one_hot).sum(dim=sum_dims)
    predicted = probabilities.sum(dim=sum_dims)
    ground_truth = one_hot.sum(dim=sum_dims)
    dice = (2 * intersection + smooth) / (predicted + ground_truth + smooth)
    return dice[1:], ground_truth[1:] > 0


def batch_dice_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Mean of (1 - D_c) over present foreground classes; 0 if none are present."""
    dice, present = batch_dice_per_class(probabilities, target, num_classes, smooth)
    weights = present.to(dice.dtype)
    # Written without branching: avoids a GPU sync and keeps the result on the graph.
    return (weights * (1 - dice)).sum() / weights.sum().clamp(min=1)


class DiceCrossEntropyLoss(nn.Module):
    """Weighted batch-level foreground Dice + cross-entropy. Expects raw logits."""

    def __init__(
        self,
        num_classes: int = 6,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        if dice_weight < 0 or ce_weight < 0 or dice_weight + ce_weight <= 0:
            raise ValueError("Loss weights must be non-negative and not both zero")
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> LossTerms:
        # Leave autocast so softmax, Dice and cross-entropy all run in FP32.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits = logits.float()
            ce = F.cross_entropy(logits, target)
            probabilities = torch.softmax(logits, dim=1)
            dice = batch_dice_loss(probabilities, target, self.num_classes, self.smooth)
            total = self.dice_weight * dice + self.ce_weight * ce
        return LossTerms(total=total, dice=dice, ce=ce)
