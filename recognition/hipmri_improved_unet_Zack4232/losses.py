"""Training losses for six-class HipMRI segmentation."""

import torch
from torch import nn
from torch.nn import functional as F


class CrossEntropyDiceLoss(nn.Module):
    """Combine cross-entropy with foreground soft Dice loss.

    Soft Dice is computed separately for each slice and foreground
    class (1-5). Only pairs with nonempty ground truth are averaged.
    Cross-entropy covers all pixels, including absent-class errors.
    """

    def __init__(self, dice_weight=1.0):
        super().__init__()
        if dice_weight < 0:
            raise ValueError("dice_weight must be nonnegative.")
        self.dice_weight = float(dice_weight)

    def components(self, logits, target):
        """Return cross-entropy and soft Dice loss separately."""
        if logits.ndim != 4 or logits.shape[1] != 6:
            raise ValueError("Expected logits with shape [B, 6, H, W].")
        expected = (logits.shape[0], *logits.shape[2:])
        if tuple(target.shape) != expected:
            raise ValueError("Target shape must be [B, H, W].")
        if target.dtype != torch.int64:
            raise TypeError("Target must use torch.int64.")

        # Use float32 for stable loss calculations.
        logits_float = logits.float()
        ce = F.cross_entropy(logits_float, target)
        probabilities = logits_float.softmax(dim=1)
        encoded = F.one_hot(target, num_classes=6).permute(0, 3, 1, 2)
        encoded = encoded.to(dtype=probabilities.dtype)

        probabilities = probabilities[:, 1:]
        encoded = encoded[:, 1:]

        intersection = (probabilities * encoded).sum(dim=(2, 3))
        predicted_mass = probabilities.sum(dim=(2, 3))
        target_mass = encoded.sum(dim=(2, 3))
        present = target_mass > 0

        if present.any():
            # Selected denominators are positive because targets are nonempty.
            dice = (
                2.0 * intersection[present]
                / (predicted_mass[present] + target_mass[present])
            )
            dice_loss = (1.0 - dice).mean()
        else:
            # Retain a valid computation graph for background-only batches.
            dice_loss = probabilities.sum() * 0.0

        return ce, dice_loss

    def forward(self, logits, target):
        ce, dice_loss = self.components(logits, target)
        return ce + self.dice_weight * dice_loss
