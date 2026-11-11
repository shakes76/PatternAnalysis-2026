"""
Shared helpers for training and prediction: device selection, the Dice +
cross-entropy loss, and per-class Dice / IoU metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import NUM_CLASSES, one_hot


def get_device(preferred=None):
    """Pick a device: the one requested, else CUDA, else Apple MPS, else CPU."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DiceCELoss(nn.Module):
    """
    Cross-entropy plus soft multi-class Dice loss.

    Cross-entropy gives stable per-pixel gradients early in training, while
    the Dice term counters class imbalance (the prostate is under 1% of
    pixels). The Dice term averages over foreground classes only, so the
    large, easy background class does not dominate it.
    """

    def __init__(self, num_classes=NUM_CLASSES, dice_weight=1.0, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target)

        probs = logits.softmax(dim=1)
        target_1h = one_hot(target, self.num_classes)
        dims = (0, 2, 3)  # sum over batch and pixels, keep classes
        intersection = (probs * target_1h).sum(dims)
        denom = probs.sum(dims) + target_1h.sum(dims)
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        dice_loss = 1 - dice[1:].mean()

        return ce + self.dice_weight * dice_loss


class SegmentationMeter:
    """
    Accumulates per-class overlap counts over a whole split, then reports
    Dice and IoU per class. Counting over all pixels of the split (rather
    than averaging per-slice scores) avoids undefined scores on slices where
    a structure is absent.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(self.num_classes, dtype=torch.float64)
        self.pred_sum = torch.zeros(self.num_classes, dtype=torch.float64)
        self.target_sum = torch.zeros(self.num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, target):
        pred = logits.argmax(dim=1)
        for c in range(self.num_classes):
            p, t = pred == c, target == c
            self.intersection[c] += (p & t).sum().item()
            self.pred_sum[c] += p.sum().item()
            self.target_sum[c] += t.sum().item()

    def dice(self):
        return (2 * self.intersection / (self.pred_sum + self.target_sum).clamp(min=1)).tolist()

    def iou(self):
        union = self.pred_sum + self.target_sum - self.intersection
        return (self.intersection / union.clamp(min=1)).tolist()
