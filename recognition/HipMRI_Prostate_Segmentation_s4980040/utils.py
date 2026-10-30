"""
utils.py - Loss functions and metrics shared by train.py and predict.py.

WHY TWO LOSSES?
    CrossEntropy (CE): scores every pixel independently: "did you pick the right class here?"
        Problem: the prostate is under 2% of the pixels. A model that never predicts
        prostate still gets ~98% of pixels right, so CE alone barely cares about it.
    Dice loss: measures OVERLAP per class, with every class weighted equally no matter
        how small it is. This forces the model to take the tiny prostate seriously.
    Using CE + Dice together is a common, robust choice in medical segmentation.

THE DICE SCORE
    Dice = 2 * |prediction AND truth| / (|prediction| + |truth|)
    1.0 = perfect overlap, 0.0 = no overlap at all. Our target: >= 0.75 on prostate.
    IoU (Intersection over Union) = |prediction AND truth| / |prediction OR truth|
    is a stricter cousin of Dice; the spec asks us to report both.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Index in this list = label number in the masks
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss (1 - average Dice over the foreground classes).

    "Soft" means it uses predicted probabilities instead of hard 0/1 labels, so it is
    smooth and the model can learn from it with gradients.
    """

    def __init__(self, num_classes=6, smooth=1.0, ignore_background=True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth  # added to top and bottom so we never divide by 0
        self.ignore_background = ignore_background  # background is easy and huge; skip it

    def forward(self, logits, target):
        # logits: (B, C, H, W) raw scores    target: (B, H, W) class numbers 0..5
        probs = F.softmax(logits.float(), dim=1)  # scores -> probabilities that sum to 1 per pixel

        # Turn target labels into one channel per class, e.g. label 5 -> [0,0,0,0,0,1]
        # one_hot gives (B, H, W, C); permute moves C next to B to match probs: (B, C, H, W)
        one_hot = F.one_hot(target, self.num_classes).movedim(-1, 1).float()

        dims = (0,) + tuple(range(2, probs.ndim))  # sum over batch + all spatial dims (2D or 3D)
        intersection = (probs * one_hot).sum(dims)
        total = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (total + self.smooth)  # shape (C,)

        if self.ignore_background:
            dice = dice[1:]  # drop class 0
        return 1 - dice.mean()  # loss goes DOWN as overlap goes UP


class CombinedLoss(nn.Module):
    """CrossEntropy + dice_weight * Dice loss (the loss we actually train with)."""

    def __init__(self, num_classes=6, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()  # applies softmax internally, so the model outputs raw logits
        self.dice = DiceLoss(num_classes)
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


class SegmentationMetrics:
    """Collects per-class pixel counts over a WHOLE split, then computes Dice and IoU.

    Counts are summed across all slices before dividing ("global" Dice). This avoids a
    problem with averaging per slice: slices with no prostate at all would give
    0/0 and distort the average.

    Usage:
        m = SegmentationMetrics(6)
        for each batch: m.update(logits, masks)
        m.dice()  -> [dice_class0, ..., dice_class5]
    """

    def __init__(self, num_classes=6):
        self.num_classes = num_classes
        self.intersection = torch.zeros(num_classes, dtype=torch.long)  # pixels where pred == truth == c
        self.predicted = torch.zeros(num_classes, dtype=torch.long)     # pixels predicted as c
        self.actual = torch.zeros(num_classes, dtype=torch.long)        # pixels that truly are c
        self.count = 0  # number of slices seen

    @torch.no_grad()  # metrics never need gradients
    def update(self, logits, target):
        preds = logits.argmax(dim=1)  # highest score wins -> (B, H, W) predicted labels
        for c in range(self.num_classes):
            p, t = preds == c, target == c  # True/False maps for "is class c"
            self.intersection[c] += (p & t).sum().item()
            self.predicted[c] += p.sum().item()
            self.actual[c] += t.sum().item()
        self.count += target.shape[0]

    def dice(self):
        """Dice per class, as a list of 6 floats."""
        denom = (self.predicted + self.actual).clamp(min=1)  # clamp avoids divide-by-zero
        return (2 * self.intersection / denom).tolist()

    def iou(self):
        """IoU per class, as a list of 6 floats."""
        union = (self.predicted + self.actual - self.intersection).clamp(min=1)
        return (self.intersection / union).tolist()