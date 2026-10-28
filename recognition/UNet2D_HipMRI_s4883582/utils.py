"""
Segmentation metrics shared by train.py and predict.py.

Dice and IoU are computed from intersection / size counts accumulated over the whole split,
rather than averaging per-batch scores (which is biased by batches where a class is absent).
"""
import torch


def overlap_counts(logits, targets, n_classes):
    """Per-class intersection, predicted size and true size for one batch."""
    preds = torch.argmax(logits, dim=1)
    intersection = torch.zeros(n_classes, dtype=torch.float64)
    pred_size = torch.zeros(n_classes, dtype=torch.float64)
    true_size = torch.zeros(n_classes, dtype=torch.float64)

    for cls in range(n_classes):
        pred_mask = preds == cls
        true_mask = targets == cls
        intersection[cls] = (pred_mask & true_mask).sum().item()
        pred_size[cls] = pred_mask.sum().item()
        true_size[cls] = true_mask.sum().item()

    return intersection, pred_size, true_size


def dice_and_iou(intersection, pred_size, true_size, smooth=1e-6):
    """Per-class Dice = 2|A∩B| / (|A|+|B|) and IoU = |A∩B| / |A∪B| from accumulated counts."""
    dice = (2. * intersection + smooth) / (pred_size + true_size + smooth)
    iou = (intersection + smooth) / (pred_size + true_size - intersection + smooth)
    return dice, iou
