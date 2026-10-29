"""Pixel-level confusion matrix and per-class segmentation metrics."""

import torch


@torch.no_grad()
def confusion_matrix(prediction, target, num_classes=6):
    """Count pixels with rows as target classes and columns as predictions."""
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    if prediction.dtype != torch.long or target.dtype != torch.long:
        raise ValueError("Prediction and target must use torch.long label IDs")
    if prediction.device != target.device:
        raise ValueError("Prediction and target must be on the same device")
    if target.numel() == 0:
        raise ValueError("Prediction and target must not be empty")

    for name, labels in [("Prediction", prediction), ("Target", target)]:
        if not ((labels >= 0) & (labels < num_classes)).all():
            raise ValueError(f"{name} labels must be in range 0-{num_classes - 1}")

    indices = target.reshape(-1) * num_classes + prediction.reshape(-1)
    return torch.bincount(
        indices, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


@torch.no_grad()
def dice_iou_from_confusion(confusion):
    """Return per-class Dice and IoU; jointly absent classes receive NaN."""
    if (
        confusion.ndim != 2
        or confusion.shape[0] == 0
        or confusion.shape[0] != confusion.shape[1]
    ):
        raise ValueError("Confusion matrix must be non-empty and square")
    if confusion.dtype != torch.long:
        raise ValueError("Confusion matrix must contain torch.long counts")
    if (confusion < 0).any():
        raise ValueError("Confusion matrix counts must be non-negative")

    counts = confusion.to(dtype=torch.float64)
    true_positive = counts.diag()
    target_count = counts.sum(dim=1)
    prediction_count = counts.sum(dim=0)

    dice_denominator = target_count + prediction_count
    union = dice_denominator - true_positive
    absent = torch.full_like(true_positive, float("nan"))

    dice = torch.where(
        dice_denominator > 0,
        2 * true_positive / dice_denominator,
        absent,
    )
    iou = torch.where(
        union > 0,
        true_positive / union,
        absent,
    )
    return {"dice": dice, "iou": iou}
