"""Compute slice-wise segmentation metrics from integer label maps."""

import torch


@torch.no_grad()
def positive_slice_stats(prediction, target, class_id=5):
    """Return Dice/IoU sums and counts for ground-truth-positive slices.

    Inputs must be integer label maps of shape [B, H, W].
    Ground-truth-negative slices are excluded from Dice and IoU.
    Eligible slices with no predicted pixels receive zero.
    """
    if prediction.shape != target.shape or target.ndim != 3:
        raise ValueError("Expected matching label maps with shape [B, H, W].")
    if prediction.device != target.device:
        raise ValueError("Prediction and target must be on the same device.")
    if prediction.dtype != torch.int64 or target.dtype != torch.int64:
        raise TypeError("Label maps must use torch.int64.")
    if not 0 <= class_id < 6:
        raise ValueError("class_id must be between 0 and 5.")
    if (
        ((prediction < 0) | (prediction > 5)).any()
        or ((target < 0) | (target > 5)).any()
    ):
        raise ValueError("Labels must be between 0 and 5.")

    predicted = prediction == class_id
    actual = target == class_id

    predicted_count = predicted.sum(dim=(1, 2)).to(torch.float64)
    actual_count = actual.sum(dim=(1, 2)).to(torch.float64)
    intersection = (predicted & actual).sum(dim=(1, 2)).to(torch.float64)

    eligible = actual_count > 0
    negative = ~eligible

    # Eligible targets are nonempty, so denominators are strictly positive.
    dice = (
        2 * intersection[eligible]
        / (predicted_count[eligible] + actual_count[eligible])
    )
    union = predicted_count + actual_count - intersection
    iou = intersection[eligible] / union[eligible]

    return {
        "dice_sum": dice.sum().item(),
        "iou_sum": iou.sum().item(),
        "eligible_slices": int(eligible.sum().item()),
        "excluded_slices": int(negative.sum().item()),
        "negative_slices_with_fp": int(
            (negative & (predicted_count > 0)).sum().item()
        ),
        "negative_slice_fp_pixels": int(
            predicted_count[negative].sum().item()
        ),
    }


def summarize_stats(stats):
    """Compute means only after accumulating sums across all batches."""
    count = stats["eligible_slices"]
    return {
        **stats,
        "mean_dice": stats["dice_sum"] / count if count else None,
        "mean_iou": stats["iou_sum"] / count if count else None,
    }
