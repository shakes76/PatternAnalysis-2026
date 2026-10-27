"""Accumulate class-ID segmentation metrics over valid pixels, not batch means."""

import torch

from dataset import IGNORE_INDEX


class SegmentationMetrics:
    """Rows are target IDs and columns are predicted IDs in a CPU int64 matrix.

    A class absent from both prediction and target has undefined Dice/IoU (None).
    Foreground macros exclude class 0 and any such undefined classes. A class
    predicted without any reference pixels has score zero, rather than None.
    Create a new accumulator for each split/evaluation epoch.
    """

    def __init__(self, num_classes=6):
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64, device="cpu"
        )

    def update(self, prediction, target, valid_mask):
        """Add [H,W] or [B,H,W] integer class IDs; ignore padding and -100."""
        if not all(isinstance(value, torch.Tensor) for value in (prediction, target, valid_mask)):
            raise TypeError("prediction, target and valid_mask must be tensors.")
        if target.ndim not in (2, 3) or prediction.shape != target.shape or valid_mask.shape != target.shape:
            raise ValueError("Expected matching [H,W] or [B,H,W] class-ID tensors.")
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if prediction.dtype not in integer_types or target.dtype not in integer_types:
            raise TypeError("prediction and target must contain integer class IDs.")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean.")
        if prediction.device != target.device or valid_mask.device != target.device:
            raise ValueError("prediction, target and valid_mask must share a device.")

        # Compare the negative ignore ID after promotion, avoiding uint8 wraparound.
        target_ids = target.to(dtype=torch.int64)
        selected = valid_mask & (target_ids != IGNORE_INDEX)
        if not selected.any().item():
            return
        actual = target_ids[selected].detach().to(device="cpu")
        predicted = prediction[selected].detach().to(device="cpu", dtype=torch.int64)
        if (actual < 0).any().item() or (actual >= self.num_classes).any().item():
            raise ValueError("Valid target class IDs are outside [0, num_classes).")
        if (predicted < 0).any().item() or (predicted >= self.num_classes).any().item():
            raise ValueError("Valid predicted class IDs are outside [0, num_classes).")

        indices = actual * self.num_classes + predicted
        counts = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """Return JSON-safe split-level scores; None means an undefined score."""
        matrix = self.confusion_matrix.to(torch.float64)
        true_positive = matrix.diag()
        actual_count = matrix.sum(dim=1)
        predicted_count = matrix.sum(dim=0)
        dice_denominator = actual_count + predicted_count
        iou_denominator = dice_denominator - true_positive
        dice = [
            float(2 * true_positive[index] / dice_denominator[index])
            if dice_denominator[index] > 0 else None
            for index in range(self.num_classes)
        ]
        iou = [
            float(true_positive[index] / iou_denominator[index])
            if iou_denominator[index] > 0 else None
            for index in range(self.num_classes)
        ]
        foreground_dice = [value for value in dice[1:] if value is not None]
        foreground_iou = [value for value in iou[1:] if value is not None]
        valid_pixels = int(self.confusion_matrix.sum().item())
        return {
            "confusion_matrix": self.confusion_matrix.tolist(),
            "dice_per_class": dice,
            "iou_per_class": iou,
            "mean_dice_excluding_class0": (
                sum(foreground_dice) / len(foreground_dice) if foreground_dice else None
            ),
            "mean_iou_excluding_class0": (
                sum(foreground_iou) / len(foreground_iou) if foreground_iou else None
            ),
            "pixel_accuracy": (
                float(true_positive.sum().item()) / valid_pixels if valid_pixels else None
            ),
            "valid_pixels": valid_pixels,
        }
