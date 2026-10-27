"""Valid-region one-hot encoding and cross-entropy for the baseline."""

import torch
import torch.nn.functional as F

from dataset import IGNORE_INDEX


def masked_one_hot(target, valid_mask, num_classes):
    """Encode [B,H,W] IDs as [B,C,H,W], with all-zero channels on padding."""
    if target.ndim != 3 or target.shape != valid_mask.shape:
        raise ValueError("Expected matching [B,H,W] targets and valid_mask.")
    if target.dtype != torch.long or valid_mask.dtype != torch.bool:
        raise TypeError("Expected int64 targets and boolean valid_mask.")
    if target.device != valid_mask.device or num_classes < 2:
        raise ValueError("Targets/mask must share a device; num_classes must be >=2.")
    if not valid_mask.any():
        raise ValueError("At least one valid pixel is required.")
    selected = target[valid_mask]
    if (selected < 0).any() or (selected >= num_classes).any():
        raise ValueError("Valid target class IDs are outside the configured range.")
    if (target[~valid_mask] != IGNORE_INDEX).any():
        raise ValueError("Invalid target pixels must contain the ignore index.")
    safe_target = target.masked_fill(~valid_mask, 0)
    encoded = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2)
    return encoded.to(torch.float32) * valid_mask.unsqueeze(1)


def segmentation_cross_entropy(logits, target, valid_mask):
    """One-hot CE averaged over valid pixels; equivalent to ignore-index CE."""
    if logits.ndim != 4 or logits.shape[0] != target.shape[0]:
        raise ValueError("Expected logits [B,C,H,W] and targets [B,H,W].")
    if logits.shape[-2:] != target.shape[-2:] or logits.device != target.device:
        raise ValueError("Logits and targets must share spatial shape and device.")
    if not torch.isfinite(logits).all():
        raise ValueError("Logits must be finite.")
    one_hot = masked_one_hot(target, valid_mask, logits.shape[1])
    per_pixel = -(one_hot * F.log_softmax(logits.float(), dim=1)).sum(dim=1)
    return per_pixel[valid_mask].mean()
