"""Sliding-window inference for full MRI volumes with a patch-based 3D model.

1. If any side of the volume is smaller than the patch, the volume is
   zero-padded at the end of that axis (0 is the normalised background value).
2. Windows start every ``patch * (1 - overlap)`` voxels (50% overlap by
   default); the last window on each axis is placed flush with the edge, so
   every voxel is covered.
3. Window logits are summed in FP32 with uniform weight 1 and divided by the
   number of windows covering each voxel (uniform averaging). A voxel covered
   by no window raises an error instead of returning zeros.
4. The padding is cropped off, giving logits with the volume's own shape.

Layout follows PyTorch: image [1, D, H, W] in, logits [C, D, H, W] out.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product

import torch
from torch.nn import functional as F


def window_starts(length: int, patch: int, overlap: float = 0.5) -> list[int]:
    """Start indices along one axis; the last window ends exactly at ``length``."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if length <= patch:
        return [0]
    stride = max(1, int(patch * (1 - overlap)))
    starts = list(range(0, length - patch, stride))
    return [*starts, length - patch]


def pad_to_patch(image: torch.Tensor, size: Sequence[int]) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Zero-pad [C, D, H, W] at the end of any axis shorter than the patch.

    Returns the padded tensor and the original (D, H, W) for cropping.
    """
    original = tuple(image.shape[1:])
    extra = [max(p - n, 0) for n, p in zip(original, size, strict=True)]
    # F.pad lists the last axis first: (w_before, w_after, h_before, h_after, d_before, d_after).
    padding = (0, extra[2], 0, extra[1], 0, extra[0])
    return F.pad(image, padding), original


@torch.no_grad()
def sliding_window_logits(
    predict: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    size: Sequence[int],
    num_classes: int,
    overlap: float = 0.5,
    batch_size: int = 1,
) -> torch.Tensor:
    """Averaged logits [num_classes, D, H, W] (FP32) for one image [1, D, H, W].

    ``predict`` maps a batch [n, 1, *size] to logits [n, num_classes, *size];
    it is typically the model, wrapped in autocast by the caller.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected image [C, D, H, W], got {tuple(image.shape)}")
    padded, original = pad_to_patch(image, size)
    shape = padded.shape[1:]
    logit_sum = torch.zeros((num_classes, *shape), dtype=torch.float32, device=image.device)
    count = torch.zeros(shape, dtype=torch.float32, device=image.device)

    corners = list(
        product(*(window_starts(n, p, overlap) for n, p in zip(shape, size, strict=True)))
    )
    for first in range(0, len(corners), batch_size):
        chunk = corners[first : first + batch_size]
        boxes = [
            tuple(slice(s, s + p) for s, p in zip(corner, size, strict=True)) for corner in chunk
        ]
        logits = predict(torch.stack([padded[(slice(None), *box)] for box in boxes])).float()
        for box, window_logits in zip(boxes, logits, strict=True):
            logit_sum[(slice(None), *box)] += window_logits
            count[box] += 1

    if (count == 0).any():
        raise RuntimeError(f"{int((count == 0).sum())} voxels were not covered by any window")
    averaged = logit_sum / count
    d, h, w = original
    return averaged[:, :d, :h, :w]
