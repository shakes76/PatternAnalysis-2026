"""Pure valid-pixel CE and ID5 overlap diagnostics; no model or data loading.

The CE breakdown groups pixels by their true class. Each pixel contributes
only -log(p_true), not a sum of six class errors. CPU float64 sums can be
combined across batches before computing class means and global contributions.
"""

import torch
import torch.nn.functional as F

from losses import segmentation_cross_entropy


def classwise_ce_statistics(logits, target, valid_mask, num_classes=6, return_pixel_nll=False):
    """Return true-class support and NLL sums, with a shared-loss consistency check.

    Inputs follow the shared loss contract: logits [B,C,H,W], int64 class-ID
    target [B,H,W], boolean valid_mask, and -100 targets at padded locations.
    The local gather uses a safe ID at padding before selecting valid pixels.
    Absent classes return support=0 and nll_sum=0, not a fabricated class mean.
    By default the returned Python numbers are JSON-safe. With
    return_pixel_nll=True, also return pixel_nll_cpu: a detached CPU float64
    [B,H,W] tensor with zero at padding, for per-slice/fixed-set aggregation.
    It reuses the same gathered NLL values, without another diagnostic forward
    or log_softmax. The caller must aggregate or remove this tensor before JSON.
    """
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 2:
        raise ValueError("num_classes must be an integer >=2.")
    if not isinstance(return_pixel_nll, bool):
        raise TypeError("return_pixel_nll must be boolean.")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[1] != num_classes:
        raise ValueError("Expected logits [B,num_classes,H,W].")
    if not logits.is_floating_point():
        raise TypeError("Diagnostic logits must be floating-point scores.")
    with torch.no_grad():
        # This validates target IDs, padding, finiteness, shape and device via
        # the same authoritative loss actually used during training/validation.
        shared_ce = segmentation_cross_entropy(logits, target, valid_mask)
        if not torch.isfinite(shared_ce).item():
            raise ValueError("The shared CE is non-finite; no breakdown is accepted.")
        safe_target = target.masked_fill(~valid_mask, 0)
        per_pixel_nll = -F.log_softmax(logits.float(), dim=1).gather(
            1, safe_target.unsqueeze(1)
        ).squeeze(1)
        actual = target[valid_mask].detach().to(device="cpu", dtype=torch.int64)
        if return_pixel_nll:
            pixel_nll_cpu = per_pixel_nll.masked_fill(~valid_mask, 0).detach().to(
                device="cpu", dtype=torch.float64
            )
            nll = pixel_nll_cpu[valid_mask.to(device="cpu")]
        else:
            nll = per_pixel_nll[valid_mask].detach().to(device="cpu", dtype=torch.float64)
        support = torch.bincount(actual, minlength=num_classes)
        nll_sum = torch.bincount(actual, weights=nll, minlength=num_classes)
        valid_pixels = int(support.sum().item())
        total = nll_sum.sum()
        reconstructed_ce = total / valid_pixels
        shared_cpu = shared_ce.detach().to(device="cpu", dtype=torch.float64)
        # The authoritative loss reduces in float32; decomposition reduces its
        # per-pixel float32 NLL values in float64. A tiny rounding gap is expected.
        torch.testing.assert_close(
            reconstructed_ce, shared_cpu, rtol=1e-5, atol=1e-6
        )
        result = {
            "support": [int(value) for value in support.tolist()],
            "nll_sum": [float(value) for value in nll_sum.tolist()],
            "valid_pixels": valid_pixels,
            "nll_sum_total": float(total.item()),
            "ce_reconstructed": float(reconstructed_ce.item()),
            "ce_shared": float(shared_cpu.item()),
            "ce_abs_difference": float((reconstructed_ce - shared_cpu).abs().item()),
        }
        if return_pixel_nll:
            result["pixel_nll_cpu"] = pixel_nll_cpu
        return result


def class5_overlap(prediction, target, valid_mask):
    """Count ID5 TP/FP/FN in valid pixels of matching [H,W] or [B,H,W] IDs.

    This is pixel-pooled overlap for numeric class 5, with no tissue-name or
    per-case interpretation added. Values at invalid positions are ignored.
    Dice is None when both reference/prediction are empty; precision is None
    with no predictions and recall is None with no reference positives.
    """
    if not all(isinstance(value, torch.Tensor) for value in (prediction, target, valid_mask)):
        raise TypeError("prediction, target and valid_mask must be tensors.")
    if target.ndim not in (2, 3) or prediction.shape != target.shape or valid_mask.shape != target.shape:
        raise ValueError("Expected matching [H,W] or [B,H,W] tensors.")
    integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if prediction.dtype not in integer_types or target.dtype not in integer_types:
        raise TypeError("Prediction and target must contain integer class IDs.")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean.")
    if prediction.device != target.device or valid_mask.device != target.device:
        raise ValueError("Prediction, target and valid_mask must share a device.")
    with torch.no_grad():
        actual = target[valid_mask].to(dtype=torch.int64)
        predicted = prediction[valid_mask].to(dtype=torch.int64)
        if ((actual < 0) | (actual > 5)).any().item() or ((predicted < 0) | (predicted > 5)).any().item():
            raise ValueError("Valid pixels must contain the six raw IDs 0..5.")
        reference_positive = actual == 5
        predicted_positive = predicted == 5
        tp = int((reference_positive & predicted_positive).sum().item())
        fp = int((~reference_positive & predicted_positive).sum().item())
        fn = int((reference_positive & ~predicted_positive).sum().item())
        reference_pixels, predicted_pixels = tp + fn, tp + fp
        denominator = reference_pixels + predicted_pixels
        return {
            "class_id": 5,
            "tp": tp, "fp": fp, "fn": fn,
            "reference_pixels": reference_pixels,
            "predicted_pixels": predicted_pixels,
            "valid_pixels": int(actual.numel()),
            "dice": 2 * tp / denominator if denominator else None,
            "precision": tp / predicted_pixels if predicted_pixels else None,
            "recall": tp / reference_pixels if reference_pixels else None,
        }
