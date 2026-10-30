"""
utils.py - Evaluation helpers shared by train.py and predict.py.

    * dice_iou_per_class   - volumetric Sorensen-Dice and IoU per class
    * extremity_dice       - prostate Dice on base/apex slices vs central slices
    * predict_volume       - run either model on one (1, D, H, W) volume
    * evaluate             - metrics over a list of cases, both models

Dice is computed on the whole 3D volume for BOTH models (the 2D baseline's
slice predictions are stacked back into a volume first), so the comparison is
like-for-like.
"""

import time

import torch

from dataset import NUM_CLASSES, PROSTATE


def amp_enabled(device) -> bool:
    """Use bfloat16 mixed precision only on GPUs that support it (e.g. A100)."""
    return device.type == "cuda" and torch.cuda.is_bf16_supported()


def dice_iou_per_class(pred: torch.Tensor, target: torch.Tensor,
                       num_classes: int = NUM_CLASSES):
    """
    Dice and IoU for each class on integer label volumes of equal shape.

    A class absent from both prediction and ground truth gives NaN (it is
    excluded from averages rather than counted as a perfect score).
    """
    dice = torch.full((num_classes,), float("nan"))
    iou = torch.full((num_classes,), float("nan"))
    for c in range(num_classes):
        p, t = pred == c, target == c
        inter = (p & t).sum().item()
        p_sum, t_sum = p.sum().item(), t.sum().item()
        if p_sum + t_sum == 0:
            continue
        dice[c] = 2 * inter / (p_sum + t_sum)
        iou[c] = inter / (p_sum + t_sum - inter)
    return dice, iou


def extremity_dice(pred: torch.Tensor, target: torch.Tensor,
                   label: int = PROSTATE, frac: float = 0.25):
    """
    Split the prostate's slice range (from the ground truth) into its two
    extremities (outer `frac` of slices at each end: the superior base and
    inferior apex) and the central remainder, and compute Dice on each part.

    These extremity slices are where the brief's dilemma says 2D models fail:
    the prostate is small and its boundary with bladder/rectum is ambiguous.

    pred, target: (D, H, W) integer volumes. Returns (extremity, central) Dice
    or (nan, nan) if the prostate is absent from the ground truth.
    """
    slices = torch.nonzero((target == label).flatten(1).any(1)).flatten()
    if len(slices) == 0:
        return float("nan"), float("nan")
    lo, hi = slices.min().item(), slices.max().item()
    n = hi - lo + 1
    k = max(1, round(n * frac))
    ext = torch.zeros(target.shape[0], dtype=torch.bool)
    ext[lo:lo + k] = True
    ext[hi - k + 1:hi + 1] = True
    cen = torch.zeros_like(ext)
    cen[lo:hi + 1] = True
    cen &= ~ext

    def _dice(mask):
        p = (pred[mask] == label)
        t = (target[mask] == label)
        denom = p.sum().item() + t.sum().item()
        return 2 * (p & t).sum().item() / denom if denom else float("nan")

    return _dice(ext), _dice(cen)


@torch.no_grad()
def predict_volume(model, image: torch.Tensor, mode: str, device,
                   slice_batch: int = 32) -> torch.Tensor:
    """
    Predict a label volume.

    image: (1, D, H, W) float tensor.
    mode:  '3d' -> whole volume in one pass; '2d' -> axial slices in batches,
           then restacked.
    Returns logits (C, D, H, W) on CPU (float32).
    """
    model.eval()
    if mode == "3d":
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp_enabled(device)):
            out = model(image[None].to(device))[0]
        return out.float().cpu()

    slices = image[0][:, None]                       # (D, 1, H, W)
    outs = []
    for i in range(0, slices.shape[0], slice_batch):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp_enabled(device)):
            outs.append(model(slices[i:i + slice_batch].to(device)).float().cpu())
    return torch.cat(outs).permute(1, 0, 2, 3)        # (C, D, H, W)


def evaluate(model, data: dict, indices: list, mode: str, device,
             keep_predictions: bool = False) -> dict:
    """
    Evaluate a model on the given case indices.

    Returns a dict with per-case Dice/IoU (N, C), mean Dice/IoU per class,
    prostate extremity vs central Dice, mean inference time per volume, and
    optionally the predicted label volumes.
    """
    dices, ious, ext, cen, times, preds = [], [], [], [], [], {}
    for idx in indices:
        image = data["images"][idx].float()[None]
        target = data["labels"][idx].long()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = predict_volume(model, image, mode, device).argmax(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        d, i = dice_iou_per_class(pred, target)
        e, c = extremity_dice(pred, target)
        dices.append(d)
        ious.append(i)
        ext.append(e)
        cen.append(c)
        if keep_predictions:
            preds[data["keys"][idx]] = pred.to(torch.uint8)

    dices, ious = torch.stack(dices), torch.stack(ious)
    result = {
        "keys": [data["keys"][i] for i in indices],
        "dice_per_case": dices,
        "iou_per_case": ious,
        "dice_mean": torch.nanmean(dices, 0),
        "iou_mean": torch.nanmean(ious, 0),
        "prostate_extremity_dice": torch.nanmean(torch.tensor(ext)).item(),
        "prostate_central_dice": torch.nanmean(torch.tensor(cen)).item(),
        "prostate_extremity_per_case": ext,
        "seconds_per_volume": sum(times[1:]) / max(1, len(times) - 1),  # skip warm-up
    }
    if keep_predictions:
        result["predictions"] = preds
    return result
