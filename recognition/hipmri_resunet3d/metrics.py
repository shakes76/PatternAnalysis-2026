"""Case-level overlap and boundary metrics; all inputs are binary arrays."""

import numpy as np
from scipy import ndimage as ndi


def _overlap(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(pred, truth).sum())
    p, t = int(pred.sum()), int(truth.sum())
    if p + t == 0:
        return 1.0, 1.0
    return 2 * intersection / (p + t), intersection / max(p + t - intersection, 1)


def _hd95(
    pred: np.ndarray, truth: np.ndarray,
    spacing_dhw: tuple[float, float, float] | None = None,
) -> float:
    spacing = np.ones(3) if spacing_dhw is None else np.asarray(spacing_dhw)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Voxel spacing must contain three positive finite values")
    if not pred.any() and not truth.any():
        return 0.0
    if not pred.any() or not truth.any():
        # Penalize complete misses rather than silently dropping them from the mean.
        return float(np.linalg.norm((np.asarray(pred.shape) - 1) * spacing))
    structure = ndi.generate_binary_structure(3, 1)
    p_border = np.logical_xor(pred, ndi.binary_erosion(pred, structure=structure))
    t_border = np.logical_xor(truth, ndi.binary_erosion(truth, structure=structure))
    p_to_t = ndi.distance_transform_edt(~t_border, sampling=spacing)[p_border]
    t_to_p = ndi.distance_transform_edt(~p_border, sampling=spacing)[t_border]
    return float(np.percentile(np.concatenate((p_to_t, t_to_p)), 95))


def case_metrics(
    pred: np.ndarray, truth: np.ndarray, anatomy: np.ndarray | None = None,
    bladder_label: int | None = None, rectum_label: int | None = None,
    spacing_dhw: tuple[float, float, float] | None = None,
) -> dict[str, float | None]:
    pred, truth = np.asarray(pred, dtype=bool), np.asarray(truth, dtype=bool)
    if pred.shape != truth.shape or pred.ndim != 3:
        raise ValueError("Expected equal [D,H,W] binary arrays")
    prostate_dice, prostate_iou = _overlap(pred, truth)
    background_dice, background_iou = _overlap(~pred, ~truth)
    active = np.flatnonzero(truth.any(axis=(1, 2)))
    if len(active):
        width = max(1, round(len(active) * 0.2))
        apex = active[:width]
        base = active[-width:]
        apex_dice = float(np.mean([_overlap(pred[z], truth[z])[0] for z in apex]))
        base_dice = float(np.mean([_overlap(pred[z], truth[z])[0] for z in base]))
        outside_extent = np.ones(pred.shape[0], dtype=bool)
        outside_extent[active[0]:active[-1] + 1] = False
        outside_extent_fp_voxels = int(pred[outside_extent].sum())
        outside_extent_fp_slices = int(pred[outside_extent].any(axis=(1, 2)).sum())
    else:
        apex_dice = base_dice = None
        outside_extent_fp_voxels = outside_extent_fp_slices = None
    results = {
        "prostate_dice": prostate_dice,
        "prostate_iou": prostate_iou,
        "background_dice": background_dice,
        "background_iou": background_iou,
        "apex_dice": apex_dice,
        "base_dice": base_dice,
        "outside_extent_false_positive_voxels": outside_extent_fp_voxels,
        "outside_extent_false_positive_slices": outside_extent_fp_slices,
        "hd95_mm" if spacing_dhw else "hd95_resampled_voxels": _hd95(
            pred, truth, spacing_dhw
        ),
        "false_positive_voxels": int(np.logical_and(pred, ~truth).sum()),
        "false_negative_voxels": int(np.logical_and(~pred, truth).sum()),
    }
    if anatomy is not None:
        if anatomy.shape != truth.shape:
            raise ValueError("Anatomy shape must match truth")
        for name, label in (("bladder", bladder_label), ("rectum", rectum_label)):
            if label is not None:
                region = anatomy == label
                count = int(np.logical_and(pred, region).sum())
                total = int(region.sum())
                results[f"false_positive_{name}_voxels"] = count
                results[f"{name}_voxels"] = total
                results[f"false_positive_{name}_fraction"] = count / total if total else None
    return results
