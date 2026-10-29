"""Shared loss functions, segmentation metrics, inference and file utilities.

Metrics use the downsampled physical grid after model padding is removed.
Patient-level scores combine visits before averaging across patients.
"""
from __future__ import annotations
# Integer-count overlap metrics, reusable for patient and pooled aggregation.


import numpy as np

NUM_CLASSES = 6


def confusion(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if target.shape != prediction.shape or target.ndim != 3:
        raise ValueError("Expected matching 3D target and prediction")
    if target.dtype.kind not in "iu" or prediction.dtype.kind not in "iu":
        raise ValueError("Expected integer labels")
    if np.any((target < 0) | (target >= NUM_CLASSES) | (prediction < 0) | (prediction >= NUM_CLASSES)):
        raise ValueError("Labels must be 0..5")
    encoded = NUM_CLASSES * target.astype(np.int64).ravel() + prediction.astype(np.int64).ravel()
    return np.bincount(encoded, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def overlap(matrix: np.ndarray) -> dict[str, list[float | None]]:
    """Rows are truth; columns are prediction; absent/absent metrics are null."""
    if matrix.shape != (NUM_CLASSES, NUM_CLASSES) or np.any(matrix < 0):
        raise ValueError("Expected nonnegative 6x6 confusion counts")
    truth = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = np.diag(matrix)
    result: dict[str, list[float | None]] = {"dice": [], "iou": [], "precision": [], "recall": []}
    for c in range(NUM_CLASSES):
        def ratio(numerator, denominator):
            return None if denominator == 0 else float(numerator / denominator)
        result["dice"].append(ratio(2 * tp[c], truth[c] + predicted[c]))
        result["iou"].append(ratio(tp[c], truth[c] + predicted[c] - tp[c]))
        result["precision"].append(ratio(tp[c], predicted[c]))
        result["recall"].append(ratio(tp[c], truth[c]))
    return result

# Explicit six-class objectives; objective choice must be fixed per run.


import torch
from torch.nn import functional as F

NUM_CLASSES = 6


def _check(logits: torch.Tensor, target: torch.Tensor) -> None:
    if logits.ndim not in (4, 5) or logits.shape[1] != NUM_CLASSES:
        raise ValueError("Expected 2D/3D six-class logits")
    if target.shape != (logits.shape[0], *logits.shape[2:]) or target.dtype != torch.long:
        raise ValueError("Expected matching integer class-index target")
    if not torch.isfinite(logits).all() or ((target < 0) | (target >= NUM_CLASSES)).any():
        raise ValueError("Non-finite logits or labels outside 0..5")


def foreground_soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    _check(logits, target)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    probability = logits.softmax(dim=1)
    onehot = F.one_hot(target, NUM_CLASSES).movedim(-1, 1).to(probability.dtype)
    axes = (0, *range(2, logits.ndim))
    intersection = (probability * onehot).sum(dim=axes)
    denominator = probability.sum(dim=axes) + onehot.sum(dim=axes)
    dice = (2 * intersection + epsilon) / (denominator + epsilon)
    return 1 - dice[1:].mean()


def ce_dice_loss(logits: torch.Tensor, target: torch.Tensor, *, ce_weight: float = 0.5,
                 class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Combine weighted cross-entropy and foreground soft Dice with the given weight."""
    _check(logits, target)
    if not 0 <= ce_weight <= 1:
        raise ValueError("ce_weight must be in [0,1]")
    ce = F.cross_entropy(logits, target, weight=class_weights)
    dice = foreground_soft_dice_loss(logits, target)
    return ce_weight * ce + (1 - ce_weight) * dice

# Geometry-aware prostate boundary diagnostics for full XYZ volumes.


import numpy as np


def prostate_axial_regions(truth_xyz: np.ndarray, affine: np.ndarray,
                            end_fraction: float = 0.2) -> dict[str, tuple[int, ...]]:
    """Partition truth-positive axial slices by physical superior position.

    The superior end is called base and inferior end apex. Refuse an oblique
    grid whose in-plane axes materially change superior coordinate. This is a
    geometric proxy; selected cases still need anatomical image review.
    """
    if truth_xyz.ndim != 3 or affine.shape != (4, 4) or not 0 < end_fraction < 0.5:
        raise ValueError("Invalid volume, affine, or end fraction")
    if not np.isfinite(affine).all() or abs(affine[2, 2]) < 1e-8:
        raise ValueError("Invalid superior axis")
    if max(abs(affine[2, 0]), abs(affine[2, 1])) > 1e-3 * abs(affine[2, 2]):
        raise ValueError("Oblique grid requires anatomical reorientation review")
    indices = np.flatnonzero((truth_xyz == 5).any(axis=(0, 1)))
    if len(indices) < 5:
        raise ValueError("Too few prostate-positive slices to partition")
    superior_mm = affine[2, 2] * indices + affine[2, 3]
    order = indices[np.argsort(superior_mm)]
    end_count = max(1, int(np.ceil(len(order) * end_fraction)))
    if 2 * end_count >= len(order):
        raise ValueError("No central prostate slices remain")
    return {"apex": tuple(int(v) for v in order[:end_count]),
            "central": tuple(int(v) for v in order[end_count:-end_count]),
            "base": tuple(int(v) for v in order[-end_count:])}


def slice_dice(truth_xyz: np.ndarray, prediction_xyz: np.ndarray, label: int = 5) -> list[float | None]:
    """Per-array-slice overlap; absent/absent is null, not a perfect score."""
    if truth_xyz.shape != prediction_xyz.shape or truth_xyz.ndim != 3:
        raise ValueError("Expected matching XYZ volumes")
    values = []
    for z in range(truth_xyz.shape[2]):
        truth = truth_xyz[:, :, z] == label
        predicted = prediction_xyz[:, :, z] == label
        denominator = int(truth.sum() + predicted.sum())
        values.append(None if denominator == 0 else float(2 * (truth & predicted).sum() / denominator))
    return values


def symmetric_surface_distances(truth_xyz: np.ndarray, prediction_xyz: np.ndarray,
                                spacing_mm: tuple[float, float, float], label: int = 5) -> dict:
    """HD95 and mean symmetric surface distance in mm for a specified class.

    One-empty masks yield status `one_empty` and null distances; both-empty
    masks yield `both_empty`. They must be counted separately in reports.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    if truth_xyz.shape != prediction_xyz.shape or truth_xyz.ndim != 3:
        raise ValueError("Expected matching XYZ volumes")
    if len(spacing_mm) != 3 or any(not np.isfinite(s) or s <= 0 for s in spacing_mm):
        raise ValueError("Spacing must be three positive finite millimetre values")
    truth = truth_xyz == label
    prediction = prediction_xyz == label
    if not truth.any() or not prediction.any():
        return {"status": "both_empty" if not truth.any() and not prediction.any() else "one_empty",
                "hd95_mm": None, "assd_mm": None}
    truth_surface = truth & ~binary_erosion(truth)
    predicted_surface = prediction & ~binary_erosion(prediction)
    to_truth = distance_transform_edt(~truth_surface, sampling=spacing_mm)[predicted_surface]
    to_prediction = distance_transform_edt(~predicted_surface, sampling=spacing_mm)[truth_surface]
    distances = np.concatenate((to_truth, to_prediction))
    return {"status": "both_present", "hd95_mm": float(np.percentile(distances, 95)),
            "assd_mm": float(distances.mean())}


def prostate_organ_confusion(truth_xyz: np.ndarray, prediction_xyz: np.ndarray) -> dict[str, int]:
    """Count prostate predictions inside true bladder/rectum and missed prostate."""
    if truth_xyz.shape != prediction_xyz.shape or truth_xyz.ndim != 3:
        raise ValueError("Expected matching XYZ volumes")
    return {"prostate_pred_in_true_bladder": int(((prediction_xyz == 5) & (truth_xyz == 3)).sum()),
            "prostate_pred_in_true_rectum": int(((prediction_xyz == 5) & (truth_xyz == 4)).sum()),
            "true_prostate_pred_other": int(((truth_xyz == 5) & (prediction_xyz != 5)).sum())}

# Original checkpoint selection score, with equal weight for each patient.


import numpy as np


SMALL_ORGAN_IDS = (3, 4, 5)


def patient_small_organ_score(matrices: dict[str, np.ndarray]) -> tuple[float, dict]:
    """Mean available Bladder/Rectum/Prostate Dice over patients equally.

    Each input matrix pools one patient's visits before Dice is calculated.
    Both-absent organ/patient pairs are excluded; missed or extra organs score
    zero through the standard confusion-matrix Dice rule.
    """
    if not matrices:
        raise ValueError("No Validation patient confusion matrices")
    details = {}
    patient_scores = []
    for patient, matrix in sorted(matrices.items()):
        if matrix.shape != (6, 6):
            raise ValueError(f"{patient}: expected six-class confusion matrix")
        dice = overlap(matrix)["dice"]
        small = {name: dice[index] for name, index in
                 (("Bladder", 3), ("Rectum", 4), ("Prostate", 5))}
        details[patient] = small
        available = [value for value in small.values() if value is not None]
        if not available:
            raise ValueError(f"{patient}: no small-organ Validation Dice available")
        patient_scores.append(float(np.mean(available)))
    return float(np.mean(patient_scores)), details

# Common-grid 2D/3D inference without changing the source NIfTI files.


import numpy as np
import torch

from dataset import Volume, from_3d_model_prediction, to_3d_model_arrays


@torch.inference_mode()
def predict_volume(model: torch.nn.Module, volume: Volume, *, mode: str,
                   device: torch.device | str, slice_batch: int = 8) -> np.ndarray:
    """Return class labels in the input volume's XYZ voxel order.

    2D and 3D arms use the same prepared MRI, padding and inverse mapping.
    Peak VRAM and latency are measured outside this function so preprocessing
    and model time can be reported separately.
    """
    if mode not in ("2d", "3d") or slice_batch < 1:
        raise ValueError("mode must be 2d/3d and slice_batch positive")
    image, _, original_shape = to_3d_model_arrays(volume)
    model.eval()
    if mode == "3d":
        tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None].to(device)
        logits = model(tensor)
        if logits.shape != (1, 6, *image.shape):
            raise ValueError("3D model returned an unexpected logit shape")
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    else:
        predictions = []
        for start in range(0, image.shape[0], slice_batch):
            batch = np.ascontiguousarray(image[start:start + slice_batch, None])
            logits = model(torch.from_numpy(batch).to(device))
            if logits.shape != (len(batch), 6, *image.shape[1:]):
                raise ValueError("2D model returned an unexpected logit shape")
            predictions.append(logits.argmax(dim=1).cpu().numpy().astype(np.uint8))
        prediction = np.concatenate(predictions, axis=0)
    return from_3d_model_prediction(prediction, original_shape)


CLASS_NAMES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")


def build_model(name):
    from modules import StandardUNet3D, ImprovedUNet3DV3
    if name not in ("standard", "v3"):
        raise ValueError("Model must be standard or v3")
    return (StandardUNet3D if name == "standard" else ImprovedUNet3DV3)()


def sha256_file(path):
    import hashlib
    from pathlib import Path
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    import json, os
    from pathlib import Path
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path, state):
    import os
    from pathlib import Path
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_model(name, checkpoint, device):
    """Only load trusted local checkpoints: historical files contain Python state."""
    model = build_model(name).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = state.get("model")
    allowed = {"standard": {"standard", "standard_3d", "baseline_3d"},
               "v3": {"v3", "improved_3d_v3"}}
    if identity is not None and identity not in allowed[name]:
        raise ValueError("Checkpoint/model identity mismatch")
    model.load_state_dict(state["model_state"], strict=True)
    return model.eval()


def aggregate_cases(rows):
    """Combine each patient's visit counts, then give every patient equal weight."""
    grouped = {}
    for row in rows:
        patient = row["patient"]
        grouped.setdefault(patient, np.zeros((6, 6), dtype=np.int64))
        grouped[patient] += np.asarray(row["confusion"], dtype=np.int64)
    score, details = patient_small_organ_score(grouped)
    metrics = {p: overlap(m) for p, m in grouped.items()}
    averaged = {}
    for metric in ("dice", "iou", "precision", "recall"):
        averaged[metric] = []
        for k in range(6):
            values = [r[metric][k] for r in metrics.values() if r[metric][k] is not None]
            averaged[metric].append(float(np.mean(values)) if values else None)
    return {"selection_score": score, "patient_small_organ_dice": details,
            "patient_equal": averaged, "patients": metrics,
            "pooled_confusion": sum(grouped.values()).tolist(),
            "foreground_dice": float(np.mean([v for v in averaged["dice"][1:] if v is not None]))}


def spatial_diagnostics(volume, prediction):
    from scipy.ndimage import distance_transform_edt
    truth = volume.mask
    gram = volume.affine[:3, :3].T @ volume.affine[:3, :3]
    diagonal = np.diag(np.diag(gram))
    if not np.allclose(gram, diagonal, rtol=1e-5, atol=1e-4):
        raise ValueError("Sheared grid: spacing-only Euclidean surface distances invalid")
    result = {"surfaces": {CLASS_NAMES[k]: symmetric_surface_distances(
        truth, prediction, volume.spacing_mm, label=k) for k in range(1, 6)},
        "far_background_to_body_voxels": int(((truth == 0) & (prediction == 1) &
            (distance_transform_edt(truth != 1, sampling=volume.spacing_mm) >= 20)).sum()),
        "organ_confusion": prostate_organ_confusion(truth, prediction)}
    try:
        regions = prostate_axial_regions(truth, volume.affine)
        values = slice_dice(truth, prediction)
        result["prostate_regions"] = {k: float(np.mean([values[z] for z in zz]))
                                       for k, zz in regions.items()}
    except ValueError as error:
        result["prostate_regions"] = {"unavailable": str(error)}
    return result


def evaluate(model, cases, device, *, detailed=False):
    """Evaluate a fixed model on complete volumes without changing its weights."""
    from dataset import load_case, downsample, zscore_nonzero
    from time import perf_counter
    rows = []
    model.eval()
    for case in cases:
        volume = zscore_nonzero(downsample(load_case(case)))
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        prediction = predict_volume(model, volume, mode="3d", device=device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        row = {"case": case.identifier, "patient": case.patient,
               "confusion": confusion(volume.mask, prediction).tolist(),
               "inference_seconds": perf_counter() - started,
               "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
                    if str(device).startswith("cuda") else None}
        if detailed:
            row.update(spatial_diagnostics(volume, prediction))
            row.update(mri_sha256=sha256_file(case.mri), mask_sha256=sha256_file(case.mask))
        rows.append(row)
        print(f"{case.split} {len(rows)}/{len(cases)}: {case.identifier}", flush=True)
    return {**aggregate_cases(rows), "cases": rows}
