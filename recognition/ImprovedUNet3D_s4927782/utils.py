"""Whole-volume hard-label metrics, patient-balanced aggregation and plots.

Class zero is background. Organ names are deliberately not inferred. Both-empty
class masks yield None and are excluded from means; a false positive for an
absent class yields zero. No smoothing is applied to reported Dice or IoU.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def mean_defined(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _integer_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Metrics require integer class IDs.")
    return array


def compute_case_metrics(pred, target, num_classes):
    """Compute hard-label per-class metrics over one entire volume/case."""
    pred, target = _integer_array(pred), _integer_array(target)
    if pred.shape != target.shape or not target.size:
        raise ValueError("Prediction and target must have the same nonempty shape.")
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer >= 2.")
    if min(int(pred.min()), int(target.min())) < 0 or max(int(pred.max()), int(target.max())) >= num_classes:
        raise ValueError("Class IDs must be in [0, num_classes-1].")
    per_class = []
    for index in range(num_classes):
        predicted, actual = pred == index, target == index
        predicted_count, target_count = int(predicted.sum()), int(actual.sum())
        intersection = int(np.logical_and(predicted, actual).sum())
        union = predicted_count + target_count - intersection
        denominator = predicted_count + target_count
        per_class.append({
            "class_index": index,
            "dice": 2 * intersection / denominator if denominator else None,
            "iou": intersection / union if union else None,
            "target_voxels": target_count,
            "predicted_voxels": predicted_count,
            "intersection": intersection,
            "union": union,
        })
    return {
        "per_class": per_class,
        "foreground_mean_dice": mean_defined(item["dice"] for item in per_class[1:]),
        "foreground_mean_iou": mean_defined(item["iou"] for item in per_class[1:]),
    }


def aggregate_by_patient(records):
    """Mean cases within each patient, then mean patients with equal weight.

    Each case's foreground mean excludes its both-empty classes. The headline
    score averages these case foreground means within patient and then across
    patients. Per-class means use the same patient weighting independently.
    Raw voxel counts remain in the input case records, never pooled for scoring.
    """
    grouped = defaultdict(list)
    for record in records:
        if not isinstance(record.get("patient"), str) or not record["patient"]:
            raise ValueError("Every case must have a nonempty patient string.")
        grouped[record["patient"]].append(record)
    if not records:
        return {"num_cases": 0, "num_patients": 0, "patients": [], "per_class": [],
                "foreground_mean_dice": None, "foreground_mean_iou": None}
    class_indices = [item["class_index"] for item in records[0]["per_class"]]
    for record in records:
        if [item["class_index"] for item in record["per_class"]] != class_indices:
            raise ValueError("All cases must use identical class indices.")
    patients = []
    for patient, cases in sorted(grouped.items()):
        per_class = [{"class_index": index, **{
            metric: mean_defined(case["per_class"][position][metric] for case in cases)
            for metric in ("dice", "iou")
        }} for position, index in enumerate(class_indices)]
        patients.append({
            "patient": patient, "num_cases": len(cases), "per_class": per_class,
            **{f"foreground_mean_{metric}": mean_defined(case[f"foreground_mean_{metric}"] for case in cases)
               for metric in ("dice", "iou")},
        })
    return {
        "num_cases": len(records), "num_patients": len(patients), "patients": patients,
        "per_class": [{"class_index": index, **{
            metric: mean_defined(patient["per_class"][position][metric] for patient in patients)
            for metric in ("dice", "iou")
        }} for position, index in enumerate(class_indices)],
        **{f"foreground_mean_{metric}": mean_defined(patient[f"foreground_mean_{metric}"] for patient in patients)
           for metric in ("dice", "iou")},
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_case_name(patient, visit):
    raw = f"{patient}__{visit}"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip(".")[:100] or "case"
    return f"{stem}_{hashlib.sha256(raw.encode()).hexdigest()[:8]}"


def synchronize(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_volume(model, image, model_name, device, slice_batch_size=8, amp=False):
    """Predict 1xDxHxW preprocessed image; return CPU DxHxW class IDs.

    Timing includes preprocessed input transfer, forward, argmax and output
    transfer. Loading/preprocessing/metrics/plotting are excluded. Caller warms
    the model first. Both 2D and 3D evaluate the same complete volume grid.
    """
    import torch
    if image.ndim != 4 or image.shape[0] != 1 or slice_batch_size < 1:
        raise ValueError("Expected image shape (1,D,H,W) and a positive slice batch size.")
    if model_name not in ("unet2d", "improved3d"):
        raise ValueError("Unknown model name.")
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        if model_name == "improved3d":
            prediction = model(image.unsqueeze(0).to(device))["logits"].argmax(dim=1)[0].cpu()
        else:
            slices = image[0].unsqueeze(1)
            parts = []
            for chunk in slices.split(slice_batch_size):
                parts.append(model(chunk.to(device))["logits"].argmax(dim=1).cpu())
            prediction = torch.cat(parts, dim=0)
    synchronize(device)
    milliseconds = (time.perf_counter() - start) * 1000
    memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return prediction, milliseconds, memory


def evaluate_volumes(model, dataset, model_name, num_classes, device,
                     slice_batch_size=8, amp=False, on_case=None, max_cases=None):
    """Evaluate whole cases. No test split is selected implicitly."""
    was_training = model.training
    model.eval()
    records = []
    try:
        for index in range(len(dataset)):
            if max_cases is not None and index >= max_cases:
                break
            item = dataset[index]
            if index == 0:
                # Exclude one full-volume warmup from reported latency.
                predict_volume(model, item["image"], model_name, device, slice_batch_size, amp)
            prediction, milliseconds, memory = predict_volume(
                model, item["image"], model_name, device, slice_batch_size, amp)
            record = {"patient": str(item["patient"]), "visit": str(item["visit"]),
                      **compute_case_metrics(prediction, item["mask"], num_classes),
                      "inference_ms": milliseconds,
                      "peak_allocated_cuda_bytes": memory,
                      "shape_dhw": list(prediction.shape),
                      "spacing_dhw": [float(value) for value in item["spacing"]]}
            records.append(record)
            if on_case is not None:
                on_case(item, prediction, record)
    finally:
        model.train(was_training)
    return {"cases": records, "aggregate": aggregate_by_patient(records),
            "mean_inference_ms": mean_defined(record["inference_ms"] for record in records),
            "peak_allocated_cuda_bytes": max((record["peak_allocated_cuda_bytes"] or 0 for record in records), default=0) or None,
            "latency_protocol": "One full-case warmup; synchronized transfer+forward+argmax+output-transfer, excluding loading/preprocessing/metrics/plots."}


def plot_training(history, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o")
    axes[0].set(xlabel="Epoch", ylabel="CE + soft Dice loss", title="Training objective")
    axes[1].plot(epochs, [row.get("validation_foreground_dice") for row in history], marker="o")
    axes[1].set(xlabel="Epoch", ylabel="Patient-balanced foreground Dice", ylim=(0, 1), title="Whole-volume validation")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def plot_case(image, target, prediction, destination, num_classes, target_class=1, title=""):
    """Show target-extent endpoints and a maximum-error slice, including outer FP.

    Class labels remain index-based until their anatomical meanings are verified.
    All panels share the preprocessed image grid; this is not native geometry.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    if not 1 <= target_class < num_classes:
        raise ValueError("target_class must identify a foreground class.")
    image = image.detach().cpu().numpy()[0] if hasattr(image, "detach") else np.asarray(image)[0]
    target, prediction = _integer_array(target), _integer_array(prediction)
    bearing = np.flatnonzero((target == target_class).reshape(target.shape[0], -1).any(axis=1))
    error_counts = np.logical_xor(target == target_class, prediction == target_class).reshape(target.shape[0], -1).sum(axis=1)
    chosen = {int(error_counts.argmax())}
    if len(bearing):
        chosen.update((int(bearing[0]), int(bearing[-1])))
    for depth in np.linspace(0, target.shape[0] - 1, min(3, target.shape[0])).round().astype(int):
        if len(chosen) >= min(3, target.shape[0]):
            break
        chosen.add(int(depth))
    indices = sorted(chosen)
    figure, axes = plt.subplots(len(indices), 4, figsize=(12, 3 * len(indices)), squeeze=False)
    class_cmap = plt.get_cmap("tab20", num_classes)
    class_norm = BoundaryNorm(np.arange(-0.5, num_classes + 0.5), num_classes)
    errors_cmap = ListedColormap(["black", "red", "cyan"])
    for row, depth in enumerate(indices):
        actual, predicted = target[depth], prediction[depth]
        errors = np.zeros_like(actual)
        errors[(predicted == target_class) & (actual != target_class)] = 1
        errors[(predicted != target_class) & (actual == target_class)] = 2
        axes[row, 0].imshow(image[depth], cmap="gray")
        gt_plot = axes[row, 1].imshow(actual, cmap=class_cmap, norm=class_norm, interpolation="nearest")
        axes[row, 2].imshow(predicted, cmap=class_cmap, norm=class_norm, interpolation="nearest")
        axes[row, 3].imshow(errors, cmap=errors_cmap, vmin=0, vmax=2, interpolation="nearest")
        for axis, label in zip(axes[row], [f"Input slice {depth}", "Ground truth (class IDs)", "Prediction (class IDs)", f"Class {target_class}: red FP / cyan FN"]):
            axis.set_title(label, fontsize=9)
            axis.axis("off")
    figure.colorbar(gt_plot, ax=axes[:, 1:3].ravel().tolist(), ticks=range(num_classes), fraction=0.025, pad=0.02, label="Class index")
    figure.suptitle(title + " — preprocessed grid", fontsize=10)
    figure.subplots_adjust(top=0.9, bottom=0.03, left=0.02, right=0.87, wspace=0.06, hspace=0.22)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
