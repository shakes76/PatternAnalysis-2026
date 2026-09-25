"""Shared inference, reproducibility, and artifact helpers for the baseline.

Seeding and checkpoint loading follow the PyTorch documentation:
https://docs.pytorch.org/docs/2.6/notes/randomness.html
https://docs.pytorch.org/docs/2.6/notes/serialization.html
"""

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

from dataset import ADNISliceDataset
from metrics import aggregate_scans, binary_metrics


def seed_everything(seed):
    """Request deterministic execution within one software/hardware environment."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def select_device(requested):
    """Use CUDA when available; explicitly requested unavailable CUDA is an error."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in this Python environment.")
    return torch.device(requested)


def seed_worker(worker_id):
    """Seed Python in each loader process, including future stochastic transforms."""
    random.seed(torch.initial_seed() % 2**32)


def make_loader(rows, data_root, image_size, batch_size, workers, seed, shuffle, device):
    """Keep all samples; only the training loader shuffles its own manifest."""
    return DataLoader(
        ADNISliceDataset(rows, data_root, image_size=image_size),
        batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=workers,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def sync_device(device):
    """Synchronize CUDA to avoid reporting asynchronous dispatch time as latency."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(model, loader, device, expected_slices):
    """Aggregate every slice, then compute metrics once per complete scan.

    Forward timing excludes loading and host/device transfer. It is an average
    per slice at the configured batch size, not single-request latency.
    """
    model.eval()
    predictions = []
    forward_seconds = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        sync_device(device)
        started = time.perf_counter()
        logits = model(images)
        sync_device(device)
        forward_seconds += time.perf_counter() - started
        if not torch.isfinite(logits).all().item():
            raise ValueError("Model produced non-finite logits; evaluation stopped.")
        probabilities = torch.sigmoid(logits).cpu().tolist()
        for index, probability in enumerate(probabilities):
            predictions.append({
                "patient_id": batch["patient_id"][index],
                "image_id": batch["image_id"][index],
                "slice_index": int(batch["slice_index"][index]),
                "relative_path": batch["relative_path"][index],
                "label": int(batch["label"][index]),
                "probability": probability,
            })
    scans = aggregate_scans(predictions, expected_slices=expected_slices)
    scores = {
        level: binary_metrics([r["label"] for r in rows], [r["probability"] for r in rows])
        for level, rows in (("scan", scans), ("slice", predictions))
    }
    scores["n_patients"] = len({row["patient_id"] for row in scans})
    scores["forward_seconds"] = forward_seconds
    scores["mean_forward_ms_per_slice"] = forward_seconds * 1000 / len(predictions)
    return scores, predictions, scans


def write_json(path, value):
    """Reject NaN/Infinity instead of writing nonstandard JSON."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    """Export auditable predictions or history with explicit column names."""
    if not rows:
        raise ValueError("Cannot export an empty result table.")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment_info(device):
    """Record actual package/device versions instead of claiming cross-platform equivalence."""
    return {
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "Pillow", "matplotlib")},
        "device": str(device), "cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def code_fingerprints():
    """Identify the exact source files used, even outside a Git checkout."""
    root = Path(__file__).resolve().parent
    names = ("adni_splits.py", "dataset.py", "modules.py", "metrics.py", "training_utils.py", "train.py", "predict.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def validate_output(output, data_root, splits_dir):
    """Protect source data and frozen manifests from generated experiment artifacts."""
    output = Path(output).resolve()
    for protected in (Path(data_root).resolve(), Path(splits_dir).resolve()):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Run output must be separate from source data and split directories.")
    if output.exists():
        raise ValueError(f"Output already exists; refusing to overwrite an experiment: {output}")
    return output


def plot_history(output, history):
    """Write a standalone plot containing training and early-stop data only."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(output) / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(epochs, [r["train_slice_loss"] for r in history], marker="o", label="Train slice BCE (class-weighted)")
    axes[0].plot(epochs, [r["early_stop_scan_loss"] for r in history], marker="o", label="Early-stop scan log loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training and checkpoint selection")
    axes[1].plot(epochs, [r["early_stop_scan_accuracy"] for r in history], marker="o", label="Scan accuracy")
    axes[1].plot(epochs, [r["early_stop_scan_macro_f1"] for r in history], marker="o", label="Scan macro F1")
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1), title="Early-stop patients only")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(Path(output) / "learning_curves.png", dpi=160)
    plt.close(figure)
