"""Shared baseline training/evaluation helpers; no test-image access here.

Probability aggregation is slice -> scan mean -> patient mean of scan means.
This gives equal weight to scans within a patient, not to all repeated slices.
Evaluation metrics use scikit-learn, outside the PyTorch-only model module.
Development assistance: ChatGPT; review, test and disclose in the final README.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)
import torch
from torch import nn

import dataset as data_module
from dataset import (ADNISliceDataset, DataError, SliceRecord, SplitBundle,
                     load_finalized_splits, preprocessing_spec, require)
from modules import BaselineCNN, parameter_count

SOURCE_FILES = ("dataset.py", "modules.py", "training_utils.py", "check_baseline.py", "train.py")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataError("A required JSON report could not be read.") from exc
    require(isinstance(value, dict), "Expected a JSON object.")
    return value


def write_json(path: Path, value: Any) -> None:
    """Atomic replacement prevents an interrupted write becoming a valid report."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    return {name: sha256_file(base / name) for name in SOURCE_FILES}


def new_private_output(path: str | Path, data_root: str | Path,
                       splits_dir: str | Path) -> Path:
    path = Path(path).resolve()
    require(not path.exists(), "Output directory already exists; preserve old runs and choose a new name.")
    for protected in (Path(data_root).resolve(), Path(splits_dir).resolve(),
                      Path(__file__).resolve().parent):
        require(path != protected and protected not in path.parents,
                "Output must be outside dataset, manifests, and project-code directories.")
    require(not any((parent / ".git").exists() for parent in (path, *path.parents)),
            "Checkpoints and patient-linked results must not be written inside a Git repository.")
    path.mkdir(parents=True)
    return path


def checked_bundle(data_root: str | Path, splits_dir: str | Path,
                   pipeline_report: str | Path) -> SplitBundle:
    bundle = load_finalized_splits(data_root, splits_dir)
    report = read_json(pipeline_report)
    require(report.get("data_pipeline_check_passed") is True,
            "A successful Step 6 loader report is required.")
    require(report.get("baseline_training_can_begin_after_visual_review") is True,
            "Step 6 did not pass the checks required before training.")
    require(report.get("manifest_sha256") == bundle.manifest_hashes,
            "The Step 6 report and current finalized manifests differ.")
    current_hash = sha256_file(Path(data_module.__file__).resolve())
    require(report.get("source_code_sha256", {}).get("dataset.py") == current_hash,
            "dataset.py changed since the passed check. Do not bypass this; review/rerun Step 6.")
    require(report.get("preprocessing") == preprocessing_spec(0.0),
            "Preprocessing differs from the checked initial configuration.")
    return bundle


def seed_everything(seed: int) -> None:
    # Must precede CUDA BLAS work for deterministic algorithms on supported setups.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def choose_device(name: str) -> torch.device:
    require(name in ("cuda", "cpu"), "Device must be cuda or cpu.")
    if name == "cuda":
        require(torch.cuda.is_available(), "CUDA unavailable; do not silently fall back to CPU training.")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def hardware_info(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {"torch_version": str(torch.__version__), "device": device.type,
                              "precision": "float32; mixed precision and TF32 disabled"}
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        result.update(device_name=prop.name, total_device_memory_gib=prop.total_memory / 2**30)
    return result


def memory_info(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"peak_allocated_gib": None, "peak_reserved_gib": None,
                "note": "CPU run; GPU memory not measured."}
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "note": "PyTorch process allocator peaks, not total VRAM used by Windows/other processes."}


def reset_memory_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy, expressed as -log_softmax(true_class).

    This is mathematically the usual two-class cross-entropy. Explicit gather
    avoids relying on a CUDA NLLLoss reduction that can be unavailable under
    strict deterministic algorithms. It is tested against PyTorch CE on CPU.
    """
    if logits.ndim != 2 or logits.shape[1] != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("Expected [B,2] logits and [B] class labels.")
    return -torch.log_softmax(logits, dim=1).gather(1, labels[:, None]).mean()


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    require(y.ndim == 1 and p.shape == y.shape and len(y) > 0, "Invalid metric input dimensions.")
    require(np.isin(y, (0, 1)).all() and np.isfinite(p).all()
            and ((p >= 0) & (p <= 1)).all(), "Labels/probabilities are invalid.")
    predicted = (p >= 0.5).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=[0, 1], zero_division=0)
    clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    nll = -np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
    return {"n": int(len(y)), "threshold": 0.5,
            "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "log_loss": float(nll),
            "confusion_matrix_rows_true_columns_predicted_NC_AD": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
            "per_class": {
                name: {"precision": float(precision[i]), "recall": float(recall[i]),
                       "f1": float(f1[i]), "support": int(support[i])}
                for i, name in enumerate(("NC", "AD"))}}


def aggregate_predictions(records: Sequence[SliceRecord], probabilities: Sequence[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    require(len(records) == len(probabilities) and bool(records), "Prediction and record counts disagree.")
    require(all(record.split in ("train", "validation") for record in records),
            "Step 7 does not evaluate the final test partition.")
    groups: dict[str, list[tuple[SliceRecord, float]]] = defaultdict(list)
    for record, probability in zip(records, probabilities):
        groups[record.scan_id].append((record, float(probability)))
    scans: list[dict[str, Any]] = []
    for scan_id in sorted(groups):
        group = groups[scan_id]
        require(len({(r.subject_id, r.label, r.split) for r, _ in group}) == 1,
                "A scan has inconsistent subject/label/split assignments.")
        first = group[0][0]
        scans.append({"scan_id": scan_id, "subject_id": first.subject_id,
                      "label": first.label, "p_ad": math.fsum(p for _, p in group) / len(group),
                      "slices": len(group)})
    subjects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scans:
        subjects[row["subject_id"]].append(row)
    patients: list[dict[str, Any]] = []
    for subject in sorted(subjects):
        group = subjects[subject]
        require(len({r["label"] for r in group}) == 1, "A patient has inconsistent diagnoses.")
        patients.append({"subject_id": subject, "label": group[0]["label"],
                         "p_ad": math.fsum(r["p_ad"] for r in group) / len(group),
                         "scans": len(group)})
    metrics = {"slice": binary_metrics([r.label for r in records], probabilities),
               "scan": binary_metrics([r["label"] for r in scans], [r["p_ad"] for r in scans]),
               "patient": binary_metrics([r["label"] for r in patients], [r["p_ad"] for r in patients])}
    private = {"slices": [{"image_path": r.image_path, "subject_id": r.subject_id,
                           "scan_id": r.scan_id, "label": r.label, "p_ad": float(p)}
                          for r, p in zip(records, probabilities)],
               "scans": scans, "patients": patients}
    return metrics, private


def predict_validation(model: nn.Module, loader: Any, device: torch.device) -> tuple[list[float], float]:
    require(getattr(loader.dataset, "split", None) == "validation",
            "This evaluator accepts the validation dataset only.")
    model.eval()
    probabilities: list[float] = []
    paths: list[str] = []
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            require(set(batch["split"]) == {"validation"}, "Unexpected evaluation partition.")
            logits = model(batch["image"].to(device, non_blocking=device.type == "cuda"))
            require(bool(torch.isfinite(logits).all()), "Non-finite validation logits.")
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            paths.extend(batch["image_path"])
    synchronize(device)
    seconds = time.perf_counter() - started
    require(paths == [r.image_path for r in loader.dataset.records],
            "Validation loader is shuffled, incomplete, or repeats images.")
    return probabilities, seconds


def selection_key(metrics: dict[str, Any]) -> tuple[float, float]:
    """Higher patient macro-F1; on an exact tie lower patient log loss; then earliest."""
    patient = metrics["patient"]
    return (patient["macro_f1"], -patient["log_loss"])


def save_checkpoint(path: Path, model: nn.Module, *, epoch: int,
                    manifest_hashes: dict[str, str], config: dict[str, Any],
                    optimizer: torch.optim.Optimizer | None = None) -> None:
    value: dict[str, Any] = {
        "format_version": 1, "architecture": "BaselineCNN-v1",
        "epoch": epoch, "model_label_map": {"NC": 0, "AD": 1},
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "manifest_sha256": manifest_hashes, "config": config,
        "preprocessing": preprocessing_spec(0.0)}
    if optimizer is not None:
        value["optimizer_state"] = optimizer.state_dict()
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, device: torch.device,
                    manifest_hashes: dict[str, str]) -> tuple[BaselineCNN, dict[str, Any]]:
    # Load only self-produced, trusted checkpoints. Never use arbitrary .pt files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(checkpoint.get("architecture") == "BaselineCNN-v1"
            and checkpoint.get("model_label_map") == {"NC": 0, "AD": 1}
            and checkpoint.get("manifest_sha256") == manifest_hashes,
            "Checkpoint architecture, labels, or split hashes do not match.")
    model = BaselineCNN()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device).eval()
    return model, checkpoint


def inference_benchmark(model: nn.Module, sample: torch.Tensor,
                        device: torch.device, repeats: int = 50) -> dict[str, Any]:
    """Model-only batch-size-1 timings; exclude JPEG loading and preprocessing."""
    require(sample.shape == (1, 1, 224, 224) and repeats >= 1, "Invalid latency benchmark input.")
    sample = sample.to(device)
    model.eval()
    timings: list[float] = []
    with torch.inference_mode():
        for _ in range(10):
            model(sample)
        synchronize(device)
        for _ in range(repeats):
            synchronize(device)
            started = time.perf_counter()
            model(sample)
            synchronize(device)
            timings.append(1000.0 * (time.perf_counter() - started))
    return {"batch_size": 1, "repeats": repeats, "warmup": 10,
            "median_ms_per_slice": statistics.median(timings),
            "p95_ms_per_slice": sorted(timings)[min(len(timings) - 1, math.ceil(.95 * len(timings)) - 1)],
            "scope": "Forward pass only, tensor already on device; not patient/end-to-end latency."}


def select_tiny_training_indices(dataset: ADNISliceDataset, per_class: int = 4,
                                 seed: int = 42) -> list[int]:
    require(dataset.split == "train", "Tiny fitting check must use training patients only.")
    patients: dict[int, dict[str, list[int]]] = {0: defaultdict(list), 1: defaultdict(list)}
    for index, row in enumerate(dataset.records):
        patients[row.label][row.subject_id].append(index)
    selected: list[int] = []
    for label in (0, 1):
        ordered = sorted(patients[label], key=lambda s: hashlib.sha256(f"{seed}|{s}".encode()).hexdigest())
        require(bool(ordered), "Tiny check needs both training classes.")
        for subject in ordered[:per_class]:
            indices = patients[label][subject]
            scan = min(dataset.records[i].scan_id for i in indices)
            scan_indices = sorted((i for i in indices if dataset.records[i].scan_id == scan),
                                  key=lambda i: dataset.records[i].slice_index)
            selected.append(scan_indices[len(scan_indices) // 2])
    return selected
