"""Reusable classification, graph-diagnostic, and resource helpers."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import psutil
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn


def _arrays(
    y_true: np.ndarray | torch.Tensor | Sequence[int],
    probabilities: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert labels/probabilities to NumPy and check their shapes and values."""
    # sklearn works with cpu numpy arrays, while inference may be on a gpu.
    labels = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    probs = probabilities.detach().cpu().numpy() if isinstance(probabilities, torch.Tensor) else np.asarray(probabilities)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if labels.size == 0 or probs.ndim != 2 or probs.shape[0] != labels.size or probs.shape[1] < 2:
        raise ValueError("Provide non-empty labels and probabilities shaped (samples, classes).")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to 1; apply softmax to logits first.")
    if (labels < 0).any() or (labels >= probs.shape[1]).any():
        raise ValueError("labels must be class indices matching probability columns")
    return labels, probs


def precision_recall_data(
    y_true: np.ndarray | torch.Tensor | Sequence[int],
    probabilities: np.ndarray | torch.Tensor,
    class_names: Sequence[str] | None = None,
) -> dict[str, dict[str, np.ndarray | float]]:
    """Return one-vs-rest precision-recall curves and trapezoidal PR-AUC per class."""
    labels, probs = _arrays(y_true, probabilities)
    names = list(class_names) if class_names is not None else [str(i) for i in range(probs.shape[1])]
    if len(names) != probs.shape[1]:
        raise ValueError("class_names must have one name per probability column")
    curves = {}
    for i, name in enumerate(names):
        # compare this class against all other classes to make a binary curve.
        binary_labels = labels == i
        if not binary_labels.any() or binary_labels.all():
            curves[name] = {
                "precision": np.array([]),
                "recall": np.array([]),
                "pr_auc": float("nan"),
            }
            continue
        precision, recall, _ = precision_recall_curve(binary_labels, probs[:, i])
        curves[name] = {
            "precision": precision,
            "recall": recall,
            # recall is returned from high to low, so reverse it for auc().
            "pr_auc": float(auc(recall[::-1], precision[::-1])),
        }
    return curves


def classification_metrics(
    y_true: np.ndarray | torch.Tensor | Sequence[int],
    probabilities: np.ndarray | torch.Tensor,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return accuracy, macro-F1, and per-class precision/recall/F1/AUROC/PR-AUC."""
    labels, probs = _arrays(y_true, probabilities)
    names = list(class_names) if class_names is not None else [str(i) for i in range(probs.shape[1])]
    if len(names) != probs.shape[1]:
        raise ValueError("class_names must have one name per probability column")

    predictions = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        # include every class, even if a model never predicts one of them.
        labels, predictions, labels=np.arange(probs.shape[1]), zero_division=0
    )
    pr_curves = precision_recall_data(labels, probs, names)
    per_class = {}
    auc_values = []
    pr_auc_values = []
    for i, name in enumerate(names):
        # auc needs both positive and negative examples for this class.
        binary_labels = labels == i
        auroc = float(roc_auc_score(binary_labels, probs[:, i])) if binary_labels.any() and (~binary_labels).any() else float("nan")
        pr_auc = float(pr_curves[name]["pr_auc"])
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "auroc_ovr": auroc,
            "pr_auc": pr_auc,
        }
        if np.isfinite(auroc):
            auc_values.append(auroc)
        if np.isfinite(pr_auc):
            pr_auc_values.append(pr_auc)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=np.arange(probs.shape[1]), average="macro", zero_division=0)),
        "macro_auroc_ovr": float(np.mean(auc_values)) if auc_values else float("nan"),
        "macro_pr_auc": float(np.mean(pr_auc_values)) if pr_auc_values else float("nan"),
        "per_class": per_class,
    }


def calibration_bins(
    y_true: np.ndarray | torch.Tensor | Sequence[int],
    probabilities: np.ndarray | torch.Tensor,
    n_bins: int = 15,
) -> dict[str, np.ndarray]:
    """Return top-label confidence, accuracy, and counts for reliability diagrams."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    labels, probs = _arrays(y_true, probabilities)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    # cap confidence 1.0 into the final bin instead of creating an extra bin.
    indices = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(indices, minlength=n_bins)
    bin_conf = np.full(n_bins, np.nan)
    bin_acc = np.full(n_bins, np.nan)
    for i in range(n_bins):
        selected = indices == i
        if selected.any():
            bin_conf[i], bin_acc[i] = confidence[selected].mean(), correct[selected].mean()
    return {
        "confidence": bin_conf,
        "accuracy": bin_acc,
        "count": counts,
        "lower": np.arange(n_bins) / n_bins,
        "upper": np.arange(1, n_bins + 1) / n_bins,
    }


def expected_calibration_error(
    y_true: np.ndarray | torch.Tensor | Sequence[int],
    probabilities: np.ndarray | torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute top-label expected calibration error with equal-width bins."""
    bins = calibration_bins(y_true, probabilities, n_bins)
    used = bins["count"] > 0
    # weight each bin's confidence gap by its share of all examples.
    return float(np.average(np.abs(bins["accuracy"][used] - bins["confidence"][used]), weights=bins["count"][used]))


def confidence_error_auroc(
    y_true: np.ndarray | torch.Tensor | Sequence[int],
    probabilities: np.ndarray | torch.Tensor,
) -> float:
    """AUROC for identifying errors, ranked by 1 - maximum class probability."""
    labels, probs = _arrays(y_true, probabilities)
    errors = probs.argmax(axis=1) != labels
    # higher scores mean more likely to be wrong.
    return float(roc_auc_score(errors, 1.0 - probs.max(axis=1))) if errors.any() and (~errors).any() else float("nan")


def risk_coverage_curve(
    y_true: np.ndarray | torch.Tensor | Sequence[int],
    probabilities: np.ndarray | torch.Tensor,
) -> dict[str, np.ndarray]:
    """Return accuracy/risk as predictions are retained from highest confidence down."""
    labels, probs = _arrays(y_true, probabilities)
    confidence = probs.max(axis=1)
    # keep the most confident predictions first, simulating rejection of uncertain ones.
    correct = (probs.argmax(axis=1) == labels)[np.argsort(-confidence, kind="stable")]
    accuracy = np.cumsum(correct) / np.arange(1, labels.size + 1)
    return {
        "coverage": np.arange(1, labels.size + 1) / labels.size,
        "accuracy": accuracy,
        "risk": 1.0 - accuracy,
        "threshold": np.sort(confidence)[::-1],
    }


def node_degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Count incoming edges per node; for this symmetric graph, this is degree."""
    _validate_edges(edge_index, num_nodes)
    degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    if edge_index.numel():
        # edge_index[1] lists each edge's destination node.
        degree.index_add_(0, edge_index[1], torch.ones_like(edge_index[1]))
    return degree


def neighbour_agreement(edge_index: torch.Tensor, y: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return the fraction of each node's neighbours with the same true label.

    Isolated nodes return NaN. Use only for retrospective analysis because
    neighbour true labels are not available at inference.
    """
    _validate_edges(edge_index, num_nodes)
    if y.ndim != 1 or y.numel() != num_nodes:
        raise ValueError("y must contain one label per node")
    src, dst = edge_index
    labels = y.to(edge_index.device)
    counts = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    matches = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    if edge_index.numel():
        # for each directed edge, compare its source label with its destination label.
        counts.index_add_(0, dst, torch.ones_like(dst))
        matches.index_add_(0, dst, (labels[src] == labels[dst]).float())
    agreement = matches / counts.clamp_min(1)
    # agreement is undefined when a node has no neighbours.
    agreement[counts == 0] = torch.nan
    return agreement


def _validate_edges(edge_index: torch.Tensor, num_nodes: int) -> None:
    if num_nodes < 0 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges) and num_nodes must be nonnegative")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("edge_index must contain integer node indices")
    if edge_index.numel() and ((edge_index < 0).any() or (edge_index >= num_nodes).any()):
        raise ValueError("edge_index contains an out-of-range node index")


def parameter_count(model: nn.Module, trainable_only: bool = True) -> int:
    """Count trainable parameters, or all parameters when requested."""
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)


def _synchronize(device: torch.device) -> None:
    # wait for queued gpu work so timing measures completed inference
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_inference(
    inference: Callable[[], Any],
    device: torch.device | str,
    warmup: int = 5,
    repeats: int = 30,
    sample_interval: float = 0.01,
) -> dict[str, float | int | str]:
    """Measure mean inference latency and peak CUDA memory or process RSS."""
    if warmup < 0 or repeats < 1 or sample_interval <= 0:
        raise ValueError("warmup >= 0, repeats >= 1, and sample_interval > 0 are required")
    target = torch.device(device)
    for _ in range(warmup):
        inference()
    _synchronize(target)

    if target.type == "cuda":
        # cuda exposes its peak allocated device memory directly
        torch.cuda.reset_peak_memory_stats(target)
        memory_kind = "cuda_peak_allocated_mb"
    else:
        # mps has no matching pytorch peak-memory counter, so sample process rss
        process = psutil.Process()
        peak_rss = [process.memory_info().rss]
        stop = threading.Event()

        def sample() -> None:
            # sample while inference runs to capture temporary memory increases
            while not stop.wait(sample_interval):
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        memory_kind = "process_peak_rss_mb"

    times = []
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            inference()
            _synchronize(target)
            times.append(time.perf_counter() - start)
    finally:
        if target.type != "cuda":
            stop.set()
            sampler.join()
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    peak_mb = (
        torch.cuda.max_memory_allocated(target) / 1024**2
        if target.type == "cuda"
        else peak_rss[0] / 1024**2
    )
    return {
        "mean_latency_ms": float(np.mean(times) * 1000),
        "std_latency_ms": float(np.std(times, ddof=1) * 1000) if repeats > 1 else 0.0,
        "peak_memory_mb": float(peak_mb),
        "memory_kind": memory_kind,
        "repeats": repeats,
    }
