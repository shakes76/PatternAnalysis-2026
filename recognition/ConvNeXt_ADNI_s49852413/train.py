"""Training, validation, calibration, resource profiling, and baseline comparison script.

Author: Lingxiao Li (s49852413)
Course: COMP3710 / COMP7710 Pattern Recognition and Analysis (2026)

This script trains both the proposed `[Hard Difficulty]` `ConvNeXt` model and the
`SimpleCNNBaseline` benchmark on identical patient-separated splits of the ADNI
T1-weighted Brain MRI dataset, fits post-hoc Temperature Scaling calibration on
the validation split, profiles computational efficiency (VRAM, parameter count,
inference latency), evaluates comprehensive classification & calibration metrics
on the held-out test set, and saves diagnostic plots to `images/`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from dataset import CLASS_NAMES, build_adni_dataloaders
from modules import TemperatureScaledClassifier, build_model, count_parameters

PROJECT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_DIR / "images"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"


def set_global_seed(seed: int = 42) -> None:
    """Enforce deterministic seeding across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    """Select optimal compute device: CUDA (Rangpur A100), Apple MPS, or CPU."""
    req = requested.lower().strip()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


def get_peak_memory_mb(device: torch.device) -> float:
    """Measure peak allocated GPU/MPS memory in Megabytes (MB)."""
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
    if device.type == "mps" and hasattr(torch, "mps"):
        try:
            return float(torch.mps.driver_allocated_memory() / (1024.0 ** 2))
        except Exception:
            return 0.0
    return 0.0


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Tuple[float, Dict[str, List[float]]]:
    """Compute Expected Calibration Error (ECE) and reliability curve bins.

    Args:
        probs: `[N, 2]` predicted class probability matrix.
        labels: `[N]` ground-truth integer labels in `{0, 1}`.
        n_bins: Number of confidence histogram bins in `[0.5, 1.0]`.

    Returns:
        `(ece_score, reliability_dict)`
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    bin_confs: List[float] = []
    bin_accs: List[float] = []
    bin_counts: List[int] = []

    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        cnt = int(np.sum(in_bin))
        bin_counts.append(cnt)
        if cnt > 0:
            avg_conf = float(np.mean(confidences[in_bin]))
            avg_acc = float(np.mean(accuracies[in_bin]))
            ece += (cnt / n) * abs(avg_acc - avg_conf)
            bin_confs.append(avg_conf)
            bin_accs.append(avg_acc)
        else:
            bin_confs.append(float(0.5 * (lo + hi)))
            bin_accs.append(float("nan"))

    return float(ece), {
        "bin_confidences": bin_confs,
        "bin_accuracies": bin_accs,
        "bin_counts": bin_counts,
    }


def evaluate_selective_triage(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """Evaluate reject-option triage metrics across confidence thresholds tau.

    Slices with `max_c p_hat_c >= tau` are automatically diagnosed by the model;
    ambiguous slices with `max_c p_hat_c < tau` are routed to a human specialist.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]

    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    n_total = len(labels)
    total_ad = max(1, int(np.sum(labels == 1)))

    results: List[Dict[str, float]] = []
    for tau in thresholds:
        accepted = confidences >= tau
        n_acc = int(np.sum(accepted))
        coverage = n_acc / max(1, n_total)
        referral_rate = 1.0 - coverage

        if n_acc > 0:
            acc_accepted = float(np.mean(preds[accepted] == labels[accepted]))
            # False negatives among accepted predictions (missed AD cases that were NOT referred)
            fn_accepted = int(np.sum((labels == 1) & accepted & (preds == 0)))
            ad_accepted = int(np.sum((labels == 1) & accepted))
            fn_rate_accepted = fn_accepted / max(1, ad_accepted) if ad_accepted > 0 else 0.0
            unflagged_missed_ad_fraction = fn_accepted / total_ad
            # System-level accuracy when human specialist resolves referred ambiguous cases
            system_correct = int(np.sum(preds[accepted] == labels[accepted])) + (n_total - n_acc)
            system_acc = system_correct / max(1, n_total)
        else:
            acc_accepted = 1.0
            fn_rate_accepted = 0.0
            unflagged_missed_ad_fraction = 0.0
            system_acc = 1.0

        results.append(
            {
                "threshold": float(tau),
                "coverage": float(coverage),
                "referral_rate": float(referral_rate),
                "accepted_accuracy": float(acc_accepted),
                "accepted_fn_rate": float(fn_rate_accepted),
                "unflagged_missed_ad_fraction": float(unflagged_missed_ad_fraction),
                "system_hybrid_accuracy": float(system_acc),
            }
        )
    return results


@torch.no_grad()
def collect_logits_and_labels(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, object]]]:
    """Run inference over a DataLoader and return stacked logits, labels, and optional metadata."""
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_metas: List[Dict[str, object]] = []

    for batch in loader:
        if len(batch) == 3:
            images, labels, meta_batch = batch
            # Unpack collated metadata dict of lists into list of dicts
            bsz = images.shape[0]
            for i in range(bsz):
                all_metas.append({k: (v[i].item() if isinstance(v[i], torch.Tensor) else v[i]) for k, v in meta_batch.items()})
        else:
            images, labels = batch

        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0), all_metas


def compute_comprehensive_metrics(
    raw_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    metas: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Compute slice-level, patient-level, calibration, and selective triage metrics."""
    y_true = labels.numpy().astype(np.int64)
    raw_probs = F.softmax(raw_logits, dim=-1).numpy()
    cal_probs = F.softmax(raw_logits / temperature, dim=-1).numpy()

    y_pred = np.argmax(cal_probs, axis=1)
    y_score_ad = cal_probs[:, 1]

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec, rec, f1_per_cls, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    auroc = float(roc_auc_score(y_true, y_score_ad))
    pr_auc = float(average_precision_score(y_true, y_score_ad))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    # Calibration & confidence analysis (before vs after Temperature Scaling)
    ece_raw, rel_raw = compute_ece(raw_probs, y_true)
    ece_cal, rel_cal = compute_ece(cal_probs, y_true)
    brier_raw = float(np.mean((raw_probs[:, 1] - y_true) ** 2))
    brier_cal = float(np.mean((cal_probs[:, 1] - y_true) ** 2))

    raw_conf = np.max(raw_probs, axis=1)
    cal_conf = np.max(cal_probs, axis=1)
    correct_mask = y_pred == y_true
    wrong_mask = ~correct_mask

    # Normalized binary Shannon entropy
    eps = 1e-8
    entropy_cal = -(
        cal_probs[:, 0] * np.log(np.clip(cal_probs[:, 0], eps, 1.0))
        + cal_probs[:, 1] * np.log(np.clip(cal_probs[:, 1], eps, 1.0))
    ) / math.log(2.0)

    # Patient-level aggregation (averaging slice probabilities per patient scan)
    patient_metrics: Dict[str, float] = {}
    if metas:
        pid_probs: Dict[str, List[float]] = defaultdict(list)
        pid_labels: Dict[str, int] = {}
        for i, m in enumerate(metas):
            pid = str(m["patient_id"])
            pid_probs[pid].append(float(y_score_ad[i]))
            pid_labels[pid] = int(y_true[i])
        p_true = np.array([pid_labels[p] for p in sorted(pid_probs.keys())])
        p_score = np.array([np.mean(pid_probs[p]) for p in sorted(pid_probs.keys())])
        p_pred = (p_score >= 0.5).astype(np.int64)
        patient_metrics = {
            "patient_accuracy": float(accuracy_score(p_true, p_pred)),
            "patient_macro_f1": float(f1_score(p_true, p_pred, average="macro", zero_division=0)),
            "patient_auroc": float(roc_auc_score(p_true, p_score)),
            "num_patients": int(len(p_true)),
        }

    # ROC and PR curves for plotting
    fpr, tpr, _ = roc_curve(y_true, y_score_ad)
    pr_p, pr_r, _ = precision_recall_curve(y_true, y_score_ad)

    triage_sweep = evaluate_selective_triage(cal_probs, y_true)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "per_class": {
            "NC": {
                "precision": float(prec[0]),
                "recall": float(rec[0]),
                "f1": float(f1_per_cls[0]),
            },
            "AD": {
                "precision": float(prec[1]),
                "recall": float(rec[1]),
                "f1": float(f1_per_cls[1]),
            },
        },
        "confusion_matrix": cm,
        "temperature": float(temperature),
        "ece_uncalibrated": ece_raw,
        "ece_calibrated": ece_cal,
        "brier_uncalibrated": brier_raw,
        "brier_calibrated": brier_cal,
        "mean_conf_correct_raw": float(np.mean(raw_conf[correct_mask])) if np.any(correct_mask) else 0.0,
        "mean_conf_wrong_raw": float(np.mean(raw_conf[wrong_mask])) if np.any(wrong_mask) else 0.0,
        "mean_conf_correct_cal": float(np.mean(cal_conf[correct_mask])) if np.any(correct_mask) else 0.0,
        "mean_conf_wrong_cal": float(np.mean(cal_conf[wrong_mask])) if np.any(wrong_mask) else 0.0,
        "mean_entropy_correct": float(np.mean(entropy_cal[correct_mask])) if np.any(correct_mask) else 0.0,
        "mean_entropy_wrong": float(np.mean(entropy_cal[wrong_mask])) if np.any(wrong_mask) else 0.0,
        "patient_level": patient_metrics,
        "reliability_raw": rel_raw,
        "reliability_calibrated": rel_cal,
        "triage_sweep": triage_sweep,
        "curves": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "pr_precision": [float(x) for x in pr_p],
            "pr_recall": [float(x) for x in pr_r],
        },
        "distributions": {
            "conf_correct_raw": [float(x) for x in raw_conf[correct_mask]],
            "conf_wrong_raw": [float(x) for x in raw_conf[wrong_mask]],
            "conf_correct_cal": [float(x) for x in cal_conf[correct_mask]],
            "conf_wrong_cal": [float(x) for x in cal_conf[wrong_mask]],
        },
    }


def profile_inference_latency(
    model: nn.Module,
    device: torch.device,
    image_size: int = 128,
    batch_size: int = 32,
    num_iters: int = 30,
) -> Dict[str, float]:
    """Profile single-slice latency (ms) and batched throughput (slices/sec)."""
    model.eval()
    x_single = torch.randn(1, 1, image_size, image_size, device=device)
    x_batch = torch.randn(batch_size, 1, image_size, image_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(x_single)
            _ = model(x_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(x_single)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    single_ms = ((time.perf_counter() - t0) / num_iters) * 1000.0

    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(x_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    batch_elapsed = time.perf_counter() - t1
    throughput_fps = (num_iters * batch_size) / max(batch_elapsed, 1e-6)
    batch_per_slice_ms = 1000.0 / max(throughput_fps, 1e-6)

    return {
        "single_slice_latency_ms": float(single_ms),
        "batched_per_slice_latency_ms": float(batch_per_slice_ms),
        "throughput_slices_per_sec": float(throughput_fps),
    }


def train_single_model(
    model_name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 8e-4,
    weight_decay: float = 1e-2,
    image_size: int = 128,
    seed: int = 42,
) -> Dict[str, object]:
    """Train, validate, calibrate, and benchmark a single model architecture."""
    set_global_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = build_model(model_name, in_channels=1, num_classes=2).to(device)
    total_params, trainable_params = count_parameters(model)

    is_convnext = "convnext" in model_name.lower()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay if is_convnext else 1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_auroc": [],
        "epoch_time_sec": [],
    }

    best_score = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 1

    print(f"\n--- Training [{model_name.upper()}] on {device} ({total_params:,} params) ---")
    total_train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            bsz = images.size(0)
            running_loss += float(loss.item()) * bsz
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += bsz

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation pass
        val_logits, val_labels, _ = collect_logits_and_labels(model, val_loader, device)
        val_loss = float(F.cross_entropy(val_logits, val_labels).item())
        val_probs = F.softmax(val_logits, dim=-1).numpy()
        val_preds = np.argmax(val_probs, axis=1)
        val_y = val_labels.numpy()
        val_acc = float(accuracy_score(val_y, val_preds))
        val_f1 = float(f1_score(val_y, val_preds, average="macro", zero_division=0))
        val_auc = float(roc_auc_score(val_y, val_probs[:, 1]))
        ep_sec = time.perf_counter() - ep_t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["val_auroc"].append(val_auc)
        history["epoch_time_sec"].append(ep_sec)

        score = 0.6 * val_f1 + 0.4 * val_auc
        if score >= best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} auc={val_auc:.4f} | "
                f"{ep_sec:.2f}s"
            )

    total_train_sec = time.perf_counter() - total_train_start
    peak_mem_mb = get_peak_memory_mb(device)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Fit post-hoc Temperature Scaling on validation set
    val_logits_best, val_labels_best, _ = collect_logits_and_labels(model, val_loader, device)
    cal_wrapper = TemperatureScaledClassifier(model, init_temperature=1.0).to(device)
    optimal_temp = cal_wrapper.fit_temperature(val_logits_best.to(device), val_labels_best.to(device))

    # Save checkpoint to git-ignored checkpoints/ directory
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINTS_DIR / f"{model_name}_best.pt"
    torch.save(
        {
            "model_name": model_name,
            "best_epoch": best_epoch,
            "temperature": optimal_temp,
            "image_size": image_size,
            "state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    # Final evaluation on held-out test set
    test_logits, test_labels, test_metas = collect_logits_and_labels(model, test_loader, device)
    test_metrics = compute_comprehensive_metrics(
        test_logits, test_labels, temperature=optimal_temp, metas=test_metas
    )
    val_metrics = compute_comprehensive_metrics(
        val_logits_best, val_labels_best, temperature=optimal_temp
    )

    latency_profile = profile_inference_latency(model, device=device, image_size=image_size)
    resource_profile = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "peak_memory_mb": peak_mem_mb,
        "total_train_time_sec": float(total_train_sec),
        "mean_epoch_time_sec": float(np.mean(history["epoch_time_sec"])),
        "best_epoch": best_epoch,
        **latency_profile,
    }

    print(
        f"[{model_name.upper()} TEST RESULTS] "
        f"Slice Acc={test_metrics['accuracy']:.4f} | "
        f"Patient Acc={test_metrics['patient_level'].get('patient_accuracy', 0.0):.4f} | "
        f"Macro-F1={test_metrics['macro_f1']:.4f} | "
        f"AUROC={test_metrics['auroc']:.4f} | "
        f"PR-AUC={test_metrics['pr_auc']:.4f} | "
        f"T*={optimal_temp:.3f} | "
        f"ECE: {test_metrics['ece_uncalibrated']:.4f} -> {test_metrics['ece_calibrated']:.4f}"
    )

    return {
        "model_name": model_name,
        "checkpoint": str(ckpt_path),
        "history": history,
        "resources": resource_profile,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def plot_training_curves(results_by_model: Dict[str, Dict[str, object]], out_path: Path) -> None:
    """Plot training and validation loss, accuracy, and AUROC curves."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=180)
    styles = {
        "convnext": ("#1f77b4", "-", "ConvNeXt [Hard]"),
        "baseline": ("#d62728", "--", "SimpleCNN [Baseline]"),
    }

    for m_name, res in results_by_model.items():
        hist = res["history"]
        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        color, ls, label = styles.get(m_name, ("#2ca02c", "-", m_name))

        # Panel 1: Loss
        axes[0].plot(epochs, hist["train_loss"], color=color, linestyle=":", alpha=0.65, lw=1.6, label=f"{label} Train")
        axes[0].plot(epochs, hist["val_loss"], color=color, linestyle=ls, lw=2.2, label=f"{label} Val")

        # Panel 2: Accuracy
        axes[1].plot(epochs, hist["train_acc"], color=color, linestyle=":", alpha=0.65, lw=1.6, label=f"{label} Train")
        axes[1].plot(epochs, hist["val_acc"], color=color, linestyle=ls, lw=2.2, label=f"{label} Val")

        # Panel 3: Validation AUROC & Macro-F1
        axes[2].plot(epochs, hist["val_auroc"], color=color, linestyle="-", lw=2.2, label=f"{label} Val AUROC")
        axes[2].plot(epochs, hist["val_macro_f1"], color=color, linestyle="--", alpha=0.85, lw=1.8, label=f"{label} Val Macro-F1")

    axes[0].set_title("Cross-Entropy Loss (Train vs. Val)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8.5)

    axes[1].axhline(0.80, color="#2ca02c", linestyle="-.", lw=1.5, label="Course Target (0.80)")
    axes[1].set_title("Classification Accuracy (Strict Patient Split)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.50, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8.5, loc="lower right")

    axes[2].axhline(0.80, color="#2ca02c", linestyle="-.", lw=1.5)
    axes[2].set_title("Validation AUROC & Macro-F1 Progression", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_ylim(0.50, 1.0)
    axes[2].grid(True, linestyle="--", alpha=0.35)
    axes[2].legend(fontsize=8.5, loc="lower right")

    fig.suptitle(
        "ADNI Brain MRI Classification: Training & Validation Dynamics (Zero Patient Leakage)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_and_confusion(results_by_model: Dict[str, Dict[str, object]], out_path: Path) -> None:
    """Plot held-out test ROC curves, Precision-Recall curves, and Confusion Matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=180)
    styles = {
        "convnext": ("#1f77b4", "ConvNeXt [Hard]"),
        "baseline": ("#d62728", "SimpleCNN [Baseline]"),
    }

    for m_name, res in results_by_model.items():
        tm = res["test_metrics"]
        curves = tm["curves"]
        color, label = styles.get(m_name, ("#2ca02c", m_name))

        axes[0].plot(
            curves["fpr"],
            curves["tpr"],
            color=color,
            lw=2.4,
            label=f"{label} (AUROC = {tm['auroc']:.3f})",
        )
        axes[1].plot(
            curves["pr_recall"],
            curves["pr_precision"],
            color=color,
            lw=2.4,
            label=f"{label} (PR-AUC = {tm['pr_auc']:.3f})",
        )

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Chance (0.500)")
    axes[0].set_title("Held-Out Test ROC Curve (AD vs. NC)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].set_title("Held-Out Test Precision-Recall Curve", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Recall (AD Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].set_ylim(0.45, 1.02)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="lower left", fontsize=9)

    # Panel 3: Side-by-side normalized confusion comparison bar/table or ConvNeXt CM
    target_key = "convnext" if "convnext" in results_by_model else list(results_by_model.keys())[0]
    cm = np.array(results_by_model[target_key]["test_metrics"]["confusion_matrix"])
    im = axes[2].imshow(cm, cmap="Blues")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["Predicted NC", "Predicted AD"], fontweight="bold")
    axes[2].set_yticklabels(["Actual NC", "Actual AD"], fontweight="bold")
    axes[2].set_title("ConvNeXt Test Confusion Matrix", fontsize=11, fontweight="bold")

    for r in range(2):
        for c in range(2):
            row_sum = max(1, np.sum(cm[r]))
            pct = 100.0 * cm[r, c] / row_sum
            text_color = "white" if cm[r, c] > np.max(cm) * 0.55 else "black"
            axes[2].text(
                c,
                r,
                f"{cm[r, c]}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Quantitative Discrimination Benchmark on Held-Out ADNI Test Cohort",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_dilemma_calibration_and_triage(
    results_by_model: Dict[str, Dict[str, object]],
    out_path: Path,
    recommended_tau: float = 0.80,
) -> None:
    """Plot Reliability Diagram, Confidence Separation on Errors, and Selective Triage Trade-off."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0), dpi=180)

    # Panel 1: Reliability Diagram
    axes[0].plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.5, label="Perfect Calibration")
    if "baseline" in results_by_model:
        bm = results_by_model["baseline"]["test_metrics"]
        rel_b = bm["reliability_raw"]
        confs_b = np.array(rel_b["bin_confidences"])
        accs_b = np.array(rel_b["bin_accuracies"])
        valid_b = ~np.isnan(accs_b)
        axes[0].plot(
            confs_b[valid_b],
            accs_b[valid_b],
            "o--",
            color="#d62728",
            lw=2.0,
            label=f"SimpleCNN Uncal (ECE={bm['ece_uncalibrated']:.3f})",
        )

    if "convnext" in results_by_model:
        cm = results_by_model["convnext"]["test_metrics"]
        rel_raw = cm["reliability_raw"]
        rel_cal = cm["reliability_calibrated"]
        c_r, a_r = np.array(rel_raw["bin_confidences"]), np.array(rel_raw["bin_accuracies"])
        c_c, a_c = np.array(rel_cal["bin_confidences"]), np.array(rel_cal["bin_accuracies"])
        v_r, v_c = ~np.isnan(a_r), ~np.isnan(a_c)
        axes[0].plot(
            c_r[v_r],
            a_r[v_r],
            "s:",
            color="#ff7f0e",
            lw=1.8,
            label=f"ConvNeXt Uncal (ECE={cm['ece_uncalibrated']:.3f})",
        )
        axes[0].plot(
            c_c[v_c],
            a_c[v_c],
            "^-",
            color="#1f77b4",
            lw=2.4,
            label=f"ConvNeXt + TempScale (ECE={cm['ece_calibrated']:.3f})",
        )

    axes[0].set_title("A. Reliability & Calibration Curve", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Mean Predicted Confidence")
    axes[0].set_ylabel("Empirical Diagnostic Accuracy")
    axes[0].set_xlim(0.5, 1.0)
    axes[0].set_ylim(0.35, 1.02)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8.2, loc="lower right")

    # Panel 2: Confidence Distribution on Correct vs. Incorrect Predictions
    if "convnext" in results_by_model:
        dist = results_by_model["convnext"]["test_metrics"]["distributions"]
        c_corr = np.array(dist["conf_correct_cal"])
        c_err = np.array(dist["conf_wrong_cal"])
        bins = np.linspace(0.5, 1.0, 16)
        axes[1].hist(
            c_corr,
            bins=bins,
            density=True,
            alpha=0.65,
            color="#1f77b4",
            edgecolor="black",
            label=f"Correct Predictions (mean={np.mean(c_corr):.3f})",
        )
        axes[1].hist(
            c_err,
            bins=bins,
            density=True,
            alpha=0.65,
            color="#d62728",
            edgecolor="black",
            label=f"Error Cases / FP+FN (mean={np.mean(c_err):.3f})",
        )
        axes[1].axvline(
            recommended_tau,
            color="#2ca02c",
            linestyle="--",
            lw=2.2,
            label=f"Reject-Option Threshold τ* = {recommended_tau:.2f}",
        )
    axes[1].set_title("B. Confidence Separation (Correct vs. Errors)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Calibrated Max Class Probability")
    axes[1].set_ylabel("Probability Density")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8.2, loc="upper left")

    # Panel 3: Selective Classification / Triage Operating Frontier
    for m_name, color, label in [
        ("convnext", "#1f77b4", "ConvNeXt [Hard]"),
        ("baseline", "#d62728", "SimpleCNN [Baseline]"),
    ]:
        if m_name not in results_by_model:
            continue
        sweep = results_by_model[m_name]["test_metrics"]["triage_sweep"]
        taus = [s["threshold"] for s in sweep]
        acc_auto = [100.0 * s["accepted_accuracy"] for s in sweep]
        covs = [100.0 * s["coverage"] for s in sweep]
        axes[2].plot(taus, acc_auto, "o-", color=color, lw=2.3, label=f"{label} Auto-Cleared Acc (%)")
        axes[2].plot(taus, covs, "s--", color=color, alpha=0.55, lw=1.6, label=f"{label} Automated Coverage (%)")

    axes[2].axvline(recommended_tau, color="#2ca02c", linestyle="--", lw=2.0, label=f"Recommended τ* = {recommended_tau:.2f}")
    axes[2].set_title("C. Selective Triage: Accuracy vs. Coverage Frontier", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Confidence Triage Threshold (τ)")
    axes[2].set_ylabel("Percentage (%)")
    axes[2].set_ylim(25.0, 102.0)
    axes[2].grid(True, linestyle="--", alpha=0.35)
    axes[2].legend(fontsize=7.8, loc="lower left")

    fig.suptitle(
        "Resolving the Engineering Dilemma: Confidence Calibration & Automated Triage Routing",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and benchmark ConvNeXt vs. SimpleCNNBaseline on ADNI Brain MRI."
    )
    parser.add_argument("--data_root", type=str, default=None, help="Path to ADNI/AD_NC directory.")
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["both", "convnext", "baseline", "convnext_tiny"],
        help="Model architecture to train (`both` runs ConvNeXt and SimpleCNNBaseline).",
    )
    parser.add_argument("--image_size", type=int, default=96, help="Input spatial resolution.")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1.2e-3, help="Base learning rate for AdamW.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--val_fraction", type=float, default=0.15, help="Validation patient fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto/cuda/mps/cpu).")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional slice cap for quick test.")
    parser.add_argument(
        "--split_mode",
        type=str,
        default="longitudinal",
        choices=["longitudinal", "subject", "official"],
        help="Evaluation protocol (see dataset.build_adni_dataloaders). Non-default modes write to "
        "images/<mode>/ and checkpoints/<mode>/.",
    )
    return parser.parse_args()


def main() -> None:
    global IMAGES_DIR, CHECKPOINTS_DIR
    args = parse_args()
    set_global_seed(args.seed)
    device = select_device(args.device)
    if args.split_mode != "longitudinal":
        IMAGES_DIR = IMAGES_DIR / args.split_mode
        CHECKPOINTS_DIR = CHECKPOINTS_DIR / args.split_mode

    train_loader, val_loader, test_loader, audit_summary = build_adni_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_samples_per_split=args.max_samples,
        return_meta_eval=True,
        split_mode=args.split_mode,
    )
    audit_summary["split_mode"] = args.split_mode

    print("=== Patient-Level Data Hygiene Audit ===")
    print(json.dumps(audit_summary, indent=2))

    models_to_run = ["baseline", "convnext"] if args.model == "both" else [args.model]
    results_by_model: Dict[str, Dict[str, object]] = {}

    for m_name in models_to_run:
        res = train_single_model(
            model_name=m_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            image_size=args.image_size,
            seed=args.seed,
        )
        results_by_model[m_name] = res

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_curves(results_by_model, IMAGES_DIR / "training_curves.png")
    plot_roc_pr_and_confusion(results_by_model, IMAGES_DIR / "roc_pr_comparison.png")
    plot_dilemma_calibration_and_triage(results_by_model, IMAGES_DIR / "calibration_and_uncertainty.png")

    summary_payload = {
        "device": str(device),
        "audit": audit_summary,
        "models": {
            k: {
                "resources": v["resources"],
                "val_metrics": {
                    mk: mv
                    for mk, mv in v["val_metrics"].items()
                    if mk not in ("curves", "distributions")
                },
                "test_metrics": {
                    mk: mv
                    for mk, mv in v["test_metrics"].items()
                    if mk not in ("curves", "distributions")
                },
            }
            for k, v in results_by_model.items()
        },
    }

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINTS_DIR / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\nSaved plots to {IMAGES_DIR} and metrics to {CHECKPOINTS_DIR / 'benchmark_summary.json'}.")


if __name__ == "__main__":
    main()
