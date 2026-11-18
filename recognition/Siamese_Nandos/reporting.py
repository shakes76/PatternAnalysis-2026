"""Shared experiment reports; all operating rules are fitted on validation only."""
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, auc, brier_score_loss,
                             precision_recall_curve, precision_recall_fscore_support)


def extra_metrics(labels, scores, threshold):
    """Report both classes, distinguishing trapezoidal PR-AUC from average precision."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, scores >= threshold, labels=[0, 1], zero_division=0)
    pr, re, _ = precision_recall_curve(labels, scores)
    result = {"accuracy": float(accuracy_score(labels, scores >= threshold)),
              "macro_f1": float(f1.mean()), "brier_score": float(brier_score_loss(labels, scores)),
              "pr_auc": float(auc(re, pr))}
    for i, name in enumerate(("benign", "melanoma")):
        result.update({f"{name}_precision": float(precision[i]),
                       f"{name}_recall": float(recall[i]), f"{name}_f1": float(f1[i])})
    return result


def confidence_report(labels, scores, threshold, output, plots=True):
    """Equal-width calibration and score assigned to the actual thresholded class."""
    correct = (scores >= threshold) == labels
    confidence = np.where(scores >= threshold, scores, 1 - scores)
    rows = []
    bins = np.minimum((scores * 10).astype(int), 9)
    for i in range(10):
        mask = bins == i
        rows.append({"lower": i / 10, "upper": (i + 1) / 10, "count": int(mask.sum()),
                     "mean_score": float(scores[mask].mean()) if mask.any() else np.nan,
                     "melanoma_fraction": float(labels[mask].mean()) if mask.any() else np.nan})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "calibration.csv", index=False)
    pd.DataFrame([{"correct": value, "count": int((correct == value).sum()),
                   "mean_decision_confidence": float(confidence[correct == value].mean())
                   if (correct == value).any() else np.nan}
                  for value in (False, True)]).to_csv(output / "confidence_summary.csv", index=False)
    if plots:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.plot(frame.mean_score, frame.melanoma_fraction, "o-")
        ax.set(xlabel="Mean melanoma score", ylabel="Observed melanoma fraction",
               xlim=(0, 1), ylim=(0, 1), title="Test reliability (uncalibrated scores)")
        fig.savefig(output / "reliability.png", bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots()
        for value, name in ((False, "Incorrect"), (True, "Correct")):
            ax.hist(confidence[correct == value], bins=np.linspace(0, 1, 21), alpha=.5, label=name)
        ax.set(xlabel="Score assigned to predicted class", ylabel="Images")
        ax.legend()
        fig.savefig(output / "confidence_correctness.png", bbox_inches="tight")
        plt.close(fig)
    return float((frame["count"] / len(labels) *
                  (frame.mean_score - frame.melanoma_fraction).abs()).sum())


def choose_referral_band(scores, threshold, fraction):
    """Select a validation distance quantile; ties can exceed the requested workload."""
    if not 0 <= fraction <= 1:
        raise ValueError("referral-fraction must be between 0 and 1")
    if fraction == 0:
        return threshold, threshold
    if not np.isfinite(threshold):
        raise ValueError("Referral requires a finite cutoff; provide --threshold or another policy")
    width = float(np.quantile(np.abs(scores - threshold), fraction))
    return threshold - width, np.nextafter(threshold + width, np.inf)


def referral_report(labels, scores, threshold, low, high):
    referred = (scores >= low) & (scores < high)
    accepted = ~referred
    predicted = scores >= threshold
    return referred, {"referral_lower": low, "referral_upper": high,
        "referral_rate": float(referred.mean()), "automatic_coverage": float(accepted.mean()),
        "automatic_images": int(accepted.sum()),
        "automatic_accuracy": float((predicted[accepted] == labels[accepted]).mean()) if accepted.any() else np.nan,
        "unreferred_melanoma_misses": int(((labels == 1) & ~predicted & accepted).sum()),
        "referred_melanomas": int(((labels == 1) & referred).sum())}


def profile_model(model, model_name, image_size, device, iterations=20):
    """Batch-one forward latency excluding transfers/decoding; allocated GPU memory."""
    x = torch.zeros(1, 3, image_size, image_size, device=device)
    forward = model if model_name == "baseline" else model.classify
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    timings = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                              enabled=device.type == "cuda"):
        for _ in range(5):
            forward(x)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(iterations):
            sync()
            started = time.perf_counter()
            forward(x)
            sync()
            timings.append((time.perf_counter() - started) * 1000)
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "inference_batch_size": 1, "image_size": image_size, "latency_iterations": iterations,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "precision": "float16_autocast" if device.type == "cuda" else "float32",
            "path": "baseline" if model_name == "baseline" else "auxiliary_classifier",
            "latency_mean_ms": float(np.mean(timings)), "latency_median_ms": float(np.median(timings)),
            "peak_inference_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda" else None}


def write_manifest(output, args, hashes, filename="run_manifest.json"):
    """Record source identity, dependencies, arguments and frozen split hashes."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             cwd=Path(__file__).parent, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "matplotlib", "Pillow")}
    payload = {"config": vars(args), "split_sha256": hashes, "git_commit": commit, "git_dirty": dirty,
               "python": platform.python_version(), "packages": packages}
    (Path(output) / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_history(history, output):
    """Plot losses separately from ranking metrics; Siamese objectives are not identical."""
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for column in frame:
        if "loss" in column:
            axes[0].plot(frame.epoch, frame[column], label=column)
        elif column in ("val_ap", "val_auroc"):
            axes[1].plot(frame.epoch, frame[column], label=column)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
    axes[0].set_ylabel("Loss (objectives differ; see legend)")
    axes[1].set_ylabel("Validation ranking metric")
    fig.tight_layout()
    fig.savefig(Path(output) / "learning_curves.png", dpi=160)
    plt.close(fig)
