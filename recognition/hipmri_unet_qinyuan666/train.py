"""Reproducible midpoint training; synthetic mode never represents HipMRI results.

Validation selects the best checkpoint. This script does not tune on the test
split. Metrics aggregate pixels within each split; a full patient-level clinical
evaluation remains future work. See README for scope and architecture references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HipMRIDataset, SyntheticSegmentationDataset, validate_label_values
from metrics import confusion_matrix, scores_from_confusion, segmentation_loss
from modules import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_epoch(model, loader, device, num_classes, optimizer=None) -> dict:
    """Train or evaluate a split, accumulating confusion counts over all pixels."""
    training = optimizer is not None
    model.train(training)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    loss_total, sample_count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            images, targets = batch["image"].to(device), batch["mask"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = segmentation_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; stop and inspect the pipeline.")
            if training:
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("Non-finite gradients; refusing to update the model.")
                optimizer.step()
            matrix += confusion_matrix(logits.detach(), targets, num_classes)
            loss_total += float(loss.detach()) * images.shape[0]
            sample_count += images.shape[0]
    return {"loss": loss_total / sample_count, **scores_from_confusion(matrix.cpu())}


def benchmark_inference(model, image, device, warmup=5, repeats=20) -> dict:
    """Batch-one model-only latency, excluding file I/O and preprocessing."""
    model.eval()
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(image)
        synchronize(device)
        for _ in range(repeats):
            start = time.perf_counter()
            model(image)
            synchronize(device)
            times.append((time.perf_counter() - start) * 1000)
    return {"batch_size": 1, "warmup": warmup, "repeats": repeats,
            "median_ms": statistics.median(times), "mean_ms": statistics.mean(times),
            "scope": "model only; excludes disk I/O and preprocessing; CUDA synchronized"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("unet", "improved_unet"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="geometric software checks only")
    source.add_argument("--manifest", type=Path, help="audited patient-disjoint CSV")
    parser.add_argument("--label-values", type=int, nargs="+")
    parser.add_argument("--prostate-label", type=int, help="documented raw label, never guessed")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--resize-mode", choices=("pad", "stretch"), default="pad",
                        help="Real slices: aspect-preserving fit and pad, or legacy square stretch.")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples", type=int, default=32, help="synthetic training samples only")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--augment", action="store_true", help="train-only aligned flips on real slices")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.base_channels, args.samples, args.threads) < 1:
        parser.error("epochs, batch-size, width, samples and threads must be positive")
    if args.image_size < 32 or not math.isfinite(args.lr) or args.lr <= 0 or args.seed < 0:
        parser.error("image-size must be >=32, lr finite and positive, and seed non-negative")
    if args.synthetic:
        if args.label_values is not None or args.prostate_label is not None or args.augment:
            parser.error("synthetic mode has fixed labels [0,1]; real label/augmentation flags do not apply")
        args.label_values = [0, 1]
    else:
        if args.label_values is None or args.prostate_label is None:
            parser.error("real data requires explicit --label-values and --prostate-label after audit")
        try:
            args.label_values = list(validate_label_values(args.label_values))
        except ValueError as exc:
            parser.error(str(exc))
        if args.prostate_label == 0 or args.prostate_label not in args.label_values:
            parser.error("prostate-label must be a documented foreground member of label-values")
    return args


def train(args) -> dict:
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "best.pt").exists():
        raise FileExistsError("Output already contains best.pt; choose a new experiment directory.")
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config["num_classes"] = len(args.label_values)
    config["data_kind"] = "synthetic_geometric" if args.synthetic else "HipMRI_manifest"
    if args.synthetic:
        training_data = SyntheticSegmentationDataset(args.samples, args.image_size, args.seed)
        validation_data = SyntheticSegmentationDataset(max(4, args.samples // 4), args.image_size, args.seed + 1000)
    else:
        config["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        training_data = HipMRIDataset(args.manifest, "train", args.image_size,
                                     args.label_values, augment=args.augment,
                                     resize_mode=args.resize_mode)
        validation_data = HipMRIDataset(args.manifest, "val", args.image_size,
                                       args.label_values, augment=False,
                                       resize_mode=args.resize_mode)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, generator=generator)
    val_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(args.model, num_classes=config["num_classes"],
                        base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    initial_validation = run_epoch(model, val_loader, device, config["num_classes"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    history, best_loss, best_epoch = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_scores = run_epoch(model, train_loader, device, config["num_classes"], optimizer)
        val_scores = run_epoch(model, val_loader, device, config["num_classes"])
        history.append({"epoch": epoch, "train": train_scores, "val": val_scores})
        print(json.dumps({"epoch": epoch, "train_loss": train_scores["loss"],
                          "val_loss": val_scores["loss"], "data_kind": config["data_kind"]}), flush=True)
        if val_scores["loss"] < best_loss:
            best_loss, best_epoch = val_scores["loss"], epoch
            torch.save({"format_version": 1, "model_state": model.state_dict(),
                        "config": config, "epoch": epoch}, args.output / "best.pt")
    synchronize(device)
    elapsed = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    saved = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    model.eval()
    sample = validation_data[0]["image"].unsqueeze(0).to(device)
    with torch.inference_mode():
        reference = model(sample)
    reloaded = build_model(args.model, num_classes=config["num_classes"],
                           base_channels=args.base_channels).to(device)
    reloaded.load_state_dict(saved["model_state"])
    reloaded.eval()
    with torch.inference_mode():
        restored = reloaded(sample)
    exact_reload = torch.equal(reference, restored)
    reload_max_error = float((reference - restored).abs().max())
    if not exact_reload:
        raise RuntimeError("Reloaded model predictions differ from the saved model.")
    del reloaded, restored, reference
    latency = benchmark_inference(model, sample, device)
    result = {
        "status": "completed", "data_kind": config["data_kind"],
        "clinical_performance_claim": False if args.synthetic else None,
        "config": config, "training_samples": len(training_data),
        "validation_samples": len(validation_data), "test_used_for_selection": False,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
                        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                        "os": platform.platform(), "dtype": "float32"},
        "initial_validation": initial_validation, "history": history,
        "best_epoch": best_epoch, "best_validation": history[best_epoch - 1]["val"],
        "training_seconds_including_validation_and_checkpoint": elapsed,
        "peak_training_allocated_bytes": peak_allocated,
        "peak_training_reserved_bytes": peak_reserved,
        "memory_scope": "PyTorch CUDA allocator; excludes driver and other processes",
        "inference": latency, "checkpoint_reload_exact": exact_reload,
        "checkpoint_reload_max_abs_error": reload_max_error,
        "limitations": ["Synthetic data is not evidence of HipMRI segmentation performance."] if args.synthetic
        else ["Pixel-aggregated metrics; patient-level and physical-boundary evaluation still required."],
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    epochs = [row["epoch"] for row in history]
    for split in ("train", "val"):
        axes[0].plot(epochs, [row[split]["loss"] for row in history], label=split)
        axes[1].plot(epochs, [row[split]["macro_foreground_dice"] for row in history], label=split)
    axes[0].set(ylabel="CE + foreground soft Dice loss", xlabel="Epoch")
    axes[1].set(ylabel="Pixel-aggregated foreground Dice", xlabel="Epoch", ylim=(0, 1))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.suptitle(("SYNTHETIC SOFTWARE CHECK ONLY - " if args.synthetic else "HipMRI - ") + args.model)
    fig.tight_layout()
    fig.savefig(args.output / "training_curves.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"summary": str(args.output / "summary.json"), "reload_exact": exact_reload,
                      "best_epoch": best_epoch, "seconds": round(elapsed, 2)}), flush=True)
    return result


if __name__ == "__main__":
    train(parse_args())
