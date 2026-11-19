"""Train and validate a 3D U-Net variant (standard or residual) on HipMRI patches.

Example (GPU Lab):
    python train3d.py --model unet3d_256 \
        --metadata /home/s4890175/hipmri/metadata_gpu.csv \
        --out-dir runs/unet3d_256_p96_seed3710_e20

    The residual models (improved/unet3d) use the same script and settings:
    --model resunet3d_256 or resunet3d_512, with a separate --out-dir.

Training
    Each epoch draws a fresh seeded patch plan (dataset.PatchSampler) from the
    training split: 96^3 class-balanced patches with fg_prob 0.33 by default.
    By default an epoch has as many patches as it takes to cover every training
    voxel once (dataset.volume_equivalent_patches), the same voxel budget as one
    2D epoch. AdamW, fixed epoch budget, no scheduler, no augmentation.
    Loss: 0.5 batch-level foreground Dice + 0.5 cross-entropy (losses.py).

Validation (every epoch)
    Every validation volume is predicted whole by sliding-window inference
    (sliding_window.py, 50% overlap, uniform averaging) and scored with the
    same global and per-volume Dice as the 2D baselines (metrics.py).
    ``best.pt`` is the epoch with the highest global mean foreground Dice.
    The test split is never loaded by this script.

Overfit sanity check
    ``--overfit-patches 4`` trains on 4 fixed foreground-centred training
    patches and reports Dice on those same patches. 0.90 is a diagnostic target.

GPU memory probe
    ``--memory-probe`` loads no data: it times a few training steps on random
    patches for each ``--probe-batch-sizes`` value, plus one sliding-window pass
    over a 128x256x256 volume (--probe-volume), and writes memory_probe.json with peak memory.

Outputs in --out-dir: history.csv, summary.json, best.pt, last.pt,
loss_curves.png, dice_curves.png (memory_probe.json for the probe).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    EXPECTED_PATIENTS_PER_SPLIT,
    EXPECTED_SPLIT_SHA256,
    NUM_CLASSES,
    SAMPLING_MODES,
    PatchDataset,
    PatchSampler,
    VolumeSet,
    load_manifest,
    split_sha256,
    to_model_layout,
    volume_equivalent_patches,
)
from improved.unet3d.residual_blocks import RESUNET3D_VARIANTS, build_resunet3d
from losses import DiceCrossEntropyLoss
from metrics import confusion_matrix, global_metrics, per_volume_metrics
from modules import UNET3D_VARIANTS, build_model3d, count_parameters
from sliding_window import sliding_window_logits
from train import (
    OVERFIT_DIAGNOSTIC_TARGET,
    autocast,
    environment_info,
    git_revision,
    history_row,
    peak_rss_mib,
    plot_curves,
    prepare_out_dir,
    resolve_device,
    seed_everything,
    train_one_epoch,
    write_history,
)

# Standard (modules.py) and improved residual (improved/unet3d) models; bottleneck widths.
MODEL_VARIANTS: dict[str, int] = {**UNET3D_VARIANTS, **RESUNET3D_VARIANTS}


def build_variant(variant: str):
    """Standard UNet3D or residual ResUNet3D; training is identical for both."""
    if variant in RESUNET3D_VARIANTS:
        return build_resunet3d(variant)
    return build_model3d(variant)


# =============================================================================
# Setup helpers
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_VARIANTS))
    parser.add_argument("--metadata", type=Path, help="Frozen scan manifest CSV")
    parser.add_argument("--out-dir", required=True, type=Path, help="New, empty run directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--eval-batch-size", type=int, default=4, help="Sliding windows per inference batch"
    )
    parser.add_argument("--patch-size", type=int, default=96, help="Cubic patch side")
    parser.add_argument(
        "--patches-per-epoch",
        type=int,
        default=None,
        help="Default: enough patches to cover every training voxel once",
    )
    parser.add_argument("--sampling", choices=SAMPLING_MODES, default="class_balanced")
    parser.add_argument("--fg-prob", type=float, default=0.33, help="class_balanced only")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument(
        "--overfit-patches", type=int, default=0, help="Train on N fixed patches only (0 = off)"
    )
    parser.add_argument("--memory-probe", action="store_true", help="Measure memory, no data")
    parser.add_argument("--probe-batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--probe-steps", type=int, default=3)
    parser.add_argument(
        "--probe-volume",
        type=int,
        nargs=3,
        default=[128, 256, 256],
        metavar=("D", "H", "W"),
        help="Volume for the inference probe (default: the usual HipMRI scan shape)",
    )
    args = parser.parse_args(argv)

    if args.patch_size % 16:
        parser.error(f"--patch-size must be divisible by 16 (four poolings), got {args.patch_size}")
    if args.sampling == "uniform":
        args.fg_prob = None  # PatchSampler rejects fg_prob with uniform sampling
    if args.metadata is None and not args.memory_probe:
        parser.error("--metadata is required unless --memory-probe is given")
    return args


def gpu_peak_mib(device: torch.device) -> dict:
    """Peak allocated/reserved CUDA memory since the last reset (None off CUDA)."""
    if device.type != "cuda":
        return {"allocated_mib": None, "reserved_mib": None}
    return {
        "allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
        "reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
    }


def reset_gpu_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


# =============================================================================
# Validation and evaluation
# =============================================================================
@torch.no_grad()
def predict_volume3d(
    model, image: np.ndarray, device, size, eval_batch_size: int, amp: bool, overlap: float
) -> torch.Tensor:
    """Sliding-window logits [C, D, H, W] (FP32, on ``device``) for one [H, W, D] image."""
    model.eval()

    def predict(batch: torch.Tensor) -> torch.Tensor:
        with autocast(device, amp):
            return model(batch)

    volume = torch.from_numpy(to_model_layout(image)).unsqueeze(0).to(device)  # [1, D, H, W]
    return sliding_window_logits(
        predict, volume, size, NUM_CLASSES, overlap=overlap, batch_size=eval_batch_size
    )


@torch.no_grad()
def evaluate_volumes3d(
    model, volumes: VolumeSet, criterion, device, size, eval_batch_size: int, amp: bool, overlap
) -> dict:
    """Global and per-volume metrics plus voxel-weighted mean loss over whole volumes.

    One volume is on the GPU at a time; only its uint8 prediction is kept, on the CPU.
    """
    matrices = []
    loss_sums = {"loss": 0.0, "dice_loss": 0.0, "ce": 0.0}
    voxels = 0
    for image, mask in zip(volumes.images, volumes.masks, strict=True):
        logits = predict_volume3d(model, image, device, size, eval_batch_size, amp, overlap)
        target = torch.from_numpy(to_model_layout(mask).astype(np.int64)).to(device)
        terms = criterion(logits.unsqueeze(0), target.unsqueeze(0))
        n = target.numel()
        voxels += n
        loss_sums["loss"] += terms.total.item() * n
        loss_sums["dice_loss"] += terms.dice.item() * n
        loss_sums["ce"] += terms.ce.item() * n
        prediction = logits.argmax(dim=0).to(torch.uint8).cpu()
        del logits, target, terms  # free the full-volume tensors before the next volume
        matrices.append(
            confusion_matrix(torch.from_numpy(to_model_layout(mask)), prediction, NUM_CLASSES)
        )
    total_matrix = torch.stack(matrices).sum(dim=0)
    return {
        "loss": {key: value / voxels for key, value in loss_sums.items()},
        "global": global_metrics(total_matrix),
        "per_volume": per_volume_metrics(matrices),
        "confusion_matrix": total_matrix.tolist(),
    }


@torch.no_grad()
def evaluate_patches(model, dataset: PatchDataset, device, batch_size: int, amp: bool) -> dict:
    """Global metrics over a fixed set of patches (used for the overfit check)."""
    model.eval()
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    for images, masks in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        with autocast(device, amp):
            logits = model(images.to(device))
        matrix += confusion_matrix(masks, logits.argmax(dim=1).cpu(), NUM_CLASSES)
    return global_metrics(matrix)


def save_checkpoint3d(path: Path, model, args, epoch: int, metrics: dict) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "variant": args.model,
            "bottleneck_channels": MODEL_VARIANTS[args.model],
            "patch_size": args.patch_size,
            "epoch": epoch,
            "metrics": metrics,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        path,
    )


# =============================================================================
# GPU memory probe
# =============================================================================
def run_memory_probe(args: argparse.Namespace) -> dict:
    """Peak memory and step time for a few random training steps per batch size.

    No data is loaded and nothing is saved except memory_probe.json.
    """
    seed_everything(args.seed)
    device = resolve_device(args.device)
    prepare_out_dir(args.out_dir)
    model = build_variant(args.model).to(device)
    criterion = DiceCrossEntropyLoss(NUM_CLASSES, args.dice_weight, args.ce_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    size = (args.patch_size,) * 3

    results = {}
    for batch_size in args.probe_batch_sizes:
        reset_gpu_peak(device)
        try:
            images = torch.randn(batch_size, 1, *size)
            masks = torch.randint(0, NUM_CLASSES, (batch_size, *size))
            loader = [(images, masks)] * args.probe_steps
            start = time.perf_counter()
            train_one_epoch(model, loader, criterion, optimizer, scaler, device, args.amp, 0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = (time.perf_counter() - start) / args.probe_steps
            results[str(batch_size)] = {"status": "ok", "seconds_per_step": round(seconds, 3)}
            results[str(batch_size)].update(gpu_peak_mib(device))
        except torch.cuda.OutOfMemoryError:
            results[str(batch_size)] = {"status": "out_of_memory"}
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[probe] batch {batch_size}: {results[str(batch_size)]}", flush=True)

    reset_gpu_peak(device)
    d, h, w = args.probe_volume
    start = time.perf_counter()
    logits = predict_volume3d(
        model, np.zeros((h, w, d), np.float32), device, size, args.eval_batch_size, args.amp, 0.5
    )
    del logits
    inference = {"volume_dhw": list(args.probe_volume), "eval_batch_size": args.eval_batch_size}
    inference["seconds"] = round(time.perf_counter() - start, 2)
    inference.update(gpu_peak_mib(device))
    print(f"[probe] sliding-window inference: {inference}", flush=True)

    report = {
        "variant": args.model,
        "trainable_parameters": count_parameters(model),
        "patch_size": args.patch_size,
        "amp": args.amp,
        "training_steps": results,
        "sliding_window_inference": inference,
        "environment": environment_info(device),
    }
    if device.type == "cuda":
        report["gpu_total_mib"] = round(
            torch.cuda.get_device_properties(device).total_memory / 1024**2, 1
        )
    (args.out_dir / "memory_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


# =============================================================================
# Main experiment
# =============================================================================
def run_training(
    args: argparse.Namespace,
    expected_patients: dict[str, int] | None = EXPECTED_PATIENTS_PER_SPLIT,
    expected_sha256: str | None = EXPECTED_SPLIT_SHA256,
) -> dict:
    """Run one experiment. The expected_* arguments exist so tests can use synthetic manifests."""
    seed_everything(args.seed)
    device = resolve_device(args.device)
    overfit = args.overfit_patches > 0
    size = (args.patch_size,) * 3
    prepare_out_dir(args.out_dir)
    reset_gpu_peak(device)

    frame = load_manifest(args.metadata, expected_patients, expected_sha256)
    print(
        f"[data] manifest {args.metadata}: "
        f"{frame['patient_id'].nunique()} patients, {len(frame)} scans"
    )

    # ---- Data (the test split is never loaded) -------------------------------
    if overfit:
        train_ids = frame.loc[frame["split"] == "train", "scan_id"].tolist()[: args.overfit_patches]
        train_volumes = VolumeSet.from_manifest(frame, "train", scan_ids=train_ids, verbose=True)
        sampler = PatchSampler(train_volumes, size, "class_balanced", args.seed, fg_prob=1.0)
        fixed_plan = sampler.plan(epoch=0, count=args.overfit_patches)
        overfit_dataset = PatchDataset(train_volumes, fixed_plan, size)
        val_volumes = None
        patches_per_epoch = args.overfit_patches
        print(f"[data] overfit patches (volume, d, h, w): {fixed_plan}")
    else:
        train_volumes = VolumeSet.from_manifest(frame, "train", verbose=True)
        val_volumes = VolumeSet.from_manifest(frame, "val", verbose=True)
        sampler = PatchSampler(train_volumes, size, args.sampling, args.seed, args.fg_prob)
        patches_per_epoch = args.patches_per_epoch or volume_equivalent_patches(train_volumes, size)
        print(
            f"[data] train {len(train_volumes)} volumes, {patches_per_epoch} {args.sampling} "
            f"{args.patch_size}^3 patches per epoch; val {len(val_volumes)} volumes"
        )
    resident_bytes = train_volumes.nbytes + (val_volumes.nbytes if val_volumes else 0)
    load_seconds = train_volumes.load_seconds + (val_volumes.load_seconds if val_volumes else 0)
    print(f"[data] preloaded {resident_bytes / 1024**3:.2f} GiB in {load_seconds:.0f} s")

    # ---- Model, loss, optimiser -----------------------------------------------
    model = build_variant(args.model).to(device)
    parameters = count_parameters(model)
    print(
        f"[model] {args.model}: bottleneck {MODEL_VARIANTS[args.model]}, "
        f"{parameters:,} trainable parameters"
    )
    criterion = DiceCrossEntropyLoss(
        NUM_CLASSES, dice_weight=args.dice_weight, ce_weight=args.ce_weight
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # ---- Epoch loop ----------------------------------------------------------
    rows: list[dict] = []
    best_score, best_epoch, best_eval = -1.0, None, None
    peak_train, peak_val = {}, {}
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        if overfit:
            train_dataset = overfit_dataset  # the same patches every epoch
        else:
            train_dataset = PatchDataset(
                train_volumes, sampler.plan(epoch, patches_per_epoch), size
            )
        plan_seconds = time.perf_counter() - epoch_start
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # the seeded plan already fixes a random order
            num_workers=0,  # patches come from preloaded arrays; workers would duplicate memory
            pin_memory=device.type == "cuda",
        )

        reset_gpu_peak(device)
        train_stats = train_one_epoch(
            model, loader, criterion, optimizer, scaler, device, args.amp, epoch
        )
        peak_train = gpu_peak_mib(device)
        reset_gpu_peak(device)
        if overfit:
            evaluation = evaluate_patches(
                model, overfit_dataset, device, args.eval_batch_size, args.amp
            )
            score = evaluation["mean_foreground_dice"]
        else:
            evaluation = evaluate_volumes3d(
                model,
                val_volumes,
                criterion,
                device,
                size,
                args.eval_batch_size,
                args.amp,
                args.overlap,
            )
            score = evaluation["global"]["mean_foreground_dice"]
        peak_val = gpu_peak_mib(device)
        seconds = time.perf_counter() - epoch_start

        row = history_row(epoch, train_stats, evaluation, seconds, overfit)
        row["plan_seconds"] = round(plan_seconds, 2)
        row["gpu_peak_train_allocated_mib"] = peak_train["allocated_mib"]
        row["gpu_peak_eval_allocated_mib"] = peak_val["allocated_mib"]
        rows.append(row)
        write_history(args.out_dir / "history.csv", rows)
        plot_curves(args.out_dir, rows, overfit)
        save_checkpoint3d(args.out_dir / "last.pt", model, args, epoch, evaluation)
        if score is not None and score > best_score:
            best_score, best_epoch, best_eval = score, epoch, evaluation
            save_checkpoint3d(args.out_dir / "best.pt", model, args, epoch, evaluation)

        label = "subset" if overfit else "val"
        print(
            f"[epoch {epoch:3d}/{args.epochs}] train loss {train_stats['loss']:.4f} | "
            f"{label} mean fg Dice {score if score is not None else float('nan'):.4f} | "
            f"{seconds:.0f} s",
            flush=True,
        )
    train_seconds = time.perf_counter() - train_start

    # ---- Summary -------------------------------------------------------------
    def column_max(key: str) -> float | None:
        values = [r[key] for r in rows if r[key] is not None]
        return max(values) if values else None

    summary = {
        "variant": args.model,
        "bottleneck_channels": MODEL_VARIANTS[args.model],
        "trainable_parameters": parameters,
        "mode": "overfit_patches" if overfit else "train_val",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "split_sha256": split_sha256(frame),
        "scans_per_split": {k: int(v) for k, v in frame["split"].value_counts().items()},
        "patch_size_dhw": list(size),
        "patches_per_epoch": patches_per_epoch,
        "best_epoch": best_epoch,
        "best_selection_score": best_score if best_epoch else None,
        "best_metrics": best_eval,
        "final_metrics": evaluation,
        "runtime": {
            "data_preload_seconds": round(load_seconds, 1),
            "training_seconds": round(train_seconds, 1),
            "mean_epoch_seconds": round(train_seconds / args.epochs, 1),
        },
        "memory": {
            "resident_dataset_gib": round(resident_bytes / 1024**3, 3),
            "peak_process_rss_mib": round(peak_rss_mib(), 1),
            "peak_gpu_train_allocated_mib": column_max("gpu_peak_train_allocated_mib"),
            "peak_gpu_eval_allocated_mib": column_max("gpu_peak_eval_allocated_mib"),
            "last_epoch_gpu_train": peak_train,
            "last_epoch_gpu_eval": peak_val,
        },
        "environment": environment_info(device),
        "git": git_revision(),
    }
    if overfit:
        summary["overfit_plan"] = [list(p) for p in fixed_plan]
        summary["overfit_diagnostic_target"] = OVERFIT_DIAGNOSTIC_TARGET
        summary["overfit_final_mean_foreground_dice"] = evaluation["mean_foreground_dice"]

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] best epoch {best_epoch}, score {best_score:.4f}; outputs in {args.out_dir}")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.memory_probe:
        run_memory_probe(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
