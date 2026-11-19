"""Train and validate a standard 2D U-Net variant on HipMRI axial slices.

Example (Rangpur):
    python train.py --model unet2d_256 \
        --metadata /path/to/metadata.csv --out-dir runs/unet2d_256_seed3710_e20

Training
    All training slices once per epoch, shuffled by a seeded torch.Generator.
    AdamW, fixed epoch budget, no scheduler, no early stopping, no augmentation.
    Loss: 0.5 batch-level foreground Dice + 0.5 cross-entropy (losses.py).

Validation (every epoch)
    Slices are predicted per volume, reassembled into volumes, and scored with
    global Dice (one confusion matrix over all validation volumes) and
    per-volume Dice (mean and SD). ``best.pt`` is the epoch with the highest
    global mean foreground Dice. The test split is never loaded unless
    ``--evaluate-test`` is passed.

Overfit sanity check
    ``--overfit-subset 8`` trains on 8 seeded foreground slices from the
    training split and reports foreground Dice on those same slices.
    0.90 is a diagnostic target only.

Outputs in --out-dir: history.csv, summary.json, best.pt, last.pt,
loss_curves.png, dice_curves.png (and test_metrics.json if requested).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from dataset import (  # noqa: E402
    EXPECTED_PATIENTS_PER_SPLIT,
    EXPECTED_SPLIT_SHA256,
    NUM_CLASSES,
    SliceDataset,
    VolumeSet,
    load_manifest,
    select_overfit_slices,
    split_sha256,
)
from losses import DiceCrossEntropyLoss  # noqa: E402
from metrics import CLASS_NAMES, confusion_matrix, global_metrics, per_volume_metrics  # noqa: E402
from modules import UNET2D_VARIANTS, build_model, count_parameters  # noqa: E402

OVERFIT_DIAGNOSTIC_TARGET = 0.90


# =============================================================================
# Setup helpers
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=sorted(UNET2D_VARIANTS))
    parser.add_argument("--metadata", required=True, type=Path, help="Frozen scan manifest CSV")
    parser.add_argument("--out-dir", required=True, type=Path, help="New, empty run directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--eval-batch-size", type=int, default=16, help="Slices per inference chunk"
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument(
        "--overfit-subset", type=int, default=0, help="Train on N slices only (0 = off)"
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="After training, evaluate best.pt on the held-out test split. Not for model selection.",
    )
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Seeds are fixed, but AMP/cuDNN kernels are not guaranteed bitwise reproducible.
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_out_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{path} already exists and is not empty; choose a new --out-dir")
    path.mkdir(parents=True, exist_ok=True)


def peak_rss_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return peak / 1024**2 if sys.platform == "darwin" else peak / 1024


def git_revision() -> dict:
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "."],
                cwd=here,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "uncommitted_changes_in_project_dir": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes_in_project_dir": None}


def environment_info(device: torch.device) -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    return info


# =============================================================================
# Training and evaluation steps
# =============================================================================
def autocast(device: torch.device, enabled: bool):
    # Mixed precision is used on CUDA only; the loss itself always runs in FP32.
    return torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda"
    )


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, amp: bool, epoch: int
) -> dict:
    model.train()
    sums = {"loss": 0.0, "dice_loss": 0.0, "ce": 0.0}
    seen = 0
    for step, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            logits = model(images)
        terms = criterion(logits, masks)
        if not torch.isfinite(terms.total):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}: {terms}")
        scaler.scale(terms.total).backward()
        scaler.step(optimizer)
        scaler.update()

        n = images.shape[0]
        seen += n
        sums["loss"] += terms.total.item() * n
        sums["dice_loss"] += terms.dice.item() * n
        sums["ce"] += terms.ce.item() * n
    return {key: value / seen for key, value in sums.items()}


@torch.no_grad()
def predict_volume(
    model, image: np.ndarray, device, batch_size: int, amp: bool, mask=None, criterion=None
):
    """Predict every axial slice of one volume.

    Returns (prediction uint8 [H, W, D], loss sums or None). Loss sums are
    slice-weighted totals over inference chunks of ``batch_size`` slices.
    """
    model.eval()
    depth = image.shape[2]
    predictions = []
    sums = {"loss": 0.0, "dice_loss": 0.0, "ce": 0.0} if criterion is not None else None
    for start in range(0, depth, batch_size):
        stop = min(start + batch_size, depth)
        chunk = torch.from_numpy(np.ascontiguousarray(image[:, :, start:stop].transpose(2, 0, 1)))
        chunk = chunk.unsqueeze(1).to(device)  # [n, 1, H, W]
        with autocast(device, amp):
            logits = model(chunk)
        if criterion is not None:
            target = torch.from_numpy(
                np.ascontiguousarray(mask[:, :, start:stop].transpose(2, 0, 1)).astype(np.int64)
            ).to(device)
            terms = criterion(logits, target)
            n = stop - start
            sums["loss"] += terms.total.item() * n
            sums["dice_loss"] += terms.dice.item() * n
            sums["ce"] += terms.ce.item() * n
        predictions.append(logits.argmax(dim=1).to(torch.uint8).cpu())
    prediction = torch.cat(predictions).permute(1, 2, 0).numpy()  # [H, W, D]
    return prediction, sums


def evaluate_volumes(
    model, volumes: VolumeSet, criterion, device, batch_size: int, amp: bool
) -> dict:
    """Global and per-volume metrics plus mean loss over all slices."""
    matrices = []
    loss_sums = {"loss": 0.0, "dice_loss": 0.0, "ce": 0.0}
    for image, mask in zip(volumes.images, volumes.masks, strict=True):
        prediction, sums = predict_volume(model, image, device, batch_size, amp, mask, criterion)
        matrices.append(
            confusion_matrix(torch.from_numpy(mask), torch.from_numpy(prediction), NUM_CLASSES)
        )
        for key in loss_sums:
            loss_sums[key] += sums[key]
    total_matrix = torch.stack(matrices).sum(dim=0)
    slices = volumes.num_slices
    return {
        "loss": {key: value / slices for key, value in loss_sums.items()},
        "global": global_metrics(total_matrix),
        "per_volume": per_volume_metrics(matrices),
        "confusion_matrix": total_matrix.tolist(),
    }


@torch.no_grad()
def evaluate_slices(model, dataset: SliceDataset, device, batch_size: int, amp: bool) -> dict:
    """Global metrics over a set of individual slices (used for the overfit check)."""
    model.eval()
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    for images, masks in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        with autocast(device, amp):
            logits = model(images.to(device))
        matrix += confusion_matrix(masks, logits.argmax(dim=1).cpu(), NUM_CLASSES)
    return global_metrics(matrix)


# =============================================================================
# Logging and plots
# =============================================================================
def history_row(epoch: int, train: dict, evaluation: dict, seconds: float, overfit: bool) -> dict:
    row = {
        "epoch": epoch,
        "train_loss": train["loss"],
        "train_dice_loss": train["dice_loss"],
        "train_ce": train["ce"],
        "epoch_seconds": round(seconds, 2),
    }
    if overfit:
        g = evaluation
        row["subset_mean_foreground_dice"] = g["mean_foreground_dice"]
        row["subset_pixel_accuracy"] = g["pixel_accuracy"]
        for name, value in g["dice_per_class"].items():
            row[f"subset_dice_{name}"] = value
        return row
    g, pv = evaluation["global"], evaluation["per_volume"]
    row.update(
        {
            "val_loss": evaluation["loss"]["loss"],
            "val_dice_loss": evaluation["loss"]["dice_loss"],
            "val_ce": evaluation["loss"]["ce"],
            "val_mean_foreground_dice": g["mean_foreground_dice"],
            "val_volume_fg_dice_mean": pv["mean_foreground_dice"]["mean"],
            "val_volume_fg_dice_sd": pv["mean_foreground_dice"]["sd"],
            "val_pixel_accuracy": g["pixel_accuracy"],
        }
    )
    for name, value in g["dice_per_class"].items():
        row[f"val_dice_{name}"] = value
    return row


def write_history(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(out_dir: Path, rows: list[dict], overfit: bool) -> None:
    epochs = [r["epoch"] for r in rows]
    prefix = "subset" if overfit else "val"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(
        axes, ("loss", "dice_loss", "ce"), ("Total loss", "Dice loss", "Cross-entropy"), strict=True
    ):
        ax.plot(epochs, [r[f"train_{key}"] for r in rows], label="train")
        if not overfit:
            ax.plot(epochs, [r[f"val_{key}"] for r in rows], label="val")
        ax.set(title=title, xlabel="epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))

    def series(key: str) -> list[float]:
        return [np.nan if r[key] is None else r[key] for r in rows]

    ax.plot(epochs, series(f"{prefix}_mean_foreground_dice"), "k-", lw=2, label="mean foreground")
    for name in CLASS_NAMES[1:]:
        ax.plot(epochs, series(f"{prefix}_dice_{name}"), label=name)
    ax.set(title=f"{prefix} Dice (global)", xlabel="epoch", ylabel="Dice", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dice_curves.png", dpi=120)
    plt.close(fig)


def save_checkpoint(path: Path, model, args, epoch: int, metrics: dict) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "variant": args.model,
            "bottleneck_channels": UNET2D_VARIANTS[args.model],
            "epoch": epoch,
            "metrics": metrics,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        path,
    )


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
    overfit = args.overfit_subset > 0
    prepare_out_dir(args.out_dir)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    frame = load_manifest(args.metadata, expected_patients, expected_sha256)
    print(
        f"[data] manifest {args.metadata}: {frame['patient_id'].nunique()} patients, {len(frame)} scans"
    )

    # ---- Data -----------------------------------------------------------------
    if overfit:
        # Load only as many training volumes as needed to pick the subset.
        train_ids = frame.loc[frame["split"] == "train", "scan_id"].tolist()[: args.overfit_subset]
        train_volumes = VolumeSet.from_manifest(frame, "train", scan_ids=train_ids, verbose=True)
        indices = select_overfit_slices(train_volumes, args.overfit_subset, args.seed)
        train_dataset = SliceDataset(train_volumes, indices)
        val_volumes = None
        print(f"[data] overfit subset {indices}")
    else:
        train_volumes = VolumeSet.from_manifest(frame, "train", verbose=True)
        val_volumes = VolumeSet.from_manifest(frame, "val", verbose=True)
        train_dataset = SliceDataset(train_volumes)
        print(
            f"[data] train {len(train_volumes)} volumes / {len(train_dataset)} slices, "
            f"val {len(val_volumes)} volumes / {val_volumes.num_slices} slices"
        )
    resident_bytes = train_volumes.nbytes + (val_volumes.nbytes if val_volumes else 0)
    load_seconds = train_volumes.load_seconds + (val_volumes.load_seconds if val_volumes else 0)
    print(f"[data] preloaded {resident_bytes / 1024**3:.2f} GiB in {load_seconds:.0f} s")

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,  # slices come from preloaded arrays; workers would duplicate memory
        pin_memory=device.type == "cuda",
    )

    # ---- Model, loss, optimiser -----------------------------------------------
    model = build_model(args.model).to(device)
    parameters = count_parameters(model)
    print(
        f"[model] {args.model}: bottleneck {UNET2D_VARIANTS[args.model]}, {parameters:,} trainable parameters"
    )
    criterion = DiceCrossEntropyLoss(
        NUM_CLASSES, dice_weight=args.dice_weight, ce_weight=args.ce_weight
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # ---- Epoch loop ----------------------------------------------------------
    rows: list[dict] = []
    best_score, best_epoch, best_eval = -1.0, None, None
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch(
            model, loader, criterion, optimizer, scaler, device, args.amp, epoch
        )
        if overfit:
            evaluation = evaluate_slices(
                model, train_dataset, device, args.eval_batch_size, args.amp
            )
            score = evaluation["mean_foreground_dice"]
        else:
            evaluation = evaluate_volumes(
                model, val_volumes, criterion, device, args.eval_batch_size, args.amp
            )
            score = evaluation["global"]["mean_foreground_dice"]
        seconds = time.perf_counter() - epoch_start

        rows.append(history_row(epoch, train_stats, evaluation, seconds, overfit))
        write_history(args.out_dir / "history.csv", rows)
        plot_curves(args.out_dir, rows, overfit)
        save_checkpoint(args.out_dir / "last.pt", model, args, epoch, evaluation)
        if score is not None and score > best_score:
            best_score, best_epoch, best_eval = score, epoch, evaluation
            save_checkpoint(args.out_dir / "best.pt", model, args, epoch, evaluation)

        label = "subset" if overfit else "val"
        print(
            f"[epoch {epoch:3d}/{args.epochs}] train loss {train_stats['loss']:.4f} | "
            f"{label} mean fg Dice {score if score is not None else float('nan'):.4f} | {seconds:.0f} s",
            flush=True,
        )
    train_seconds = time.perf_counter() - train_start

    # ---- Summary -------------------------------------------------------------
    summary = {
        "variant": args.model,
        "bottleneck_channels": UNET2D_VARIANTS[args.model],
        "trainable_parameters": parameters,
        "mode": "overfit_subset" if overfit else "train_val",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "split_sha256": split_sha256(frame),
        "scans_per_split": {k: int(v) for k, v in frame["split"].value_counts().items()},
        "train_slices": len(train_dataset),
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
            "peak_gpu_allocated_mib": (
                round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
                if device.type == "cuda"
                else None
            ),
            "peak_gpu_reserved_mib": (
                round(torch.cuda.max_memory_reserved(device) / 1024**2, 1)
                if device.type == "cuda"
                else None
            ),
        },
        "environment": environment_info(device),
        "git": git_revision(),
    }
    if overfit:
        summary["overfit_slices"] = [list(i) for i in train_dataset.indices]
        summary["overfit_diagnostic_target"] = OVERFIT_DIAGNOSTIC_TARGET
        summary["overfit_final_mean_foreground_dice"] = evaluation["mean_foreground_dice"]

    if args.evaluate_test and not overfit:
        print("[test] evaluating best.pt on the held-out test split")
        checkpoint = torch.load(args.out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        test_volumes = VolumeSet.from_manifest(frame, "test", verbose=True)
        test_metrics = evaluate_volumes(
            model, test_volumes, criterion, device, args.eval_batch_size, args.amp
        )
        test_metrics["checkpoint_epoch"] = checkpoint["epoch"]
        (args.out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n")

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] best epoch {best_epoch}, score {best_score:.4f}; outputs in {args.out_dir}")
    return summary


def main(argv: list[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
