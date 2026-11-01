"""Train the HipMRI residual 3D U-Net. No training runs on import."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    PatchDataset,
    discover_pairs,
    load_volume,
    split_three_way,
    prepare_cache,
    cached_volume,
    clear_volume_cache,
)
from predict import sliding_window_predict
from modules import ImprovedUNet3D, deep_supervision_loss


def seed_worker(worker_id):
    """Seed NumPy and Python sampling from the DataLoader worker seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def optimizer_step(model, optimizer, scaler):
    """Let AMP reject overflowed gradients before clipping; never apply them."""
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.grad is not None]
    if (
        scaler.is_enabled()
        and not torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()
    ):
        # unscale_ recorded the overflow. step skips the update; update lowers scale.
        scaler.step(optimizer)
        scaler.update()
        return False
    torch.nn.utils.clip_grad_norm_(parameters, 12.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return True


def validate(model, pairs, args, device):
    """Return mean foreground Dice and per-class Dice pooled over all scans.

    Inference receives only MRI data. Targets are used afterwards to accumulate
    voxel counts; the caller selects the configured class (prostate by default)
    for scheduling and best-checkpoint selection, not the foreground mean.
    """
    intersection = np.zeros(args.num_classes, dtype=np.float64)
    denominator = np.zeros_like(intersection)
    for pair in pairs:
        if args.cache_dir:
            image, target, _ = cached_volume(pair, args.num_classes, args.cache_dir)
        else:
            image, target = load_volume(pair, args.num_classes)
        prediction = sliding_window_predict(
            model,
            image,
            args.patch_size,
            args.num_classes,
            device,
            window_batch_size=args.window_batch_size,
        )
        for label in range(args.num_classes):
            predicted, actual = prediction == label, target == label
            intersection[label] += np.count_nonzero(predicted & actual)
            denominator[label] += np.count_nonzero(predicted) + np.count_nonzero(actual)
    # Classes absent in both prediction and ground truth are undefined, not perfect.
    dice = np.divide(
        2 * intersection, denominator, out=np.full_like(intersection, np.nan), where=denominator > 0
    )
    if not np.isfinite(dice[1:]).any():
        raise ValueError("Validation contains no foreground predictions or labels")
    return float(np.nanmean(dice[1:])), dice.tolist()


def save_checkpoint(path, state):
    """Replace the checkpoint only after its temporary file has been written."""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def plot_training_curves(metrics_path):
    """Rebuild plots from persisted history; unmeasured Dice remains NaN."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with metrics_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    losses = [float(row["train_loss"]) for row in rows]
    dice = [float(row["dice_5"]) if row.get("dice_5") else float("nan") for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, losses, marker="o")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, dice, marker="o")
    axes[1].set(
        title="Validation prostate Dice (label 5)", xlabel="Epoch", ylabel="Dice", ylim=(0, 1)
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(metrics_path.with_name("training_curves.png"), dpi=140)
    plt.close(figure)
    return epochs, losses, dice


def parse_args():
    """Validate training options or route a checkpoint-only evaluation request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, help="Training dataset or optional evaluation relocation"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/unet3d"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patches-per-volume", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--architecture", choices=["baseline", "improved"], default="improved")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--selection-class",
        type=int,
        default=5,
        help="Validation class used to select best model; prostate is 5",
    )
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Local disk cache of normalized volumes and foreground indices",
    )
    parser.add_argument(
        "--window-batch-size",
        type=int,
        default=1,
        help="Validation windows per forward pass; does not change training batches",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Synchronize CUDA at batch boundaries for accurate loading/compute timing",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--evaluate-only", action="store_true", help="Evaluate saved test patients without training"
    )
    parser.add_argument("--evaluation-output", type=Path)
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Validate every MRI/mask pair and exit without training",
    )
    args = parser.parse_args()
    if args.evaluate_only:
        if not args.resume or not args.evaluation_output:
            parser.error("--evaluate-only requires --resume and --evaluation-output")
        if args.check_data:
            parser.error("--evaluate-only cannot be combined with --check-data")
        return args
    if args.evaluation_output:
        parser.error("--evaluation-output requires --evaluate-only")
    if args.data_root is None:
        args.data_root = Path(__file__).resolve().parent / "HipMRI_Study_open"
    for name in (
        "epochs",
        "batch_size",
        "patches_per_volume",
        "base_channels",
        "val_every",
        "window_batch_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if min(args.patch_size) < 16 or args.num_classes < 2 or args.workers < 0:
        parser.error("Patch dimensions must be >=16, classes >=2, and workers >=0")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Learning rate must be positive and weight decay nonnegative")
    if not 1 <= args.selection_class < args.num_classes:
        parser.error("selection-class must be a foreground class")
    return args


def main():
    """Run data checks, held-out evaluation, or reproducible training with resume."""
    args = parse_args()
    if args.evaluate_only:
        # Delegate before discovering data or constructing any optimizer.
        from evaluate_3d import main as evaluate

        arguments = [
            "--checkpoint",
            str(args.resume),
            "--output-dir",
            str(args.evaluation_output),
            "--device",
            args.device,
            "--split",
            "test",
            "--window-batch-size",
            str(args.window_batch_size),
        ]
        if args.data_root is not None:
            arguments += ["--data-root", str(args.data_root)]
        evaluate(arguments)
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pairs = discover_pairs(args.data_root.resolve())
    split = split_three_way(pairs, args.val_fraction, args.test_fraction, args.seed)
    train_pairs, val_pairs = split["train"], split["validation"]
    print(
        f'Volumes: {len(train_pairs)} train / {len(val_pairs)} validation / {len(split["test"])} test',
        flush=True,
    )
    if args.check_data:
        for pair in pairs:
            image, mask = load_volume(pair, args.num_classes)
            print(
                Path(pair[0]).name,
                image.shape,
                np.unique(mask).tolist(),
                "prostate voxels:",
                np.count_nonzero(mask == 5),
                flush=True,
            )
        print("All data checks passed; no training started.")
        return
    cache_start = time.perf_counter()
    if args.cache_dir:
        prepare_cache(train_pairs + val_pairs, args.num_classes, args.cache_dir)
    cache_seconds = time.perf_counter() - cache_start
    print(f"Cache preparation: {cache_seconds:.2f}s (excluded from epoch timing)", flush=True)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; select --device cpu or use a GPU allocation"
        )
    model = ImprovedUNet3D(args.num_classes, args.base_channels, args.architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(args.seed)
    start_epoch, best = 0, -1.0
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        for key in (
            "num_classes",
            "base_channels",
            "architecture",
            "selection_class",
            "patch_size",
            "batch_size",
            "patches_per_volume",
            "workers",
            "no_amp",
        ):
            if config[key] != state["config"][key]:
                raise ValueError(f"Resume configuration differs for {key}; use original value")
        # Normalize tuples/lists after JSON or torch serialization.
        if json.dumps(split) != json.dumps(state["split"]):
            raise ValueError(
                "Resume split differs; use the original data root, seed, and validation fraction"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best = state["epoch"] + 1, state["best_dice"]
        # Restore all sampling streams, not just weights, for resumable training.
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        np_state = state["numpy_rng"]
        np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32), *np_state[2:]))
        generator.set_state(state["loader_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    elif (args.output_dir / "last.pt").exists() or (args.output_dir / "metrics.csv").exists():
        raise FileExistsError("Output contains an existing run; choose a new directory or --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (args.output_dir / "split.json").write_text(json.dumps(split, indent=2))
    dataset = PatchDataset(
        train_pairs,
        args.patch_size,
        args.num_classes,
        args.patches_per_volume,
        cache_dir=args.cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    print(
        f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; AMP: {amp}",
        flush=True,
    )
    print(
        f"Training samples: {len(dataset)}; batches per epoch: {len(loader)}; validation volumes: {len(val_pairs)}",
        flush=True,
    )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for epoch in range(start_epoch, args.epochs):
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()
        model.train()
        # Epoch loss includes only samples whose optimizer update succeeded.
        total_loss, samples = 0.0, 0
        skipped_steps, consecutive_skips = 0, 0
        data_seconds, compute_seconds, batches = 0.0, 0.0, 0
        waiting_since = time.perf_counter()
        for image, target in loader:
            data_seconds += time.perf_counter() - waiting_since
            compute_start = time.perf_counter()
            batches += 1
            image, target = image.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = deep_supervision_loss(model(image), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in epoch {epoch+1}")
            scaler.scale(loss).backward()
            if not optimizer_step(model, optimizer, scaler):
                skipped_steps += 1
                consecutive_skips += 1
                print(
                    f"AMP overflow: skipped update; loss scale reduced to {scaler.get_scale():g}",
                    flush=True,
                )
                if consecutive_skips >= 16:
                    raise FloatingPointError(
                        "16 consecutive AMP overflows; investigate data/model or retry a fresh run with --no-amp"
                    )
                if args.profile:
                    synchronize()
                compute_seconds += time.perf_counter() - compute_start
                waiting_since = time.perf_counter()
                continue
            consecutive_skips = 0
            total_loss += loss.item() * image.shape[0]
            samples += image.shape[0]
            # CUDA is asynchronous; profiling synchronizes at batch boundaries.
            if args.profile:
                synchronize()
            compute_seconds += time.perf_counter() - compute_start
            waiting_since = time.perf_counter()
        synchronize()
        training_seconds = time.perf_counter() - epoch_start
        training_peak = (
            torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
        )
        if samples == 0:
            raise FloatingPointError("No successful updates this epoch; no checkpoint saved")
        print(f"Epoch {epoch+1}: {skipped_steps} AMP updates skipped", flush=True)
        # Unmeasured validation stays None in CSV rather than becoming a zero Dice.
        score, per_class, improved = None, [None] * args.num_classes, False
        validation_start = time.perf_counter()
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            _, per_class = validate(model, val_pairs, args, device)
            score = per_class[args.selection_class]
            if not np.isfinite(score):
                raise ValueError(
                    "Selected validation class is absent from both predictions and labels"
                )
            scheduler.step(score)
            improved = score > best
            best = max(best, score)
        synchronize()
        validation_seconds = time.perf_counter() - validation_start
        io_start = time.perf_counter()
        np_state = np.random.get_state()
        # Save last every epoch; best changes only on a higher validation score.
        state = dict(
            epoch=epoch,
            best_dice=best,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            config=config,
            split=split,
            torch_rng=torch.get_rng_state(),
            python_rng=random.getstate(),
            numpy_rng=(np_state[0], np_state[1].tolist(), *np_state[2:]),
            loader_rng=generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        )
        save_checkpoint(args.output_dir / "last.pt", state)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", state)
        metrics_path = args.output_dir / "metrics.csv"
        write_header = not metrics_path.exists()
        with metrics_path.open("a", newline="") as stream:
            writer = csv.writer(stream)
            if write_header:
                writer.writerow(
                    ["epoch", "train_loss", "selection_class_dice", "learning_rate"]
                    + [f"dice_{i}" for i in range(args.num_classes)]
                )
            writer.writerow(
                [
                    epoch + 1,
                    total_loss / samples,
                    score,
                    optimizer.param_groups[0]["lr"],
                    *per_class,
                ]
            )
        plot_training_curves(metrics_path)
        synchronize()
        io_seconds = time.perf_counter() - io_start
        performance = dict(
            epoch=epoch + 1,
            batches=batches,
            successful_samples=samples,
            skipped_updates=skipped_steps,
            synchronized_batches=args.profile,
            data_wait_seconds=data_seconds,
            train_compute_seconds=compute_seconds,
            training_seconds=training_seconds,
            mean_training_batch_seconds=training_seconds / batches,
            validation_seconds=validation_seconds,
            checkpoint_logging_seconds=io_seconds,
            epoch_seconds=time.perf_counter() - epoch_start,
            cache_setup_seconds=cache_seconds,
            training_peak_gpu_gib=training_peak,
            epoch_peak_gpu_gib=(
                torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
            ),
        )
        performance_path = args.output_dir / "performance.csv"
        new_file = not performance_path.exists()
        with performance_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(performance))
            if new_file:
                writer.writeheader()
            writer.writerow(performance)
        print("PERFORMANCE " + json.dumps(performance), flush=True)
        print(
            f"Epoch {epoch+1}/{args.epochs}: loss={total_loss/samples:.5f}, validation Dice={score}, best={best:.5f}",
            flush=True,
        )
    clear_volume_cache()


if __name__ == "__main__":
    main()
