"""Fine-tune Stable Diffusion 1.5 on patient-separated ADNI training slices.

Run inside a Rangpur GPU allocation (tools/e1_train.sbatch does this):

    python train.py --output outputs/e1-<job> --epochs 10 --batch-size 16

Only the training and validation rows of data/manifest-seed3710.csv are read;
the project test split stays sealed. The run directory receives:

    history.csv    windowed training loss, learning rate, gradient norm, throughput
    epochs.csv     per epoch: mean training loss, validation loss (live and EMA)
    curves.png     both curves, redrawn after every epoch
    samples/       a fixed-seed AD/NC grid before training and after each epoch
    weights/       float16 EMA weights at the requested epochs
    metrics.json   configuration, provenance hashes, timings and peak GPU memory
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from dataset import ADNISliceDataset, defective_scans, foreground_share
from modules import (
    LABELS,
    ExponentialMovingAverage,
    build_unet,
    condition_batch,
    configure_trainable,
    decode,
    diffusion_loss,
    encode_prompts,
    guided_ddim,
    initial_noise,
    latents_from_moments,
    load_noise_schedule,
    load_vae,
    save_weights,
    training_noise,
)

PROJECT = Path(__file__).resolve().parent
# Same seeds as the per-stage grids of the E0 LoRA run, so grids line up.
SAMPLE_SEEDS = (37100, 37101, 37102, 37103)


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 1e-2
    offset_noise: float = 0.1
    cond_dropout: float = 0.1
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    max_steps: int | None = None
    save_epochs: tuple[int, ...] = (5, 10)
    seed: int = 3710
    # Validation slices, noise and timesteps; fixed across training seeds so
    # replicate runs (docs/r1-seed-replicates-protocol.md) stay comparable.
    val_seed: int = 3710
    log_every: int = 50
    time_budget_minutes: float | None = None
    guidance: float = 7.5


@dataclass
class EncodedSplit:
    """VAE posterior moments of every slice in a split, held on the device."""

    mean: torch.Tensor
    std: torch.Tensor
    label_index: torch.Tensor
    foreground: torch.Tensor | None = None  # share of bright pixels per slice

    def __len__(self) -> int:
        return self.mean.shape[0]

    def select(self, keep: torch.Tensor) -> "EncodedSplit":
        """Rows where keep is True, in their original order."""
        foreground = None if self.foreground is None else self.foreground[keep.cpu()]
        return EncodedSplit(
            self.mean[keep], self.std[keep], self.label_index[keep], foreground
        )


@torch.no_grad()
def encode_split(dataset, vae, device, batch_size=64, num_workers=2) -> EncodedSplit:
    """Run the frozen VAE encoder once per slice instead of once per step."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    means, stds, labels, shares = [], [], [], []
    for batch in loader:
        posterior = vae.encode(batch["pixel_values"].to(device)).latent_dist
        means.append(posterior.mean)
        stds.append(posterior.std)
        labels.extend(LABELS.index(name) for name in batch["class_name"])
        shares.append(foreground_share(batch["pixel_values"]))
    return EncodedSplit(
        torch.cat(means),
        torch.cat(stds),
        torch.tensor(labels, device=device),
        torch.cat(shares),
    )


def dataset_rows(dataset) -> list[dict]:
    """Manifest rows in loader order, also through a Subset."""
    if isinstance(dataset, Subset):
        return [dataset.dataset.rows[i] for i in dataset.indices]
    return dataset.rows


def exclude_defective_scans(split: EncodedSplit, rows: list[dict]):
    """Drop every scan that dataset.defective_scans flags (the E2 change).

    Returns the reduced split and the flagged scan IDs.
    """
    flagged = defective_scans(rows, split.foreground.tolist())
    keep = torch.tensor(
        [row["scan_id"] not in flagged for row in rows], device=split.mean.device
    )
    return split.select(keep), flagged


@torch.no_grad()
def validation_loss(unet, split, embeddings, schedule, scaling, seed, batch_size=32):
    """Noise-prediction loss on fixed validation latents, noise and timesteps.

    The draws are identical every epoch, so changes come from the weights alone.
    It uses plain Gaussian noise and the true class prompt, the standard
    objective without training-time offset noise or dropout, so runs trained
    with different recipes report comparable numbers.
    """
    was_training = unet.training
    unet.eval()
    generator = torch.Generator(device=split.mean.device).manual_seed(seed)
    total = 0.0
    for start in range(0, len(split), batch_size):
        part = slice(start, start + batch_size)
        latents = latents_from_moments(
            split.mean[part], split.std[part], scaling, generator
        )
        noise = training_noise(latents, 0.0, generator)
        timesteps = torch.randint(
            0,
            schedule.config.num_train_timesteps,
            (latents.shape[0],),
            generator=generator,
            device=latents.device,
        )
        conditioning = condition_batch(split.label_index[part], embeddings, 0.0)
        loss = diffusion_loss(unet, latents, conditioning, schedule, noise, timesteps)
        total += loss.item() * latents.shape[0]
    unet.train(was_training)
    return total / len(split)


@torch.no_grad()
def sample_grid(unet, vae, embeddings, schedule, guidance, path, size=256):
    """Draw the fixed AD/NC seeds with guided DDIM and save a labelled grid."""
    was_training = unet.training
    unet.eval()
    device = next(unet.parameters()).device
    grid = Image.new(
        "RGB", (len(SAMPLE_SEEDS) * size, len(LABELS) * (size + 24)), "white"
    )
    draw = ImageDraw.Draw(grid)
    for row, label in enumerate(LABELS):
        for col, seed in enumerate(SAMPLE_SEEDS):
            latent = guided_ddim(
                unet,
                schedule.config,
                initial_noise(seed, size, device),
                embeddings[label],
                embeddings["empty"],
                guidance,
            )
            pixels = decode(vae, latent)[0].mul(255).round().byte()
            tile = Image.frombytes(
                "RGB",
                (size, size),
                bytes(pixels.permute(1, 2, 0).cpu().flatten().tolist()),
            )
            grid.paste(tile, (col * size, row * (size + 24)))
            draw.text(
                (col * size + 4, row * (size + 24) + size + 4),
                f"{label} seed {seed} cfg {guidance}",
                fill="black",
            )
    unet.train(was_training)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def plot_curves(history, epochs, path):
    """Training loss per log window and validation loss per epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot([r["step"] for r in history], [r["loss"] for r in history], lw=0.8)
    left.set(
        xlabel="optimiser step",
        ylabel="noise-prediction MSE",
        title="Training loss (window mean)",
    )
    if epochs:
        numbers = [r["epoch"] for r in epochs]
        right.plot(
            numbers, [r["train_loss"] for r in epochs], marker="o", label="training"
        )
        right.plot(
            numbers,
            [r["val_loss_live"] for r in epochs],
            marker="o",
            label="validation, live",
        )
        right.plot(
            numbers,
            [r["val_loss_ema"] for r in epochs],
            marker="o",
            label="validation, EMA",
        )
        right.legend()
    right.set(xlabel="epoch", ylabel="noise-prediction MSE", title="Per-epoch loss")
    for axis in (left, right):
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit(
    config: TrainConfig,
    unet,
    parameters,
    schedule,
    embeddings,
    train_split: EncodedSplit,
    val_split: EncodedSplit,
    scaling: float,
    output: Path,
    weights_dir: Path,
    weight_prefix: str = "unet",
    sample_fn=None,
) -> dict:
    """Train with AdamW, linear warm-up, gradient clipping and an EMA of weights."""
    device = train_split.mean.device
    steps_per_epoch = len(train_split) // config.batch_size
    if steps_per_epoch == 0:
        raise ValueError("Batch size exceeds the number of training slices")
    optimizer = torch.optim.AdamW(
        parameters.values(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    warmup = max(1, config.warmup_steps)
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup)
    )
    ema = ExponentialMovingAverage(parameters, config.ema_decay)
    # One generator drives noise, timesteps and prompt dropout; a CPU one the
    # epoch order. Both are seeded, so a rerun draws the same batches.
    generator = torch.Generator(device=device).manual_seed(config.seed)
    order_generator = torch.Generator().manual_seed(config.seed)
    history, epochs, saved = [], [], []
    step, stop_reason, train_seconds = 0, "completed", 0.0
    started = time.perf_counter()
    if sample_fn:
        sample_fn(0)
    unet.train()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        order = torch.randperm(len(train_split), generator=order_generator).to(device)
        window_loss = torch.zeros((), device=device)
        window_norm = torch.zeros((), device=device)
        epoch_loss = torch.zeros((), device=device)
        window_steps = epoch_steps = 0
        window_started = time.perf_counter()
        for batch_number in range(steps_per_epoch):
            index = order[
                batch_number * config.batch_size : (batch_number + 1)
                * config.batch_size
            ]
            latents = latents_from_moments(
                train_split.mean[index], train_split.std[index], scaling, generator
            )
            noise = training_noise(latents, config.offset_noise, generator)
            conditioning = condition_batch(
                train_split.label_index[index],
                embeddings,
                config.cond_dropout,
                generator,
            )
            timesteps = torch.randint(
                0,
                schedule.config.num_train_timesteps,
                (latents.shape[0],),
                generator=generator,
                device=device,
            )
            loss = diffusion_loss(
                unet, latents, conditioning, schedule, noise, timesteps
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                parameters.values(), config.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            lr_schedule.step()
            ema.update(parameters)
            step += 1
            window_loss += loss.detach()
            window_norm += norm.detach()
            epoch_loss += loss.detach()
            window_steps += 1
            epoch_steps += 1
            stopping = bool(config.max_steps) and step >= config.max_steps
            if (
                step % config.log_every == 0
                or batch_number == steps_per_epoch - 1
                or stopping
            ):
                mean_loss = (window_loss / window_steps).item()
                if not math.isfinite(mean_loss):
                    raise RuntimeError(f"Non-finite training loss by step {step}")
                elapsed = time.perf_counter() - window_started
                history.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "loss": mean_loss,
                        "grad_norm": (window_norm / window_steps).item(),
                        "lr": lr_schedule.get_last_lr()[0],
                        "images_per_second": window_steps * config.batch_size / elapsed,
                    }
                )
                print(json.dumps(history[-1]), flush=True)
                train_seconds += elapsed
                window_loss.zero_()
                window_norm.zero_()
                window_steps = 0
                window_started = time.perf_counter()
            if stopping:
                stop_reason = "max_steps"
                break
        if window_steps:
            train_seconds += time.perf_counter() - window_started
        val_live = validation_loss(
            unet, val_split, embeddings, schedule, scaling, config.val_seed
        )
        with ema.applied_to(parameters):
            val_ema = validation_loss(
                unet, val_split, embeddings, schedule, scaling, config.val_seed
            )
            if sample_fn:
                sample_fn(epoch)
        epoch_seconds = time.perf_counter() - epoch_started
        epochs.append(
            {
                "epoch": epoch,
                "steps": step,
                "train_loss": (epoch_loss / max(1, epoch_steps)).item(),
                "val_loss_live": val_live,
                "val_loss_ema": val_ema,
                "seconds": epoch_seconds,
            }
        )
        print(json.dumps(epochs[-1]), flush=True)
        write_rows(output / "history.csv", history)
        write_rows(output / "epochs.csv", epochs)
        plot_curves(history, epochs, output / "curves.png")
        elapsed_minutes = (time.perf_counter() - started) / 60
        if (
            stop_reason == "completed"
            and config.time_budget_minutes
            and epoch < config.epochs
            and elapsed_minutes + epoch_seconds / 60 > config.time_budget_minutes
        ):
            stop_reason = "time_budget"
        last = epoch == config.epochs or stop_reason != "completed"
        if epoch in config.save_epochs or last:
            path = weights_dir / f"{weight_prefix}-ema-epoch-{epoch:02d}.safetensors"
            save_weights(ema.shadow, path)
            saved.append(str(path))
        if last:
            break
    return {
        "stop_reason": stop_reason,
        "steps": step,
        "epochs_completed": len(epochs),
        "steps_per_epoch": steps_per_epoch,
        "train_seconds": train_seconds,
        "wall_seconds": time.perf_counter() - started,
        "images_per_second": step * config.batch_size / train_seconds
        if train_seconds
        else None,
        "final_val_loss_ema": epochs[-1]["val_loss_ema"],
        "weights": saved,
    }


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--weights-dir", type=Path, help="default: OUTPUT/weights")
    parser.add_argument("--trainable", choices=("unet", "lora"), default="unet")
    parser.add_argument(
        "--unet-init",
        choices=("pretrained", "scratch"),
        default="pretrained",
        help="scratch: the smaller B3 UNet from random weights",
    )
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--offset-noise", type=float, default=0.1)
    parser.add_argument("--cond-dropout", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--max-steps", type=int, help="stop early, e.g. for a probe")
    parser.add_argument("--save-epochs", type=int, nargs="*", default=[5, 10])
    parser.add_argument("--time-budget-minutes", type=float)
    parser.add_argument(
        "--train-limit", type=int, help="use only the first N training slices"
    )
    parser.add_argument("--val-images", type=int, default=1024)
    parser.add_argument(
        "--exclude-defective-scans",
        action="store_true",
        help="drop training scans with any near-empty slice (docs/e2-protocol.md)",
    )
    parser.add_argument(
        "--expect-excluded",
        type=int,
        nargs=2,
        metavar=("SCANS", "SLICES"),
        help="stop before training unless exactly this many are excluded",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3710, help="training randomness")
    parser.add_argument(
        "--val-seed",
        type=int,
        default=3710,
        help="validation slices and their fixed noise; kept constant across replicates",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/home/groups/comp3710/ADNI")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifest-seed3710.csv")
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/sd15-smoke"))
    parser.add_argument("--vae-dir", type=Path, default=Path("models/sd15-vae"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        offset_noise=args.offset_noise,
        cond_dropout=args.cond_dropout,
        ema_decay=args.ema_decay,
        max_steps=args.max_steps,
        save_epochs=tuple(args.save_epochs),
        seed=args.seed,
        val_seed=args.val_seed,
        time_budget_minutes=args.time_budget_minutes,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        # Throughput settings. Unlike the E0 diagnostics this run is not bitwise
        # deterministic; it is seeded, and the settings are recorded below.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(config.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    embeddings = encode_prompts(args.model_dir, device)
    vae = load_vae(args.vae_dir, device)
    train_set = ADNISliceDataset(
        args.data_root, args.manifest, "train", 256, verify_hash=True
    )
    if args.train_limit:
        train_set = Subset(train_set, range(args.train_limit))
    val_full = ADNISliceDataset(
        args.data_root, args.manifest, "validation", 256, verify_hash=True
    )
    chosen = torch.randperm(
        len(val_full), generator=torch.Generator().manual_seed(config.val_seed)
    )
    val_set = Subset(val_full, sorted(chosen[: args.val_images].tolist()))
    train_split = encode_split(train_set, vae, device, num_workers=args.num_workers)
    val_split = encode_split(val_set, vae, device, num_workers=args.num_workers)
    encode_seconds = time.perf_counter() - started
    excluded = {"scans": 0, "slices": 0}
    if args.exclude_defective_scans:
        before = len(train_split)
        train_split, flagged = exclude_defective_scans(
            train_split, dataset_rows(train_set)
        )
        excluded = {"scans": len(flagged), "slices": before - len(train_split)}
        print(json.dumps({"excluded": excluded}), flush=True)
        # Scan IDs identify patients' data, so they stay in a private file.
        (args.output / "private-excluded-scans.json").write_text(
            json.dumps(sorted(flagged)) + "\n"
        )
        expected = args.expect_excluded
        if expected and [excluded["scans"], excluded["slices"]] != expected:
            raise SystemExit(
                f"Excluded {excluded}, protocol expects {expected}; stopping"
            )

    unet = build_unet(args.model_dir, device, args.unet_init)
    parameters = configure_trainable(unet, args.trainable, args.lora_rank)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    schedule = load_noise_schedule(args.model_dir)
    setup_seconds = time.perf_counter() - started

    def sample_fn(epoch):
        sample_grid(
            unet,
            vae,
            embeddings,
            schedule,
            config.guidance,
            args.output / "samples" / f"epoch-{epoch:02d}.png",
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    summary = fit(
        config,
        unet,
        parameters,
        schedule,
        embeddings,
        train_split,
        val_split,
        vae.config.scaling_factor,
        args.output,
        args.weights_dir or args.output / "weights",
        weight_prefix=args.trainable,
        sample_fn=sample_fn,
    )
    metrics = {
        "config": asdict(config),
        "trainable": args.trainable,
        "unet_init": args.unet_init,
        "lora_rank": args.lora_rank if args.trainable == "lora" else None,
        "trainable_parameters": sum(p.numel() for p in parameters.values()),
        "unet_parameters": sum(p.numel() for p in unet.parameters()),
        "train_images": len(train_split),
        "exclude_defective_scans": args.exclude_defective_scans,
        "excluded_scans": excluded["scans"],
        "excluded_slices": excluded["slices"],
        "val_images": len(val_split),
        "encode_seconds": encode_seconds,
        "setup_seconds": setup_seconds,
        **summary,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
        if device.type == "cuda"
        else None,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "torch": torch.__version__,
        "tf32_and_cudnn_benchmark": device.type == "cuda",
        "test_split_read": False,
        "provenance_sha256": {
            "model": digest(args.model_dir / "provenance.json"),
            "vae": digest(args.vae_dir / "provenance.json"),
            "manifest": digest(args.manifest),
            **{
                name: digest(PROJECT / name)
                for name in ("modules.py", "dataset.py", "train.py")
            },
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in (
                    "stop_reason",
                    "steps",
                    "images_per_second",
                    "peak_allocated_gib",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
