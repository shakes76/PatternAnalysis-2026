"""
Train a model for ADNI AD vs NC classification.

One script trains every model (--model cnn or gfnet), so all models share the same
data, splits, training recipe, evaluation and logging. Settings are constants at
the top of this file; each run records the values it used in its config.json.

Recipe: AdamW, linear learning-rate warm-up then cosine decay (stepped every
batch), mixed precision (AMP) on GPU, gradient clipping, no label smoothing.
After each epoch the model is evaluated on the validation patients; the
checkpoint with the lowest validation loss is kept, and training stops early if
validation loss has not improved for PATIENCE epochs. The test set is never used.

Each run writes to runs/<model>_<timestamp>/:
    config.json                   all settings, seed, GPU, parameter count
    history.csv                   per-epoch losses, metrics, learning rate, time, peak VRAM
    best.pt                       best checkpoint (not committed to git)
    val_slice_predictions.csv     per-slice validation outputs at the best epoch
    val_patient_predictions.csv   per-patient validation outputs at the best epoch
    curves.png                    loss and accuracy curves
    summary.json                  best epoch and its metrics, timing, peak VRAM

Usage:
    python train.py --model cnn              # full training run
    python train.py --model gfnet
    python train.py --model gfnet --overfit  # smoke test: should reach ~100% accuracy
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

from dataset import (
    AUG_BRIGHTNESS,
    AUG_CONTRAST,
    AUG_ROTATION,
    AUG_SCALE,
    AUG_TRANSLATE,
    get_datasets,
    get_loaders,
)
from modules import BaselineCNN, GFNet, count_parameters
from utils import evaluate, plot_history, save_rows_csv, set_seed

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = "/home/groups/comp3710/ADNI"
RUNS_DIR = PROJECT_DIR / "runs"

# Settings
# Per model: maximum epochs, peak learning rate, warm-up length.
MODEL_CONFIG = {
    "cnn": {"epochs": 30, "lr": 3e-4, "warmup_epochs": 1},
    "gfnet": {"epochs": 50, "lr": 1e-4, "warmup_epochs": 5},
}

GFNET_SIZES = {
    "ti": {
        "embed_dim": 256,
        "depth": 12,
        "mlp_ratio": 4.0,
        "drop_path_rate": 0.1,
    },  # 7.1M params
    "small": {
        "embed_dim": 192,
        "depth": 6,
        "mlp_ratio": 4.0,
        "drop_path_rate": 0.1,
    },  # 2.1M params
}

GFNET_ARGS = GFNET_SIZES["ti"]  # GFNet size used for training
AUGMENT = (
    True  # training-time augmentation (settings in dataset.py; off in overfit mode)
)

# Shared by all models.
BATCH_SIZE = 64
WEIGHT_DECAY = 0.05
PATIENCE = 30  # stop after this many epochs without a lower validation loss
GRAD_CLIP = 1.0  # maximum gradient norm
SEED = 42
NUM_WORKERS = 1

# Overfit smoke test: train and validate on the same few slices, no early stopping.
OVERFIT_SLICES = 64
OVERFIT_BATCH_SIZE = 8
OVERFIT_EPOCHS = 30


def build_model(name):
    """Return a freshly initialised model by name ("cnn" or "gfnet")."""
    if name == "cnn":
        return BaselineCNN()
    if name == "gfnet":
        return GFNet(**GFNET_ARGS)
    raise ValueError(f"Unknown model: {name}")


def build_scheduler(optimiser, warmup_steps, total_steps):
    """Linear warm-up from ~0 to the peak LR, then cosine decay to 0. Stepped once per batch.

    Args:
        optimiser: the optimiser whose learning rate is scheduled.
        warmup_steps: number of batches of linear warm-up (0 disables warm-up).
        total_steps: total number of batches in the run.
    """
    if warmup_steps == 0:
        return CosineAnnealingLR(optimiser, T_max=total_steps)
    warmup = LinearLR(optimiser, start_factor=1e-3, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimiser, T_max=total_steps - warmup_steps)
    return SequentialLR(optimiser, [warmup, cosine], milestones=[warmup_steps])


def make_loaders(overfit):
    """Full train/val loaders, or one small fixed subset for the overfit smoke test.

    Returns:
        (train_loader, val_loader, (mean, std))
    """
    datasets, (mean, std) = get_datasets(
        DATA_ROOT, num_workers=NUM_WORKERS, augment=AUGMENT and not overfit
    )
    if not overfit:
        loaders = get_loaders(datasets, BATCH_SIZE, NUM_WORKERS)
        return loaders["train"], loaders["val"], (mean, std)

    # The same slices are used for training and validation: a working pipeline
    # should drive their loss towards zero and accuracy towards 1.
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(datasets["train"]), generator=g)[:OVERFIT_SLICES].tolist()
    subset = Subset(datasets["train"], idx)
    train = DataLoader(subset, batch_size=OVERFIT_BATCH_SIZE, shuffle=True)
    val = DataLoader(subset, batch_size=OVERFIT_BATCH_SIZE, shuffle=False)
    return train, val, (mean, std)


def train_one_epoch(model, loader, optimiser, scheduler, scaler, loss_fn, device):
    """One pass over the training data. Returns mean loss and slice accuracy.

    Uses mixed precision on GPU. Gradients are unscaled before clipping so the
    norm limit applies to the true gradients, and the learning-rate scheduler
    is stepped after every batch.
    """
    model.train()
    use_amp = device.type == "cuda"
    total_loss, correct, n = 0.0, 0, 0
    for x, y, _, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)  # clip the true gradients, not the scaled ones
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


def main():
    """Train one model, logging every epoch and saving the best checkpoint and outputs."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model", choices=["cnn", "gfnet"], required=True, help="architecture to train"
    )
    ap.add_argument(
        "--overfit", action="store_true", help="smoke test on a few training slices"
    )
    args = ap.parse_args()

    cfg = MODEL_CONFIG[args.model]
    epochs = OVERFIT_EPOCHS if args.overfit else cfg["epochs"]
    patience = epochs if args.overfit else PATIENCE  # overfit mode never stops early

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "none (CPU)"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{args.model}_{'overfit_' if args.overfit else ''}{stamp}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, (mean, std) = make_loaders(args.overfit)
    model = build_model(args.model).to(device)
    n_params = count_parameters(model)

    config = {
        "run": name,
        "model": args.model,
        "model_args": GFNET_ARGS if args.model == "gfnet" else {},
        "overfit": args.overfit,
        "epochs": epochs,
        "lr": cfg["lr"],
        "warmup_epochs": cfg["warmup_epochs"],
        "batch_size": OVERFIT_BATCH_SIZE if args.overfit else BATCH_SIZE,
        "overfit_slices": OVERFIT_SLICES if args.overfit else None,
        "weight_decay": WEIGHT_DECAY,
        "patience": patience,
        "grad_clip": GRAD_CLIP,
        "seed": SEED,
        "amp": device.type == "cuda",
        "augment": AUGMENT and not args.overfit,
        "augment_settings": {
            "rotation": AUG_ROTATION,
            "translate": AUG_TRANSLATE,
            "scale": AUG_SCALE,
            "brightness": AUG_BRIGHTNESS,
            "contrast": AUG_CONTRAST,
        }
        if AUGMENT and not args.overfit
        else None,
        "label_smoothing": 0.0,
        "parameters": n_params,
        "norm_mean": mean,
        "norm_std": std,
        "device": str(device),
        "gpu": gpu,
        "torch": torch.__version__,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Run {name} | device {gpu} | {args.model}: {n_params:,} parameters")

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY
    )
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(
        optimiser, cfg["warmup_epochs"] * steps_per_epoch, epochs * steps_per_epoch
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history, best_loss, best_epoch, bad_epochs = [], float("inf"), 0, 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        lr = optimiser.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimiser, scheduler, scaler, loss_fn, device
        )
        val = evaluate(model, val_loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0
        peak_gb = (
            torch.cuda.max_memory_allocated() / 1e9
            if device.type == "cuda"
            else float("nan")
        )

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
        }
        row.update({f"val_slice_{k}": v for k, v in val["slice"].items()})
        row.update({f"val_patient_{k}": v for k, v in val["patient"].items()})
        row.update({"epoch_time_s": epoch_time, "peak_vram_gb": peak_gb})
        history.append(row)
        save_rows_csv(history, run_dir / "history.csv")

        improved = val["loss"] < best_loss
        if improved:
            best_loss, best_epoch, bad_epochs = val["loss"], epoch, 0
            torch.save(
                {
                    "model": args.model,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "config": config,
                    "val_loss": val["loss"],
                },
                run_dir / "best.pt",
            )
            save_rows_csv(val["slice_rows"], run_dir / "val_slice_predictions.csv")
            save_rows_csv(val["patient_rows"], run_dir / "val_patient_predictions.csv")
        else:
            bad_epochs += 1

        vram = f"{peak_gb:.2f}GB" if device.type == "cuda" else "n/a"
        print(
            f"epoch {epoch:3d} | lr {lr:.2e} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val['loss']:.4f} slice acc {val['slice']['acc']:.3f} "
            f"patient acc {val['patient']['acc']:.3f} auroc {val['patient']['auroc']:.3f} | "
            f"{epoch_time:.0f}s {vram}{' *' if improved else ''}"
        )  # * = new best checkpoint

        if bad_epochs >= patience:
            print(f"Early stop: no lower validation loss for {patience} epochs.")
            break

    total_time = time.perf_counter() - start
    plot_history(history, run_dir / "curves.png", title=f"{args.model} ({name})")
    best = history[best_epoch - 1]
    summary = {
        "run": name,
        "model": args.model,
        "gpu": gpu,
        "parameters": n_params,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "total_train_time_s": total_time,
        "time_to_best_epoch_s": sum(h["epoch_time_s"] for h in history[:best_epoch]),
        "mean_epoch_time_s": total_time / len(history),
        "peak_train_vram_gb": max(h["peak_vram_gb"] for h in history),
        "best": best,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"\nBest epoch {best_epoch}: val loss {best['val_loss']:.4f}, "
        f"slice acc {best['val_slice_acc']:.3f}, patient acc {best['val_patient_acc']:.3f}, "
        f"patient AUROC {best['val_patient_auroc']:.3f}"
    )
    print(f"Total time {total_time / 60:.1f} min | outputs in {run_dir}")


if __name__ == "__main__":
    main()
