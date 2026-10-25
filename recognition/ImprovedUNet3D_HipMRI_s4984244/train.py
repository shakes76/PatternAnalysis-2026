"""
Train, validate, test and save a segmentation model on the HipMRI 3D data.

    python train.py --model unet3d            # 3D Improved U-Net
    python train.py --model improved2d        # same network in 2D (context control)
    python train.py --model unet2d            # standard 2D U-Net baseline

All models are validated and tested on whole volumes of the same patients,
so their numbers are directly comparable. Outputs go to ``<out>/<model>/``:

    best.pt / last.pt     checkpoints (best = highest mean foreground val Dice)
    history.json          per-epoch loss, validation Dice and timing
    curves.png            loss and validation Dice curves
    test_metrics.json     per-volume and summary test Dice / IoU, resources
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import (CLASS_NAMES, DEFAULT_ROOT, NUM_CLASSES, TARGET_SHAPE,
                     HipMRISlices, HipMRIVolumes, load_split_scans)
from modules import DiceCELoss, count_parameters
from utils import build_model, dice_iou, predict_volume

# Per-model defaults, overridable from the command line.
DEFAULTS = {
    "unet3d": {"epochs": 300, "batch_size": 2, "lr": 5e-4},
    "improved2d": {"epochs": 50, "batch_size": 32, "lr": 5e-4},
    "unet2d": {"epochs": 50, "batch_size": 32, "lr": 1e-3},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=DEFAULTS, default="unet3d")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="HipMRI_Study_open directory")
    parser.add_argument("--shape", default="x".join(map(str, TARGET_SHAPE)),
                        help="volume size D x H x W after resampling")
    parser.add_argument("--cache", default="cache",
                        help="directory for the pre-processed volumes")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--limit", type=int,
                        help="use only this many train/val/test scans (smoke tests)")
    args = parser.parse_args()
    for key, value in DEFAULTS[args.model].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.shape = tuple(int(v) for v in args.shape.split("x"))
    return args


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, amp):
    model.train()
    total, count = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = loss_fn(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * images.shape[0]
        count += images.shape[0]
    return total / count


@torch.no_grad()
def evaluate(model, model_name, dataset, device, amp):
    """Per-volume Dice and IoU, each of shape (num_volumes, num_classes)."""
    model.eval()
    dices, ious = [], []
    for image, label in dataset:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            probs = predict_volume(model, image.to(device), model_name)
        dice, iou = dice_iou(probs.argmax(dim=0).cpu(), label, NUM_CLASSES)
        dices.append(dice)
        ious.append(iou)
    return torch.stack(dices), torch.stack(ious)


def plot_history(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_loss.plot(epochs, [h["train_loss"] for h in history])
    ax_loss.set(xlabel="epoch", ylabel="Dice + CE loss", title="Training loss")
    for c, name in enumerate(CLASS_NAMES):
        ax_dice.plot(epochs, [h["val_dice"][c] for h in history], label=name)
    ax_dice.set(xlabel="epoch", ylabel="Dice", ylim=(0, 1), title="Validation Dice")
    ax_dice.legend(loc="lower right")
    for ax in (ax_loss, ax_dice):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    scans, names = load_split_scans(args.root, args.cache, args.shape)
    if args.limit:
        names = {split: n[:args.limit] for split, n in names.items()}
    for split in ("train", "val", "test"):
        print(f"{split:<5}: {len(names[split])} scans")

    TrainSet = HipMRIVolumes if args.model == "unet3d" else HipMRISlices
    loader = DataLoader(TrainSet(scans, names["train"], train=True),
                        batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    val_set = HipMRIVolumes(scans, names["val"])
    test_set = HipMRIVolumes(scans, names["test"])

    model = build_model(args.model, NUM_CLASSES).to(device)
    print(f"{args.model}: {count_parameters(model):,} trainable parameters on {device}")
    loss_fn = DiceCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    def save(path, epoch, val_dice):
        torch.save({"model": args.model, "num_classes": NUM_CLASSES,
                    "state_dict": model.state_dict(), "epoch": epoch,
                    "val_dice": val_dice, "args": vars(args)}, path)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best, start = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        tic = time.time()
        train_loss = train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, amp)
        scheduler.step()
        val_dice = evaluate(model, args.model, val_set, device, amp)[0].nanmean(dim=0)
        foreground = val_dice[1:].mean().item()

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_dice": val_dice.tolist(), "seconds": time.time() - tic})
        if foreground > best:
            best = foreground
            save(out_dir / "best.pt", epoch, val_dice.tolist())
        save(out_dir / "last.pt", epoch, val_dice.tolist())

        per_class = " ".join(f"{n[:4]}={d:.3f}" for n, d in zip(CLASS_NAMES[1:], val_dice[1:]))
        print(f"epoch {epoch:3d}/{args.epochs} loss={train_loss:.4f} "
              f"val fg={foreground:.3f} [{per_class}] {time.time() - tic:.0f}s", flush=True)

    train_hours = (time.time() - start) / 3600
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    plot_history(history, out_dir / "curves.png")

    # Test the best checkpoint on the held-out patients.
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["state_dict"])
    dice, iou = evaluate(model, args.model, test_set, device, amp)
    print(f"\nTest ({len(test_set)} scans, best epoch by val Dice):")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:<10} Dice {dice[:, c].nanmean():.3f} +/- {dice[:, c].std():.3f}"
              f"   IoU {iou[:, c].nanmean():.3f}")

    results = {
        "model": args.model,
        "parameters": count_parameters(model),
        "train_hours": train_hours,
        "peak_train_vram_gb": peak_gb,
        "test_scans": names["test"],
        "dice": dice.tolist(),
        "iou": iou.tolist(),
        "mean_dice": dice.nanmean(dim=0).tolist(),
        "mean_iou": iou.nanmean(dim=0).tolist(),
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(results, indent=1))
    print(f"Saved checkpoints, curves and metrics to {out_dir}")


if __name__ == "__main__":
    main()
