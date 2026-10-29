"""
train.py - train a 3D U-Net on HipMRI, validate every epoch, keep the best model.

Author: Angus Slater (49615878)
Course: COMP3710 Pattern Recognition, The University of Queensland, 2026
Project: Improved 3D U-Net for prostate MRI segmentation (HipMRI)

Usage (on Rangpur, through train.sh):
    python train.py --model unet2d --epochs 150 --tag v1             # 2D control
    python train.py --model unet3d --epochs 150 --tag baseline       # 3D baseline
    python train.py --model improved --epochs 150 --tag v1           # Improved 3D U-Net
    python train.py --model improved --epochs 3 --limit 4 --tag smoke  # quick check

Outputs, named <model>_<tag>:
    checkpoints/<run>.pt      best weights by validation prostate Dice
    results/<run>.json        settings, per-epoch history, best epoch, time, peak VRAM
    figures/<run>_curves.png  loss and validation Dice curves

Testing is deliberately not done here: predict.py evaluates the held-out test set once,
after model selection on validation, so test data never influences training or the choice
of checkpoint.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from dataset import NUM_CLASSES, ROOT, get_loaders
from modules import DiceCELoss, ImprovedUNet3D, UNet3D, UNet2D
from utils import CLASS_NAMES, dice_per_class, plot_history

MODELS = {"unet3d": UNet3D, "improved": ImprovedUNet3D, "unet2d": UNet2D}  # --model choices
PROSTATE = 5 # label index of the target class


def parse_args():
    """Command-line settings, so every experiment is one reproducible command."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(MODELS), default="unet3d")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--data_dir", default=ROOT, help="folder containing semantic_MRs/")
    p.add_argument("--limit", type=int, default=None, help="cap volumes per split (smoke tests)")
    p.add_argument("--augment", action="store_true", help="random left-right flips on train")
    p.add_argument("--num_workers", type=int, default=4, help="match --cpus-per-task")
    p.add_argument("--seed", type=int, default=42, help="training randomness (init, batch order)")
    p.add_argument("--split_seed", type=int, default=42,
                   help="patient split seed; keep fixed so every run uses the same patients")
    p.add_argument("--tag", default="baseline", help="run name suffix")
    return p.parse_args()


def set_seed(seed):
    """Seed Python, NumPy and PyTorch so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, loss_fn, opt, device):
    """One pass over the training set with bf16 autocast.

    Args:
        model: network to train (put in train mode here).
        loader: training DataLoader yielding (image, mask) batches.
        loss_fn: DiceCELoss.
        opt: optimiser (Adam).
        device: torch.device to run on.

    Returns:
        Dict of mean "loss" (total), "ce" and "dice" (= 1 - soft Dice) over the epoch's batches.
    """
    model.train()
    sums = {"loss": 0.0, "ce": 0.0, "dice": 0.0}  # "dice" = 1 - soft Dice (a loss, lower is better)
    for img, lab in loader:
        img, lab = img.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(img)
        loss = loss_fn(logits, lab)
        loss.backward()
        opt.step()
        sums["loss"] += loss.item()
        sums["ce"] += loss_fn.last["ce"]
        sums["dice"] += loss_fn.last["dice"]
    return {k: v / len(loader) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Mean loss and mean per-class hard Dice over a split (batch size 1, one volume each).

    Args:
        model: network to evaluate (put in eval mode here).
        loader: validation (or test) DataLoader.
        loss_fn: DiceCELoss, for the validation loss curve.
        device: torch.device to run on.

    Returns:
        (losses, dice): losses is a dict of mean "loss", "ce" and "dice" (= 1 - soft Dice);
        dice is a list of 6 per-class hard Dice scores, averaged over volumes.
    """
    model.eval()
    sums = {"loss": 0.0, "ce": 0.0, "dice": 0.0}
    dice_sum = torch.zeros(NUM_CLASSES)
    for img, lab in loader:
        img, lab = img.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(img)
        sums["loss"] += loss_fn(logits, lab).item()
        sums["ce"] += loss_fn.last["ce"]
        sums["dice"] += loss_fn.last["dice"]
        dice_sum += dice_per_class(logits.float(), lab).cpu()
    n = len(loader)
    return {k: v / n for k, v in sums.items()}, (dice_sum / n).tolist()


def main():
    """Load data, build the model, train with per-epoch validation, save best checkpoint + logs."""
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = f"{args.model}_{args.tag}"
    for d in ("checkpoints", "results", "figures"):
        os.makedirs(d, exist_ok=True)
    print(f"run {run} on {device} | {vars(args)}", flush=True)

    t0 = time.time()
    loaders, _ = get_loaders(root=args.data_dir, batch_size=args.batch_size, seed=args.split_seed,
                             num_workers=args.num_workers, augment=args.augment, limit=args.limit)
    sizes = ", ".join(f"{k} {len(v.dataset)}" for k, v in loaders.items())
    print(f"data loaded in {time.time() - t0:.0f}s ({sizes})", flush=True)

    model = MODELS[args.model]().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = DiceCELoss(NUM_CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam: standard for U-Nets, little tuning needed
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)  # lr decays smoothly to ~0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = {"train": [], "val": [], "val_dice": [], "epoch_time": []}
    best = {"dice": -1.0, "epoch": 0}
    t_train = time.time()
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        tr = train_one_epoch(model, loaders["train"], loss_fn, opt, device)
        va, va_dice = evaluate(model, loaders["val"], loss_fn, device)
        sched.step()

        history["train"].append(tr)
        history["val"].append(va)
        history["val_dice"].append(va_dice)
        history["epoch_time"].append(time.time() - t_ep)

        # Model selection on VALIDATION prostate Dice; the test set is untouched until predict.py.
        if va_dice[PROSTATE] > best["dice"]:
            best = {"dice": va_dice[PROSTATE], "epoch": epoch}
            torch.save(model.state_dict(), f"checkpoints/{run}.pt")

        dice_str = " ".join(f"{n[:4]} {d:.3f}" for n, d in zip(CLASS_NAMES[1:], va_dice[1:]))
        print(f"ep {epoch:3d} | train {tr['loss']:.4f} (ce {tr['ce']:.4f} dice {tr['dice']:.4f}) "
              f"| val {va['loss']:.4f} (ce {va['ce']:.4f} dice {va['dice']:.4f}) | {dice_str} "
              f"| {history['epoch_time'][-1]:.0f}s", flush=True)

        # Save results and curves every epoch, so a killed job still leaves evidence.
        summary = {
            "run": run, "args": vars(args), "params": n_params,
            "best_val_prostate_dice": best["dice"], "best_epoch": best["epoch"],
            "train_minutes": (time.time() - t_train) / 60,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
            "history": history,
        }
        with open(f"results/{run}.json", "w") as f:
            json.dump(summary, f, indent=1)
        plot_history(history, f"figures/{run}_curves.png")

    print(f"done: best val prostate Dice {best['dice']:.3f} at epoch {best['epoch']} | "
          f"{summary['train_minutes']:.1f} min | peak VRAM {summary['peak_vram_gb']:.2f} GB | "
          f"{n_params:,} params", flush=True)


if __name__ == "__main__":
    main()