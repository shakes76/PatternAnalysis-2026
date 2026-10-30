"""
train.py - Train, validate, test and save either model on HipMRI 3D.

Usage (on Rangpur, see slurm/ for batch scripts):
    python train.py --model 3d          # proposed: 3D Improved U-Net
    python train.py --model 2d          # baseline: 2D U-Net, slice-wise

Both models use the same preprocessed volumes and the same patient-level
split (dataset.patient_split, fixed seed), and are tested with the same
volumetric metrics, so their results are directly comparable.

Outputs in runs/<model>/:
    best.pt          weights with the best validation mean foreground Dice
    history.json     per-epoch train loss and validation Dice
    curves.png       loss and validation Dice plots
    test_metrics.json  per-class test Dice/IoU, extremity Dice, resources
    split.json       the exact patients/cases in each split
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import (CLASS_NAMES, DEFAULT_CACHE, DEFAULT_ROOT, Slice2DDataset,
                     Volume3DDataset, load_data, patient_split, save_split)
from modules import DiceCELoss, ImprovedUNet3D, UNet2D, count_parameters
from utils import amp_enabled, evaluate

# Sensible defaults per model (overridable from the command line)
DEFAULTS = {
    "3d": dict(epochs=200, batch=2, lr=5e-4, val_every=5),
    "2d": dict(epochs=40, batch=32, lr=1e-3, val_every=1),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["3d", "2d"], required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--val-every", type=int)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--out", default=None, help="output folder (default runs/<model>)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot-only", action="store_true",
                   help="only redraw curves.png from an existing history.json")
    p.add_argument("--quick", action="store_true",
                   help="tiny run (2 epochs, few cases) to check everything works")
    args = p.parse_args()
    for k, v in DEFAULTS[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.quick:
        args.epochs, args.val_every = 2, 1
    args.out = args.out or os.path.join("runs", args.model)
    return args


def build(args, data, split):
    """Create model, train loader and loss for the chosen mode."""
    if args.model == "3d":
        model = ImprovedUNet3D(in_channels=1, num_classes=len(CLASS_NAMES))
        train_ds = Volume3DDataset(data, split["train"], train=True)
    else:
        model = UNet2D(in_channels=1, num_classes=len(CLASS_NAMES))
        train_ds = Slice2DDataset(data, split["train"], train=True)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)
    return model, loader, DiceCELoss(dice_weight=0.5)


def foreground_mean(dice_mean: torch.Tensor) -> float:
    """Mean Dice over classes 1-5 (background excluded)."""
    return torch.nanmean(dice_mean[1:]).item()


def plot_curves(history: dict, path: str, title: str) -> None:
    """Save loss / validation Dice curves. Plotting is optional: if matplotlib
    is missing on the compute node, training still completes and the curves
    can be redrawn later from history.json (python train.py --plot-only)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping curves.png", flush=True)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["epoch"], history["train_loss"])
    ax[0].set(title="Training loss (Dice + CE)", xlabel="epoch", ylabel="loss")
    for c, name in enumerate(CLASS_NAMES[1:], start=1):
        ax[1].plot(history["val_epoch"], [d[c] for d in history["val_dice"]], label=name)
    ax[1].plot(history["val_epoch"], history["val_fg_mean"], "k--", label="mean (fg)")
    ax[1].axhline(0.7, color="grey", lw=0.8, ls=":")
    ax[1].set(title="Validation Dice", xlabel="epoch", ylabel="Dice", ylim=(0, 1))
    ax[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    if args.plot_only:
        with open(os.path.join(args.out, "history.json")) as f:
            plot_curves(json.load(f), os.path.join(args.out, "curves.png"),
                        "3D Improved U-Net" if args.model == "3d" else "2D U-Net baseline")
        return
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  bf16: {amp_enabled(device)}", flush=True)
    print(f"Device: {device}  |  model: {args.model}  |  args: {vars(args)}", flush=True)

    # ----- Data -----
    data = load_data(args.root, args.cache)
    split = patient_split(data["keys"], seed=args.seed)
    if args.quick:   # keep a handful of cases so the smoke test is fast
        split = {k: v[:4] if not k.endswith("patients") else v for k, v in split.items()}
    save_split(split, data["keys"], os.path.join(args.out, "split.json"))
    for name in ("train", "val", "test"):
        print(f"{name}: {len(split[name + '_patients'])} patients, "
              f"{len(split[name])} volumes", flush=True)

    model, loader, criterion = build(args, data, split)
    model.to(device)
    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,}", flush=True)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    history = {"epoch": [], "train_loss": [], "val_epoch": [], "val_dice": [],
               "val_fg_mean": []}
    best_score, best_epoch = -1.0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    # ----- Training loop -----
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, steps = 0.0, 0
        for batch in loader:
            images, labels = batch[0].to(device, non_blocking=True), \
                batch[1].to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            # bfloat16 mixed precision: ~half the memory, no loss scaling needed
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=amp_enabled(device)):
                logits = model(images)
            loss = criterion(logits.float(), labels)
            loss.backward()
            optimiser.step()
            running += loss.item()
            steps += 1
        scheduler.step()
        history["epoch"].append(epoch)
        history["train_loss"].append(running / max(1, steps))
        msg = f"epoch {epoch:3d}/{args.epochs}  loss {history['train_loss'][-1]:.4f}"

        # ----- Validation -----
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val = evaluate(model, data, split["val"], args.model, device)
            fg = foreground_mean(val["dice_mean"])
            history["val_epoch"].append(epoch)
            history["val_dice"].append(val["dice_mean"].tolist())
            history["val_fg_mean"].append(fg)
            msg += (f"  | val fg Dice {fg:.4f}  prostate {val['dice_mean'][5]:.4f}"
                    f"  ({(time.time() - start) / 60:.1f} min)")
            if fg > best_score:
                best_score, best_epoch = fg, epoch
                torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
                msg += "  *saved*"
        print(msg, flush=True)

    train_minutes = (time.time() - start) / 60
    peak_train_gb = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f)
    plot_curves(history, os.path.join(args.out, "curves.png"),
                f"{'3D Improved U-Net' if args.model == '3d' else '2D U-Net baseline'}")

    # ----- Test with the best checkpoint -----
    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=device))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    test = evaluate(model, data, split["test"], args.model, device)
    peak_infer_gb = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0

    metrics = {
        "model": args.model,
        "best_epoch": best_epoch,
        "best_val_fg_dice": best_score,
        "test_dice": dict(zip(CLASS_NAMES, test["dice_mean"].tolist())),
        "test_iou": dict(zip(CLASS_NAMES, test["iou_mean"].tolist())),
        "test_fg_mean_dice": foreground_mean(test["dice_mean"]),
        "prostate_extremity_dice": test["prostate_extremity_dice"],
        "prostate_central_dice": test["prostate_central_dice"],
        "test_dice_per_case": dict(zip(test["keys"], test["dice_per_case"].tolist())),
        "parameters": n_params,
        "peak_train_vram_gb": peak_train_gb,
        "peak_inference_vram_gb": peak_infer_gb,
        "seconds_per_volume": test["seconds_per_volume"],
        "train_minutes": train_minutes,
    }
    with open(os.path.join(args.out, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n===== TEST RESULTS (best checkpoint, epoch %d) =====" % best_epoch)
    for name in CLASS_NAMES:
        print(f"  {name:10s} Dice {metrics['test_dice'][name]:.4f}  "
              f"IoU {metrics['test_iou'][name]:.4f}")
    print(f"  foreground mean Dice      {metrics['test_fg_mean_dice']:.4f}")
    print(f"  prostate extremity Dice   {metrics['prostate_extremity_dice']:.4f}")
    print(f"  prostate central Dice     {metrics['prostate_central_dice']:.4f}")
    print(f"  parameters {n_params:,} | peak train VRAM {peak_train_gb:.2f} GB | "
          f"peak inference VRAM {peak_infer_gb:.2f} GB | "
          f"{metrics['seconds_per_volume'] * 1000:.0f} ms/volume | "
          f"train {train_minutes:.1f} min")


if __name__ == "__main__":
    main()
