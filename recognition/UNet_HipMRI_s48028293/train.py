"""
train.py trains a 2D U-Net, 3D U-Net or 3D Improved U-Net on HipMRI, keeps the best checkpoint by 
validation prostate Dice, and reports its test results.
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import (CLASS_NAMES, DATA_ROOT_3D, DOWNSAMPLE, IMG_SIZE, NUM_CLASSES, PROSTATE, VOL_SIZE, get_loaders)
from modules import CombinedLoss, build_model
from utils import evaluate, format_results

DEFAULT_DATA_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
MODEL_DEFAULTS = {
    "unet2d": dict(data_root=DEFAULT_DATA_ROOT, epochs=30, batch_size=32, base=64,prostate_weight=3.0),
    "unet3d": dict(data_root=DATA_ROOT_3D, epochs=100, batch_size=2, base=16, prostate_weight=3.0),
    "improved_unet3d": dict(data_root=DATA_ROOT_3D, epochs=100, batch_size=2, base=16, prostate_weight=3.0),
}

def parse_args():
    # Parse command-line arguments and fill in defaults based on the model type.
    p = argparse.ArgumentParser(description="Train a HipMRI segmentation model")
    p.add_argument("--model", default="unet2d", choices=list(MODEL_DEFAULTS))
    p.add_argument("--data-root", default=None, help="default depends on --model")
    p.add_argument("--out-dir", default=None, help="default: ../runs/<model>")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--prostate-weight", type=float, default=None,          # <- new
                help="Dice loss weight of the prostate class (default depends on --model)")
    p.add_argument("--base", type=int, default=None, help="channels in the first level")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                    help="use only the first N slices per split (smoke test)")
    p.add_argument("--no-preload", action="store_true",
                    help="read slices from disk each epoch instead of caching in RAM")
    args = p.parse_args()
    for key, value in MODEL_DEFAULTS[args.model].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.dims = 2 if args.model == "unet2d" else 3
    args.img_size = VOL_SIZE if args.dims == 3 else IMG_SIZE
    if args.out_dir is None:                              # <- new
        args.out_dir = os.path.join(RUNS_DIR, args.model) # <- new
    return args  

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    # Train the model for one epoch and return the average loss.
    model.train()
    use_cuda = device.type == "cuda"
    total_loss, n_batches = 0.0, 0
    for images, masks, _, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
            out = model(images)
            if isinstance(out, tuple):           # Improved U-Net in training mode
                logits, aux = out
                loss = criterion(logits, masks) + sum(
                    w * criterion(a, masks) for w, a in zip((0.5, 0.25), aux))
            else:
                loss = criterion(out, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)

def plot_history(history, path):
    # Plot training and validation loss and validation Dice per class.
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy + Dice loss")
    ax1.set_title("Loss")
    ax1.legend()

    val_dice = np.array(history["val_dice"])  # (epochs, classes)
    for c, name in enumerate(CLASS_NAMES):
        ax2.plot(epochs, val_dice[:, c], label=name,
                linewidth=2.5 if c == PROSTATE else 1.0)
    ax2.axhline(0.75, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation DSC (per volume)")
    ax2.set_title("Validation Dice per class")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def main():
    # Parse command-line arguments, set up the model, data loaders, loss, optimizer, and training loop.
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print("Using device:", device)
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))

    loaders = get_loaders(args.data_root, batch_size=args.batch_size, img_size=args.img_size,
                        workers=args.workers, limit=args.limit,
                        preload=not args.no_preload, dims=args.dims)

    model = build_model(args.model, NUM_CLASSES, base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | parameters: {n_params:,}")

    dice_weights = [1.0] * NUM_CLASSES
    dice_weights[PROSTATE] = args.prostate_weight
    criterion = CombinedLoss(NUM_CLASSES, dice_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_cuda)

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice, best_epoch = -1.0, 0
    ckpt_path = os.path.join(args.out_dir, "best_model.pt")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    print("\nStarting training...")
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, device)
        meter, val_loss, _ = evaluate(model, loaders["val"], device, NUM_CLASSES, criterion)
        scheduler.step()

        val_dice = meter.mean_dice()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice.tolist())

        flag = ""
        if val_dice[PROSTATE] > best_dice:  # keep the best model, not the last one
            best_dice, best_epoch = float(val_dice[PROSTATE]), epoch
            torch.save({"model_state_dict": model.state_dict(), "model": args.model,
                        "dims": args.dims, "base": args.base, "num_classes": NUM_CLASSES,
                        "class_names": CLASS_NAMES, "img_size": args.img_size,
                        "downsample": DOWNSAMPLE,
                        "seed": args.seed,
                        "epoch": epoch, "val_prostate_dice": best_dice}, ckpt_path)
            flag = " *saved*"
        print(f"Epoch [{epoch}/{args.epochs}] train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | val DSC prostate {val_dice[PROSTATE]:.4f} "
                f"mean {np.nanmean(val_dice):.4f} | {time.time() - epoch_start:.1f}s{flag}", flush=True)

        # Rewritten every epoch so a cancelled job still leaves its curves behind.
        plot_history(history, os.path.join(args.out_dir, "training_curves.png"))
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    train_time = time.time() - train_start
    train_vram = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else 0.0
    print(f"\nTraining time: {train_time / 60:.1f} min | best epoch {best_epoch} "
            f"(val prostate DSC {best_dice:.4f})")

    # ---- Test the best checkpoint on the held-out test split ----
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    del optimizer, scheduler, scaler  # free optimizer state so it isn't counted in inference VRAM
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    meter, test_loss, forward_time = evaluate(model, loaders["test"], device, NUM_CLASSES, criterion)
    infer_vram = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else 0.0
    n_volumes = len(meter.volumes)

    report = "\n".join([
        f"Model: {args.model} (base {args.base}, {n_params:,} parameters)",
        f"Prostate Dice weight: {args.prostate_weight}",
        f"Best epoch: {best_epoch} of {args.epochs}",
        f"Training time: {train_time / 60:.1f} min",
        f"Peak VRAM, training:  {train_vram:.0f} MiB (batch size {args.batch_size})",
        f"Peak VRAM, inference: {infer_vram:.0f} MiB (batch size {args.batch_size})",
        f"Inference time per case: {forward_time / max(n_volumes, 1):.3f} s (model forward only)",
        f"Test loss: {test_loss:.4f}",
        "",
        "Test results (per volume)",
        format_results(meter, CLASS_NAMES),
    ])
    print("\n" + report)
    with open(os.path.join(args.out_dir, "test_report.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\nOutputs saved to {args.out_dir}")

if __name__ == "__main__":
    main()
