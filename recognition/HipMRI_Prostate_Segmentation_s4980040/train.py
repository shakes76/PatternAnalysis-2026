"""
train.py - Train, validate, test and save a segmentation model.

WHAT HAPPENS WHEN YOU RUN IT
    1. Load the data (dataset.py) and build the model (modules.py).
    2. For each epoch (one full pass over the training set):
         - train on the training slices (the model's weights get updated)
         - evaluate on the validation slices (no updates, just measure Dice)
         - if validation prostate Dice is the best so far, save the model to disk
    3. Plot the loss and Dice curves.
    4. Reload the BEST saved model and evaluate it once on the test set.
    5. Print/save the final per-class Dice + IoU, training time and peak GPU memory.

HOW TO RUN
    Quick code check on the Rangpur login node (CPU, only a few batches, ~1 min):
        python train.py --smoke
    Real training (on a GPU node, via the Slurm script):
        python train.py --model unet2d --epochs 30 --preload
"""

import argparse
import json
import os
import time

import torch

from dataset import get_dataloaders, get_dataloaders_3d, DATA_ROOT, DATA_ROOT_3D, NUM_CLASSES, PROSTATE_LABEL
from modules import UNet2D, ImprovedUNet3D, count_parameters
from utils import CombinedLoss, SegmentationMetrics, CLASS_NAMES


def parse_args():
    """Command-line options, so settings can change without editing the code.
    e.g. `python train.py --epochs 50 --lr 5e-4`"""
    p = argparse.ArgumentParser(description="Train a HipMRI prostate segmentation model")
    p.add_argument("--model", default="unet2d", choices=["unet2d", "improved3d", "plain3d"],
                   help="unet2d = 2D baseline, improved3d = 3D Improved U-Net (main), "
                        "plain3d = same 3D net with residuals + deep supervision switched off")
    p.add_argument("--epochs", type=int, default=30, help="full passes over the training set")
    p.add_argument("--batch-size", type=int, default=None, help="samples per step (default: 16 for 2D, 2 for 3D)")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate: how big each weight update is")
    p.add_argument("--base", type=int, default=None, help="channels at the first level (default: 64 for 2D, 16 for 3D)")
    p.add_argument("--num-workers", type=int, default=4, help="CPU processes loading data in the background (2D only)")
    p.add_argument("--preload", action="store_true", help="load all 2D slices into RAM first (3D always preloads)")
    p.add_argument("--data-root", default=None, help="data folder (default: the Rangpur path for 2D or 3D)")
    p.add_argument("--out-dir", default="outputs", help="where checkpoints, plots and results go")
    p.add_argument("--seed", type=int, default=42, help="random seed, for reproducible results")
    p.add_argument("--smoke", action="store_true", help="tiny CPU test run to check the code works")
    return p.parse_args()


def get_device():
    """Use the NVIDIA GPU if there is one (Rangpur), else Apple's GPU (MPS), else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(name, base):
    """Create the requested model."""
    if name == "unet2d":
        return UNet2D(in_channels=1, num_classes=NUM_CLASSES, base=base)
    if name == "improved3d":  # main model: residual context modules + deep supervision ON
        return ImprovedUNet3D(in_channels=1, num_classes=NUM_CLASSES, base=base)
    if name == "plain3d":     # same network with both extras OFF (fallback / ablation)
        return ImprovedUNet3D(in_channels=1, num_classes=NUM_CLASSES, base=base,
                              residual=False, deep_supervision=False)
    raise ValueError(f"Unknown model {name}")


def run_epoch(model, loader, loss_fn, device, optimizer=None, max_batches=None):
    """One pass over a DataLoader.

    With an optimizer -> TRAINING (weights get updated).
    Without one       -> EVALUATION (weights frozen, just measure performance).
    Returns (average loss, SegmentationMetrics).
    """
    training = optimizer is not None
    # model.train() / model.eval() switch BatchNorm between "learn batch statistics"
    # and "use the statistics learned during training"
    model.train(training)
    metrics = SegmentationMetrics(NUM_CLASSES)
    total_loss, n_slices = 0.0, 0

    # Mixed precision on NVIDIA GPUs: do most maths in bfloat16 instead of float32.
    # Roughly 2x faster and uses less memory on an A100, with no real accuracy loss.
    use_amp = device.type == "cuda"

    # Only track gradients while training (evaluation is faster and lighter without them)
    with torch.set_grad_enabled(training):
        for i, (images, masks) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break  # used by --smoke to stop early
            # Move the batch from CPU RAM to the GPU
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(images)          # forward pass: (B, 6, 256, 128) class scores
                loss = loss_fn(logits, masks)   # how wrong were we?

            if training:
                optimizer.zero_grad(set_to_none=True)  # clear gradients from the previous step
                loss.backward()                        # work out how each weight affected the loss
                optimizer.step()                       # nudge every weight to reduce the loss

            total_loss += loss.item() * images.size(0)  # loss.item() is the batch average
            n_slices += images.size(0)
            metrics.update(logits, masks)

    return total_loss / max(n_slices, 1), metrics


def plot_history(history, out_dir, model_name):
    """Save the training curves (the spec requires plotting losses and metrics)."""
    import matplotlib
    matplotlib.use("Agg")  # no screen on Rangpur: draw straight to a file
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: loss. Both lines should go down. If val loss starts rising while train loss
    # keeps falling, the model is overfitting (memorising the training set).
    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="validation")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("CE + Dice loss")
    ax[0].set_title("Loss")
    ax[0].legend()

    # Right: validation Dice for each foreground class, prostate drawn thicker
    for c in range(1, NUM_CLASSES):
        dice_c = [d[c] for d in history["val_dice"]]
        ax[1].plot(epochs, dice_c, label=CLASS_NAMES[c], linewidth=3 if c == PROSTATE_LABEL else 1.2)
    ax[1].axhline(0.75, color="grey", linestyle="--", label="prostate target (0.75)")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Dice")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Validation Dice per class")
    ax[1].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_training_curves.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    args = parse_args()
        # Fill in defaults that depend on 2D vs 3D
    is3d = args.model != "unet2d"
    if args.base is None:
        args.base = 16 if is3d else 64
    if args.batch_size is None:
        args.batch_size = 2 if is3d else 16  # 3D volumes are big, so far fewer per batch
    if args.data_root is None:
        args.data_root = DATA_ROOT_3D if is3d else DATA_ROOT
    torch.manual_seed(args.seed)  # same seed -> same random init and shuffling -> reproducible
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    # --smoke = just check nothing crashes: 1 epoch, 3 batches per split, no worker processes
    max_batches = 3 if args.smoke else None
    epochs = 1 if args.smoke else args.epochs
    num_workers = 0 if args.smoke else args.num_workers

    # ---- Data and model ----
    if is3d:
        train_loader, val_loader, test_loader = get_dataloaders_3d(args.batch_size, args.data_root)
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            args.batch_size, args.data_root, num_workers, args.preload and not args.smoke)
    model = build_model(args.model, args.base).to(device)  # .to(device) moves the weights onto the GPU
    n_params = count_parameters(model)
    print(f"Model: {args.model} | trainable parameters: {n_params:,}")

    loss_fn = CombinedLoss(NUM_CLASSES)
    # Adam: a standard optimiser that adapts the step size for each weight automatically
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Cosine schedule: start at lr and smoothly lower it to ~0 by the last epoch,
    # so the model makes big moves early and fine adjustments at the end
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ---- Training loop ----
    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = -1.0
    ckpt_path = os.path.join(args.out_dir, f"best_{args.model}.pt")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()  # so the peak VRAM reading only covers this run
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, _ = run_epoch(model, train_loader, loss_fn, device, optimizer, max_batches)
        val_loss, val_metrics = run_epoch(model, val_loader, loss_fn, device, None, max_batches)
        scheduler.step()  # lower the learning rate a little for the next epoch

        val_dice = val_metrics.dice()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        print(f"Epoch {epoch:3d}/{epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
              f"| val prostate Dice {val_dice[PROSTATE_LABEL]:.4f} | {time.time() - t0:.0f}s")

        # Keep only the best model (by validation prostate Dice), not just the last one,
        # in case the model gets worse (overfits) in later epochs
        if val_dice[PROSTATE_LABEL] > best_dice:
            best_dice = val_dice[PROSTATE_LABEL]
            torch.save({"model": args.model, "base": args.base, "epoch": epoch,
                        "val_dice": val_dice, "state_dict": model.state_dict()}, ckpt_path)
            print(f"  -> new best, saved to {ckpt_path}")

    train_time = time.time() - train_start
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else None

    plot_history(history, args.out_dir, args.model)
    with open(os.path.join(args.out_dir, f"{args.model}_history.json"), "w") as f:
        json.dump(history, f)

    # ---- Final test evaluation with the BEST checkpoint ----
    # The test set is only touched once, here at the end, so it stays an honest
    # estimate of performance on brand-new patients.
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    t0 = time.time()
    _, test_metrics = run_epoch(model, test_loader, loss_fn, device, None, max_batches)
    if device.type == "cuda":
        torch.cuda.synchronize()  # wait for the GPU to actually finish before stopping the clock
    test_time = time.time() - t0

    test_dice, test_iou = test_metrics.dice(), test_metrics.iou()
    print(f"\nTest results (best epoch {ckpt['epoch']}, {test_metrics.count} slices):")
    print(f"{'class':<12}{'Dice':>8}{'IoU':>8}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"{name:<12}{test_dice[c]:>8.4f}{test_iou[c]:>8.4f}")
    print(f"\nParameters: {n_params:,}")
    print(f"Training time: {train_time / 60:.1f} min")
    if peak_vram_gb is not None:
        print(f"Peak GPU memory (training): {peak_vram_gb:.2f} GB")
    # Note: this time includes loading files from disk; predict.py will measure pure model time
    print(f"Test time per slice (incl. data loading): {1000 * test_time / max(test_metrics.count, 1):.1f} ms")

    # Save everything for the README tables
    results = {"model": args.model, "best_epoch": ckpt["epoch"], "params": n_params,
               "train_time_min": train_time / 60, "peak_vram_gb": peak_vram_gb,
               "test_dice": dict(zip(CLASS_NAMES, test_dice)),
               "test_iou": dict(zip(CLASS_NAMES, test_iou)), "args": vars(args)}
    with open(os.path.join(args.out_dir, f"{args.model}_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()