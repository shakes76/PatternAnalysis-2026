"""
Train, validate, test and save a segmentation model on HipMRI 2D slices.

Examples:
    python train.py --model can --epochs 30
    python train.py --model unet --epochs 30
    python train.py --model can --epochs 2 --limit 64   # smoke test

Outputs go to runs/<model>/:
    best.pt          weights with the best validation prostate Dice
    history.json     per-epoch losses and validation Dice
    test_metrics.json  per-class Dice / IoU on the test split, plus profiling
    curves.png       loss and Dice curves
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator

from dataset import CLASS_NAMES, PROSTATE, get_dataloaders
from modules import CAN, ResidualCAN, UNet, count_parameters
from utils import DiceCELoss, SegmentationMeter, get_device

MODELS = {"unet": UNet, "can": CAN, "can_res": ResidualCAN}
HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default="can")
    p.add_argument("--data-root", default=os.path.join(HERE, "data", "keras_slices_data"))
    p.add_argument("--run-name", default=None, help="name for this run; defaults to the model name")
    p.add_argument("--out-dir", default=None, help="defaults to runs/<run-name>")
    p.add_argument("--base-channels", type=int, default=None,
                   help="first-level width (default: 32 for unet, 64 for can and can_res)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    # Adam with lr 5e-4 decayed by 0.985 each epoch (Isensee et al., 2017), same for every model.
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-decay", type=float, default=0.985)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="use only N slices per split")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        loss = loss_fn(model(images), masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Return (mean loss, SegmentationMeter) over a whole split."""
    model.eval()
    meter = SegmentationMeter()
    total, n = 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total += loss_fn(logits, masks).item() * images.size(0)
        n += images.size(0)
        meter.update(logits, masks)
    return total / n, meter


def peak_memory_mb(device):
    """Peak allocated accelerator memory in MB, where the backend reports it."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    if device.type == "mps":
        # MPS has no peak counter; driver memory at the end of a run is the
        # closest available figure and over-estimates the true peak.
        return torch.mps.driver_allocated_memory() / 2**20
    return None


def plot_history(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set(xlabel="epoch", ylabel="Dice + CE loss", title="Loss")
    ax1.legend()
    for c, name in enumerate(CLASS_NAMES):
        if c == 0:
            continue
        ax2.plot(epochs, [d[c] for d in history["val_dice"]], label=name)
    ax2.axhline(0.75, color="grey", linestyle="--", linewidth=1, label="target (prostate)")
    ax2.set(xlabel="epoch", ylabel="Dice", title="Validation Dice per class", ylim=(0, 1))
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    run_name = args.run_name or args.model
    out_dir = args.out_dir or os.path.join(HERE, "runs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    loaders = get_dataloaders(args.data_root, args.batch_size, args.num_workers, limit=args.limit)
    # Only pass options that were set, so each model keeps its own defaults.
    model_kwargs = {}
    if args.base_channels is not None:
        model_kwargs["base_channels"] = args.base_channels
    model = MODELS[args.model](**model_kwargs).to(device)
    loss_fn = DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    n_params = count_parameters(model)
    print(f"{args.model}: {n_params:,} params on {device}, "
          f"{len(loaders['train'].dataset)} train / {len(loaders['validate'].dataset)} val slices")

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "epoch_time": []}
    best_dice, ckpt_path = -1.0, os.path.join(out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device)
        val_loss, meter = evaluate(model, loaders["validate"], loss_fn, device)
        scheduler.step()

        val_dice = meter.dice()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["epoch_time"].append(time.time() - start)

        # Model selection on validation prostate Dice, the target structure.
        is_best = val_dice[PROSTATE] > best_dice
        if is_best:
            best_dice = val_dice[PROSTATE]
            torch.save({"model": args.model, "kwargs": model_kwargs, "run_name": run_name,
                        "seed": args.seed, "state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)

        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"prostate Dice {val_dice[PROSTATE]:.4f}  "
              f"({history['epoch_time'][-1]:.0f}s){'  *saved' if is_best else ''}")

        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        plot_history(history, os.path.join(out_dir, "curves.png"))

    # Test the best checkpoint, not the last epoch.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    test_loss, meter = evaluate(model, loaders["test"], loss_fn, device)
    test = {
        "model": args.model,
        "run_name": run_name,
        "kwargs": model_kwargs,
        "seed": args.seed,
        "best_epoch": torch.load(ckpt_path, map_location="cpu")["epoch"],
        "test_loss": test_loss,
        "dice": dict(zip(CLASS_NAMES, meter.dice())),
        "iou": dict(zip(CLASS_NAMES, meter.iou())),
        "params": n_params,
        "device": str(device),
        "mean_epoch_time_s": sum(history["epoch_time"]) / len(history["epoch_time"]),
        "peak_memory_mb": peak_memory_mb(device),
    }
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test, f, indent=2)

    print("\nTest set (best checkpoint, epoch {}):".format(test["best_epoch"]))
    for name in CLASS_NAMES:
        print(f"  {name:10s} Dice {test['dice'][name]:.4f}  IoU {test['iou'][name]:.4f}")


if __name__ == "__main__":
    main()
