"""
Train narrow ConvNeXt for AD/NC classification.

Defaults match the reported 77.40% validation experiment: 60 epochs,
LR 3e-4, WD 0.05, batch size 64, no EMA, and no brightness/contrast jitter.
Use --amp for the reported mixed-precision run and reuse its manifest.
Optional --ema and --jitter reproduce later exploratory settings.
Save best_model.pth by validation image-level balanced accuracy.

Example:
  python train.py --manifest manifest.csv --out_dir checkpoints --amp
"""
import os
import time
import argparse
import csv
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules import ConvNeXtNarrow, build_model
from dataset import (
    build_manifest, load_manifest, ADNISliceDataset, split_dataframe,
    ensure_data_root, DEFAULT_DATA_ROOT, DEFAULT_ZIP_PATH, DEFAULT_OUTPUT_DIR,
)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_msg(log_file, msg):
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")


def save_training_outputs(history, out_dir):
    """Save machine-readable history and update loss/metric curves."""
    csv_path = os.path.join(out_dir, "training_history.csv")
    fieldnames = [
        "epoch", "train_loss", "val_loss", "train_acc", "val_acc",
        "val_bal_acc", "lr", "epoch_seconds",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    # Use a non-interactive backend so plotting also works on Colab/HPC.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        epochs,
        [row["train_loss"] for row in history],
        label="Train loss (label smoothing)",
    )
    axes[0].plot(
        epochs,
        [row["val_loss"] for row in history],
        label="Validation loss (hard targets)",
    )
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="Train accuracy")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="Validation accuracy")
    axes[1].plot(
        epochs,
        [row["val_bal_acc"] for row in history],
        label="Validation balanced accuracy",
    )
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0.0, 1.0), title="Metrics")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=200)
    plt.close(fig)


class ModelEMA:
    """Maintain an exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1")
        self.decay = float(decay)
        self.module = deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_parameters = dict(model.named_parameters())
        for name, ema_parameter in self.module.named_parameters():
            model_parameter = model_parameters[name].detach()
            ema_parameter.mul_(self.decay).add_(
                model_parameter,
                alpha=1.0 - self.decay,
            )

        # Keep non-trainable state synchronized. ConvNeXt currently has no
        # running-stat buffers, but copying them makes EMA safe for extensions.
        model_buffers = dict(model.named_buffers())
        for name, ema_buffer in self.module.named_buffers():
            ema_buffer.copy_(model_buffers[name].detach())


# --------------------------------------------------------------------------- #
# Training and validation loops
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, optimizer, device, scaler=None, ema=None):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:  # AMP mixed precision
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # GradScaler skips optimizer.step() when gradients overflow.
            if ema is not None and scaler.get_scale() >= previous_scale:
                ema.update(model)
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Return validation loss, balanced accuracy, and ordinary accuracy."""
    model.eval()
    total_loss, correct, total_n = 0.0, 0, 0
    tp_ad = fp_ad = fn_ad = 0
    tp_nc = fp_nc = fn_nc = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(1)
        total_loss += loss.item() * x.size(0)
        correct += (pred == y).sum().item()
        total_n += x.size(0)
        for p, t in zip(pred.tolist(), y.tolist()):
            if t == 1:  # AD
                if p == 1:
                    tp_ad += 1
                else:
                    fn_ad += 1
            else:       # NC
                if p == 0:
                    tp_nc += 1
                else:
                    fp_nc += 1
    fp_ad = fp_nc  # Predicted AD, actual NC.
    fn_nc = fp_nc  # Actual NC, predicted AD.
    rec_ad = tp_ad / (tp_ad + fn_ad) if (tp_ad + fn_ad) else 0.0
    rec_nc = tp_nc / (tp_nc + fn_nc) if (tp_nc + fn_nc) else 0.0
    balanced_acc = (rec_ad + rec_nc) / 2
    return (
        total_loss / max(total_n, 1),
        balanced_acc,
        correct / max(total_n, 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--zip_path", default=DEFAULT_ZIP_PATH)
    ap.add_argument("--manifest", default=None,
                    help="Existing manifest CSV; scan the dataset if omitted.")
    ap.add_argument("--out_dir", "--output_dir", dest="out_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="Enable mixed-precision training.")
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--drop_path_rate", type=float, default=0.2)
    ap.add_argument("--ema", action="store_true", help="Use EMA weights; off for the final report model.")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="Brightness/contrast strength; 0 disables jitter.")
    ap.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay applied after each successful optimizer update.",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, "train_log.txt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Data ----------------
    args.data_root = ensure_data_root(args.data_root, args.zip_path)
    if args.manifest:
        if not os.path.isfile(args.manifest):
            raise FileNotFoundError(f"Manifest not found: {args.manifest}")
        df = load_manifest(args.manifest, data_root=args.data_root)
        df.to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)
        log_msg(log_file, f"Loaded existing manifest: {args.manifest}")
    else:
        df = build_manifest(
            args.data_root,
            out_csv=os.path.join(args.out_dir, "manifest.csv"),
            seed=args.seed,
        )
    subsets = split_dataframe(df)

    train_ds = ADNISliceDataset(subsets["train"], args.img_size, train=True, jitter=args.jitter)
    val_ds = ADNISliceDataset(subsets["val"], args.img_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ---------------- Model and optimizer ----------------
    model = ConvNeXtNarrow(
        in_channels=1, num_classes=2,
        base_channels=args.base_channels,
        depths=(3, 3, 9, 3),
        drop_path_rate=args.drop_path_rate,
    ).to(device)
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None
    log_msg(log_file, f"Parameters: {model.num_parameters():,}")
    log_msg(log_file, f"device: {device}")
    log_msg(log_file, f"EMA: {args.ema}; decay: {args.ema_decay if args.ema else 'off'}")
    log_msg(log_file, f"Brightness/contrast jitter: {args.jitter}")
    log_msg(log_file, f"LR: {args.lr}; WD: {args.weight_decay}; epochs: {args.epochs}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Validation loss uses hard targets and does not affect training or
    # checkpoint selection. This keeps it interpretable across experiments.
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device == "cuda") else None

    # ---------------- Training loop ----------------
    best_bal_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, ema
        )
        # The final report selects ordinary model weights; EMA is optional.
        evaluation_model = ema.module if ema is not None else model
        val_loss, val_bal_acc, val_acc = evaluate(
            evaluation_model, val_loader, val_criterion, device
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_seconds = time.time() - t0

        msg = (f"epoch {epoch:3d}/{args.epochs} | "
               f"train_loss {train_loss:.4f} | "
               f"val_loss {val_loss:.4f} | "
               f"train_acc {train_acc:.4f} | "
               f"val_acc {val_acc:.4f} | "
               f"val_bal_acc {val_bal_acc:.4f} | "
               f"lr {current_lr:.2e} | "
               f"{epoch_seconds:.1f}s")
        log_msg(log_file, msg)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_bal_acc": val_bal_acc,
            "lr": current_lr,
            "epoch_seconds": epoch_seconds,
        })
        # Update after every epoch so partial runs still retain their plots.
        save_training_outputs(history, args.out_dir)

        if val_bal_acc >= best_bal_acc:
            best_bal_acc = val_bal_acc
            checkpoint = {
                "epoch": epoch,
                "state_dict": evaluation_model.state_dict(),
                "weight_source": "ema" if ema is not None else "model",
                "val_loss": val_loss,
                "val_bal_acc": val_bal_acc,
                "val_acc": val_acc,
                "args": vars(args),
            }
            if ema is not None:
                checkpoint.update({
                    "ema_state_dict": ema.module.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "ema_decay": args.ema_decay,
                })
            torch.save(checkpoint, os.path.join(args.out_dir, "best_model.pth"))
            log_msg(log_file, f"  -> Saved best model (val_bal_acc={val_bal_acc:.4f})")

    log_msg(log_file, f"Training complete; best val_bal_acc = {best_bal_acc:.4f}")


if __name__ == "__main__":
    main()