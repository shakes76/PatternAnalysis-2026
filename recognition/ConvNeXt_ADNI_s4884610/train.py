"""
Training, validation and testing for ADNI classification.

Usage:
    python3 train.py --args
"""

import argparse
import json
import time
import os
import torch
import torch.nn as nn
from dataset import get_dataloaders
from modules import BaselineCNN, ConvNeXt
import matplotlib.pyplot as plt


def get_device():
    """Use an NVIDIA GPU if available, then Apple, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(name, drop_path=0.0):
    """Return an initialised model from its name."""
    if name == "baseline":
        return BaselineCNN()
    if name == "convnext":
        return ConvNeXt(drop_path=drop_path)
    raise ValueError(f"Unknown model: {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on ADNI AD vs NC")
    parser.add_argument("--model", default="baseline", choices=["baseline", "convnext"])
    parser.add_argument(
        "--data_root", default=os.path.expanduser("~/comp3710_data/AD_NC")
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--schedule", default="constant", choices=["constant", "cosine"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--strong_aug", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", default="outputs")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training set. Returns (mean loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a validation or test loader.
    Returns (mean loss, accuracy, AD probabilities, true labels).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)

        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        all_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
        all_labels.append(labels.cpu())

    return (
        total_loss / total,
        correct / total,
        torch.cat(all_probs),
        torch.cat(all_labels),
    )


def plot_history(history, out_path, title):
    """Save loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history["train_loss"], marker="o", label="Train")
    ax_loss.plot(epochs, history["val_loss"], marker="o", label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid()

    ax_acc.plot(epochs, history["train_acc"], marker="o", label="Train")
    ax_acc.plot(epochs, history["val_acc"], marker="o", label="Validation")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        strong_aug=args.strong_aug,
    )

    model = build_model(args.model, drop_path=args.drop_path).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params:,} parameters)")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine: smoothly lower the learning rate from lr to ~0 over all epochs.
    scheduler = None
    if args.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.model}_best.pt")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
            f"lr {current_lr:.2e} | {time.time() - start:.0f}s"
        )

    train_time = time.time() - train_start

    # Plot the loss and accuracy itself.
    plot_history(
        history,
        os.path.join(args.out_dir, f"{args.model}_curves.png"),
        f"{args.model} training curves",
    )

    # Final test evaluation using the best validation checkpoint.
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"Best val accuracy {best_val_acc:.3f} | "
        f"Test loss {test_loss:.4f} | Test accuracy {test_acc:.3f}"
    )
    print(f"Training time {train_time / 60:.1f} min | Peak VRAM {peak_vram_gb:.2f} GB")

    # Save the numbers needed for the README results and resources tables.
    results = {
        "model": args.model,
        "params": n_params,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "schedule": args.schedule,
        "label_smoothing": args.label_smoothing,
        "drop_path": args.drop_path,
        "strong_aug": args.strong_aug,
        "batch_size": args.batch_size,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_time_min": train_time / 60,
        "peak_vram_gb": peak_vram_gb,
        "history": history,
    }
    with open(os.path.join(args.out_dir, f"{args.model}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
