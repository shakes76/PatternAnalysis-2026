"""
train.py — train a model on ADNI (AD vs NC) and evaluate it on the test set.

Usage (Mac):
    python train.py --root "/Users/ryanhuang/Documents/UQ/Sem 2 2026/COMP3710/ADNI/AD_NC" --epochs 5

Usage (Rangpur, default root):
    python train.py --epochs 30 --batch-size 64

"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn

from dataset import DEFAULT_ROOT, get_dataloaders
from modules import SimpleCNN, count_parameters


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")          # Rangpur GPU
    if torch.backends.mps.is_available():
        return torch.device("mps")           # Apple GPU
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training data. Returns (mean loss, accuracy)."""
    model.train()                            # enables dropout, BatchNorm updates
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()                # clear old gradients
        logits = model(x)                    # forward pass
        loss = criterion(logits, y)
        loss.backward()                      # backpropagation
        optimizer.step()                     # update weights

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


@torch.no_grad()                             # no gradients needed for evaluation
def evaluate(model, loader, criterion, device):
    """Returns (mean loss, slice accuracy, AD probability per slice)."""
    model.eval()                             # disables dropout, freezes BatchNorm
    total_loss, correct, n = 0.0, 0, 0
    probs = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
        probs.append(logits.softmax(1)[:, 1].cpu())
    return total_loss / n, correct / n, torch.cat(probs)


def patient_accuracy(dataset, probs) -> float:
    """Average each patient's slice probabilities, threshold at 0.5.
    Works because eval loaders don't shuffle, so probs line up with dataset order."""
    per_patient = defaultdict(list)
    labels = {}
    for pid, (_, y), p in zip(dataset.patients, dataset.samples, probs.tolist()):
        per_patient[pid].append(p)
        labels[pid] = y
    correct = sum(int((sum(ps) / len(ps) > 0.5) == labels[pid]) for pid, ps in per_patient.items())
    return correct / len(per_patient)


def save_curves(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train on ADNI AD vs NC")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/simplecnn")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ---- data ----
    train_loader, val_loader, test_loader = get_dataloaders(
        args.root, batch_size=args.batch_size, img_size=args.img_size,
        num_workers=args.workers, seed=args.seed,
    )
    print(f"Slices — train {len(train_loader.dataset)}, val {len(val_loader.dataset)}, "
          f"test {len(test_loader.dataset)}")

    # ---- model, loss, optimiser, LR schedule ----
    model = SimpleCNN(in_chans=1, num_classes=2).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine decay: LR starts at --lr and smoothly falls to ~0 by the last epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- training loop ----
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = os.path.join(args.out, "best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(v)

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": va_acc,
                        "args": vars(args)}, best_path)
            marker = "  * saved best"

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s{marker}")

    # ---- final test with the best checkpoint ----
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_acc, te_probs = evaluate(model, test_loader, criterion, device)
    te_pat_acc = patient_accuracy(test_loader.dataset, te_probs)

    print(f"\nBest epoch {ckpt['epoch']} (val acc {ckpt['val_acc']:.4f})")
    print(f"Test slice accuracy:   {te_acc:.4f}")
    print(f"Test patient accuracy: {te_pat_acc:.4f}")

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({**history, "best_epoch": ckpt["epoch"], "test_acc": te_acc,
                   "test_patient_acc": te_pat_acc}, f, indent=2)
    save_curves(history, os.path.join(args.out, "curves.png"))
    print(f"Saved checkpoint, history and curves to {args.out}/")


if __name__ == "__main__":       # required on macOS: DataLoader workers re-import this file
    main()