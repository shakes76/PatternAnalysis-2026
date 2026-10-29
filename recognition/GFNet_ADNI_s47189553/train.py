"""
train.py — train GFNet on ADNI (AD vs NC). Test-set evaluation is in predict.py.

    python train.py                  # one train/val split -> runs/gfnet/best_no_cv.pt
    python train.py --fold 2         # one cross-validation fold -> runs/gfnet/fold2/best.pt
    python train.py --cv             # all 5 folds -> val mean ± std + runs/gfnet/cv_best.pt
    python train.py --subset 0.05    # quick test on 5% of patients
    python train.py --cv --no-crop-flip   # train without RandomResizedCrop / HorizontalFlip
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from dataset import DEFAULT_N_FOLDS, DEFAULT_ROOT, get_dataloaders, get_fold_dataloaders
from modules import GFNet, SimpleCNN

# --------------------------------------------------------------------------- #
# 0. Configuration
# --------------------------------------------------------------------------- #
EPOCHS = 100                    # Max passes over the training data (early stopping usually ends sooner)
BATCH_SIZE = 64                 # Slices per weight update
LR = 5e-4                       # Starting learning rate for AdamW
MIN_LR = 1e-6                   # Learning rate at the end of cosine decay
WEIGHT_DECAY = 0.05             # AdamW weight decay (GFNet/DeiT default)
LABEL_SMOOTHING = 0.1           # Soft targets [0.05, 0.95] instead of [0, 1]
PATIENCE = 40                   # Stop after this many epochs without val improvement
WORKERS = 4                     # Background processes loading images
SEED = 42                       # Makes weight init, batch order and splits repeatable
OUT_DIR = "runs/gfnet"          # Where checkpoints / history.json are saved
NO_CV_CKPT = "best_no_cv.pt"    # Best model from a normal (no-CV) run
CV_CKPT = "cv_best.pt"          # Best fold's model after --cv
PATIENT_THRESHOLD = 0.5         # Patient predicted AD if mean slice P(AD) > this
AD_LABEL = 1                    # dataset.py: NC = 0, AD = 1
FULL_SUBSET = 1.0               # Fraction of patients to use (for quick tests)

# --------------------------------------------------------------------------- #
# 1. Helpers
# --------------------------------------------------------------------------- #
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")                         # NVIDIA GPU (cloud)
    if torch.backends.mps.is_available():
        return torch.device("mps")                          # Apple GPU (quick local tests)
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_time(seconds: float) -> str:
    """12345.6 -> '3h 25m 46s'."""
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------- #
# 2. Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()                                            # No gradients needed for evaluation
def evaluate(model, loader, criterion, device):
    """Return (mean loss, slice accuracy, P(AD) for every slice)."""
    model.eval()                                            # Disable dropout and drop-path
    total_loss, correct, n, probs = 0.0, 0, 0, []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                                   # Forward pass: (B, 2)
        total_loss += criterion(logits, y).item() * len(y)  # Sum loss over the batch
        correct += (logits.argmax(1) == y).sum().item()     # Count correct slices
        n += len(y)
        probs.append(logits.softmax(1)[:, AD_LABEL].cpu())  # Keep P(AD) for patient voting
    return total_loss / n, correct / n, torch.cat(probs)


def patient_accuracy(dataset, probs) -> float:
    """Average each patient's slice probabilities; patient is AD if mean > threshold.
    Works because eval loaders don't shuffle, so probs line up with dataset order."""
    per_patient, label = defaultdict(list), {}
    for pid, (_, y), p in zip(dataset.patients, dataset.samples, probs.tolist()):
        per_patient[pid].append(p)                          # Collect every slice's P(AD)
        label[pid] = y                                      # A patient's slices share one label
    correct = sum((sum(ps) / len(ps) > PATIENT_THRESHOLD) == label[pid]
                  for pid, ps in per_patient.items())
    return correct / len(per_patient)


# --------------------------------------------------------------------------- #
# 3. One full training run (used by every mode)
# --------------------------------------------------------------------------- #
def run_training(args, train_loader, val_loader, out_dir, device, ckpt_name="best.pt"):
    """Train a fresh GFNet; keep the best-val checkpoint as out_dir/ckpt_name.
    Returns (ckpt, history)."""
    set_seed(args.seed)                                     # Same starting weights for every fold
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.abspath(os.path.join(out_dir, ckpt_name))

    # ---- model ----
    model = (SimpleCNN() if args.model == "simplecnn" else GFNet()).to(device)   # Settings live in modules.py

    # ---- optimizer: no weight decay on biases, LayerNorm and pos_embed ----
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith("pos_embed") else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )

    # ---- LR schedule: cosine decay from args.lr to MIN_LR, stepped every batch ----
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=MIN_LR
    )

    # ---- losses ----
    train_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    eval_criterion = nn.CrossEntropyLoss()                  # Plain CE so val loss compares to test

    # ---- training loop ----
    history = defaultdict(list)
    best_val_acc, stale = -1.0, 0                           # stale = epochs since last improvement

    for epoch in range(1, args.epochs + 1):
        model.train()                                       # Enable dropout and drop-path
        epoch_start = time.time()
        lr = optimizer.param_groups[0]["lr"]                # LR at the start of this epoch
        running_loss, correct, n = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()                           # Clear gradients from the last batch
            logits = model(x)                               # Forward pass: (B, 2)
            loss = train_criterion(logits, y)               # Smoothed cross-entropy
            loss.backward()                                 # Backpropagate: compute dLoss/dw
            optimizer.step()                                # Update weights
            scheduler.step()                                # Advance cosine LR schedule

            running_loss += loss.item() * len(y)            # Sum loss over the batch
            correct += (logits.argmax(1) == y).sum().item() # Count correct slices
            n += len(y)

        train_loss, train_acc = running_loss / n, correct / n

        # ---- validate ----
        val_loss, val_acc, val_probs = evaluate(model, val_loader, eval_criterion, device)
        val_pat = patient_accuracy(val_loader.dataset, val_probs)
        epoch_time = time.time() - epoch_start

        for k, v in dict(train_loss=train_loss, train_acc=train_acc, val_loss=val_loss,
                         val_acc=val_acc, val_patient_acc=val_pat, lr=lr,
                         epoch_time=epoch_time).items():
            history[k].append(v)

        # ---- keep the best model (by val slice accuracy) ----
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc, stale = val_acc, 0
            with open(best_path, "wb") as f:                # File handle avoids a macOS torch.save bug
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                            "val_patient_acc": val_pat, "args": vars(args)}, f)
        else:
            stale += 1

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} pat {val_pat:.4f} | "
              f"lr {lr:.2e} | time {format_time(epoch_time)}"
              f"{'  * saved best' if improved else ''}")

        # ---- early stopping ----
        if stale >= PATIENCE:
            print(f"Early stop: no val improvement for {PATIENCE} epochs")
            break

    # ---- summary ----
    history["train_time"] = sum(history["epoch_time"])     # Seconds, sum of all epochs
    print(f"Total training time: {format_time(history['train_time'])} "
          f"({epoch} epochs, avg {format_time(np.mean(history['epoch_time']))}/epoch)")

    ckpt = torch.load(best_path, map_location=device)       # Best epoch's info, not the last
    return ckpt, history


# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Train GFNet on ADNI AD vs NC")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--model", choices=["gfnet", "simplecnn"], default="gfnet")   # Architecture to train
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--fold", type=int)                   # One CV fold (0-indexed)
    mode.add_argument("--cv", action="store_true")          # All CV folds
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--subset", type=float, default=FULL_SUBSET)    # Fraction of patients, e.g. 0.05
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-crop-flip", dest="crop_flip", action="store_false",
                    help="disable RandomResizedCrop and RandomHorizontalFlip")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Crop + flip augmentation: {'on' if args.crop_flip else 'off'}")

    loader_kw = dict(root=args.root, batch_size=args.batch_size, num_workers=args.workers,
                     seed=args.seed, subset=args.subset, crop_flip=args.crop_flip)

    # ---- cross-validation: one fold or all folds ----
    if args.cv or args.fold is not None:
        folds = range(DEFAULT_N_FOLDS) if args.cv else [args.fold]
        results, histories = [], {}
        for k in folds:
            print(f"\n========== fold {k}/{DEFAULT_N_FOLDS - 1} ==========")
            train_loader, val_loader, _ = get_fold_dataloaders(
                k, n_folds=DEFAULT_N_FOLDS, **loader_kw
            )
            print(f"Slices — train {len(train_loader.dataset)}, val {len(val_loader.dataset)}")
            fold_dir = os.path.join(args.out, f"fold{k}")
            ckpt, history = run_training(args, train_loader, val_loader, fold_dir, device)
            res = {"fold": k, "best_epoch": ckpt["epoch"], "val_acc": ckpt["val_acc"],
                   "val_patient_acc": ckpt["val_patient_acc"],
                   "train_time": history["train_time"]}
            save_json({**history, **res}, os.path.join(fold_dir, "history.json"))
            print(f"Fold {k}: best epoch {res['best_epoch']}, val acc {res['val_acc']:.4f}, "
                  f"val patient acc {res['val_patient_acc']:.4f}, "
                  f"time {format_time(res['train_time'])}")
            results.append(res)
            histories[k] = history

        if len(results) > 1:
            accs = np.array([r["val_acc"] for r in results])
            pats = np.array([r["val_patient_acc"] for r in results])
            summary = {"folds": results, "crop_flip": args.crop_flip,
                       "val_acc_mean": accs.mean(), "val_acc_std": accs.std(),
                       "val_patient_acc_mean": pats.mean(), "val_patient_acc_std": pats.std(),
                       "total_time": sum(r["train_time"] for r in results)}
            save_json(summary, os.path.join(args.out, "cv_summary.json"))
            print(f"\nCV val slice acc:   {accs.mean():.4f} ± {accs.std():.4f}")
            print(f"CV val patient acc: {pats.mean():.4f} ± {pats.std():.4f}")
            print(f"CV total time:      {format_time(summary['total_time'])} (sum of all folds)")
            print(f"Saved CV summary to {args.out}/cv_summary.json")

            # ---- save the best fold's model (with its history) as cv_best.pt ----
            best = max(results, key=lambda r: r["val_acc"])
            ckpt = torch.load(os.path.join(args.out, f"fold{best['fold']}", "best.pt"),
                              map_location="cpu")
            ckpt["fold"], ckpt["history"] = best["fold"], dict(histories[best["fold"]])
            cv_ckpt = os.path.join(args.out, CV_CKPT)
            with open(cv_ckpt, "wb") as f:
                torch.save(ckpt, f)
            print(f"Best fold: {best['fold']} (val acc {best['val_acc']:.4f}) -> saved {cv_ckpt}")
            print(f"Evaluate on test with:  python predict.py --ckpt {cv_ckpt}")
        return                                              

    # ---- single train/val split: train, keep the best epoch on val ----
    train_loader, val_loader, _ = get_dataloaders(**loader_kw)
    print(f"Slices — train {len(train_loader.dataset)}, val {len(val_loader.dataset)}")
    ckpt, history = run_training(args, train_loader, val_loader, args.out, device,
                                 ckpt_name=NO_CV_CKPT)

    print(f"\nBest epoch {ckpt['epoch']} (val acc {ckpt['val_acc']:.4f}, "
          f"val patient acc {ckpt['val_patient_acc']:.4f})")
    save_json({**history, "best_epoch": ckpt["epoch"]}, os.path.join(args.out, "history.json"))
    print(f"Saved {args.out}/{NO_CV_CKPT} and history.json")
    print(f"Evaluate on test with:  python predict.py --ckpt {args.out}/{NO_CV_CKPT}")


if __name__ == "__main__":       
    main()