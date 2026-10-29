"""Training, validation and testing for the HipMRI prostate segmentation models.

Usage:  python train.py --model unet2d --epochs 40
Evaluation is PER VOLUME: the predictions for the 128 slices of a scan are
pooled before Dice is computed, so 2D and 3D models can be compared fairly.
"""
import argparse
import csv
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import HipMRI2D, HipMRI3DCached, load_split, patient_of, random_flip
from modules import UNet2D, ImprovedUNet

NUM_CLASSES = 6   # 0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate
PROSTATE = 5


def build_model(name, base3d=16):
    """Create the model chosen on the command line (base3d = base filters of the 3D net)."""
    if name == "unet2d":
        return UNet2D()
    if name == "improved2d":
        return ImprovedUNet(dim=2, base=32, depth=5)
    if name == "improved3d":
        return ImprovedUNet(dim=3, base=base3d, depth=5)
    raise ValueError(f"unknown model: {name}")


def dice_loss(logits, target, eps=1e-6):
    """Soft Dice loss averaged over the foreground classes 1..5."""
    # class probabilities per voxel (soft values, so the loss is differentiable)
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, NUM_CLASSES).movedim(-1, 1).float()
    dims = (0,) + tuple(range(2, probs.ndim))                                  # pool over batch and all spatial axes
    # soft Dice per class = 2 * overlap / (predicted size + true size)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)          # one value per class
    # skip class 0 (background) so the huge, easy background does not dominate the loss
    return 1 - dice[1:].mean()


def combined_loss(logits, target):
    """Cross-entropy (stable gradients) + Dice (handles tiny organs)."""
    return F.cross_entropy(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def evaluate(model, ds, device, bs=16):
    """Per-volume Dice for each class: returns a [n_volumes, 6] tensor.

    Only valid when ds keeps every slice (empty_ratio=None), so that slices
    can be pooled back into whole volumes.
    """
    model.eval()
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    # running totals per volume and class: overlap, predicted size and true size
    n_vol = len(ds.cases)
    inter = torch.zeros(n_vol, NUM_CLASSES)
    psum = torch.zeros(n_vol, NUM_CLASSES)
    tsum = torch.zeros(n_vol, NUM_CLASSES)
    i0 = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()      # most likely class per pixel
        # which volume each slice of this batch belongs to
        ks = torch.tensor([ds.index[i0 + j][0] for j in range(len(y))])
        i0 += len(y)
        p = F.one_hot(pred, NUM_CLASSES)              # [B,H,W,6]
        t = F.one_hot(y, NUM_CLASSES)
        # add this batch's counts to the right volume: pooling the slices of a volume
        # gives the whole-volume Dice
        inter.index_add_(0, ks, (p & t).sum((1, 2)).float())
        psum.index_add_(0, ks, p.sum((1, 2)).float())
        tsum.index_add_(0, ks, t.sum((1, 2)).float())
    # Dice = 2 * overlap / (predicted + true), per volume and class
    return (2 * inter + 1e-6) / (psum + tsum + 1e-6)


@torch.no_grad()
def evaluate3d(model, ds, device):
    """Per-volume Dice for each class for a 3D model: returns [n_volumes, 6]."""
    model.eval()
    out = []
    for i in range(len(ds)):
        x, y = ds[i]
        # one whole volume at a time: add a batch axis, predict, then remove it again
        pred = model(x[None].to(device)).argmax(1)[0].cpu()
        p = F.one_hot(pred, NUM_CLASSES)
        t = F.one_hot(y, NUM_CLASSES)
        inter = (p & t).sum((0, 1, 2)).float()
        out.append((2 * inter + 1e-6) /
                   (p.sum((0, 1, 2)).float() + t.sum((0, 1, 2)).float() + 1e-6))
    return torch.stack(out)


def plot_history(history, path):
    """Save the training-loss and validation-Dice curves to an image."""
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history])
    ax[0].set_title("Training loss (CE + Dice)"); ax[0].set_xlabel("epoch")
    ax[1].plot(ep, [h["val_prostate_dice"] for h in history], label="prostate")
    ax[1].plot(ep, [h["val_mean_dice"] for h in history], label="mean of classes 1-5")
    ax[1].set_title("Validation Dice (per volume)"); ax[1].set_xlabel("epoch")
    ax[1].set_ylim(0, 1); ax[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def main():
    """Train the chosen model, keep the checkpoint with the best validation prostate
    Dice, then evaluate that checkpoint once on the test split."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet2d")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--empty_ratio", type=float, default=0.25,
                    help="prostate-free slices kept per prostate slice (training only)")
    ap.add_argument("--max_cases", type=int, default=0,
                    help="debug: keep only N volumes per split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base3d", type=int, default=16, help="base filters of the 3D model")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. matched")
    args = ap.parse_args()

    # 3D models take one whole volume per step and use the volume-level evaluation
    is3d = args.model.endswith("3d")
    evaluate_fn = evaluate3d if is3d else evaluate
    if is3d:
        args.bs = 1          # whole volumes: one per step
    name = args.model + (f"_{args.tag}" if args.tag else "")   # output file prefix
    # fix all random number generators so that a run can be repeated
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # the patient-level split saved by dataset.py (identical for every model)
    split = load_split()
    if args.max_cases:
        kept, seen = {}, {}
        for c, s in sorted(split.items()):
            seen[s] = seen.get(s, 0) + 1
            if seen[s] <= args.max_cases:
                kept[c] = s
        split = kept

    print("loading data ...")
    train_ds = (HipMRI3DCached(split, "train") if is3d else
                HipMRI2D(split, "train", empty_ratio=args.empty_ratio, seed=args.seed))
    val_ds = HipMRI3DCached(split, "val") if is3d else HipMRI2D(split, "val", empty_ratio=None)
    # shuffle the training data; drop_last skips a final, smaller batch
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              drop_last=True, num_workers=0)
    print(f"train slices: {len(train_ds)} | val slices: {len(val_ds)}")

    model = build_model(args.model, args.base3d).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # Adam optimiser; the cosine schedule lowers the learning rate smoothly towards 0
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for d in ("checkpoints", "results", "images"):
        os.makedirs(d, exist_ok=True)
    ckpt = f"checkpoints/{name}_best.pth"
    # best = highest validation prostate Dice so far; those weights are saved as the checkpoint
    history, best = [], -1.0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x, y = random_flip(x, y)    # augmentation lives in dataset.py
            # one training step: forward pass, loss, backward pass, weight update
            loss = combined_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.shape[0]
            n += x.shape[0]
        sched.step()      # lower the learning rate for the next epoch

        # validation Dice per volume and class (no weight updates)
        dice = evaluate_fn(model, val_ds, device)
        row = {"epoch": epoch, "train_loss": total / n,
               "val_prostate_dice": dice[:, PROSTATE].mean().item(),
               "val_mean_dice": dice[:, 1:].mean().item(),
               "epoch_seconds": time.time() - t0}
        history.append(row)
        print(f"epoch {epoch:3d} | loss {row['train_loss']:.4f} | "
              f"val prostate Dice {row['val_prostate_dice']:.4f} | "
              f"val mean Dice {row['val_mean_dice']:.4f} | {row['epoch_seconds']:.0f}s")
        # the checkpoint is chosen on the VALIDATION split only; test data is never used for this
        if row["val_prostate_dice"] > best:
            best = row["val_prostate_dice"]
            torch.save(model.state_dict(), ckpt)

    train_time = time.time() - t_start
    with open(f"results/{name}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    plot_history(history, f"images/{name}_curves.png")

    # ---- final test with the best checkpoint ----
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_ds = HipMRI3DCached(split, "test") if is3d else HipMRI2D(split, "test", empty_ratio=None)
    dice = evaluate_fn(model, test_ds, device)
    with open(f"results/{name}_test_per_volume.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "patient"] + [f"dice_class{c}" for c in range(NUM_CLASSES)])
        for case, d in zip(test_ds.cases, dice.tolist()):
            w.writerow([case, patient_of(case)] + [round(v, 4) for v in d])

    print("\n===== TEST (per volume) =====")
    for c, name in enumerate(["background", "body", "bone", "bladder", "rectum", "prostate"]):
        print(f"class {c} ({name:10s}): mean Dice {dice[:, c].mean():.4f}")
    # average the prostate Dice over each test patient's scans
    by_patient = {}
    for case, d in zip(test_ds.cases, dice[:, PROSTATE].tolist()):
        by_patient.setdefault(patient_of(case), []).append(d)
    print("prostate Dice per patient:",
          {p: round(sum(v) / len(v), 3) for p, v in by_patient.items()})
    print(f"parameters: {n_params:,} | training time: {train_time / 60:.1f} min")
    if device.type == "cuda":      # peak GPU memory over the whole run
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
