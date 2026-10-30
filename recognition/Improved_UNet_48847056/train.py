"""
train.py - train, validate, test and save a 2D or 3D U-Net on HipMRI 3D volumes.

Both models use the same patient-level split (dataset.get_files_3d), and validation/
testing is always done on whole volumes for a more accurate comparison

Usage (from this folder):
    python train.py --dims 2 --epochs 30
    python train.py --root PATH/TO/HipMRI_Study_open --dims 2 --epochs 30 --batch 16
    python train.py --root PATH/TO/HipMRI_Study_open --dims 3 --epochs 60 --batch 2
Add --early-stop for a quick smoke test on a handful of volumes.
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")  # save plots without a display
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import NUM_CLASSES, get_data_3d
from modules import UNet, DiceCELoss, predict_volume

# Label index -> name 
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
PROSTATE = 5 # target
RANGPUR_ROOT = "/home/groups/comp3710/HipMRI_Study_open"  # Rangpur data path


def build(dims, base):
    return UNet(in_ch=1, num_classes=NUM_CLASSES, base=base, depth=4, dims=dims)


@torch.no_grad()
def evaluate(model, vols, labels, dims, device, amp):
    """Score whole volumes. Returns a dict with

    dice/iou: per-class, pooled over all voxels of all volumes
    vol_prostate_dice: prostate Dice for each volume separately
    slice_prostate_dice: list per volume of per-slice prostate Dice (NaN where the
        slice has no prostate in prediction or truth) - used for base/apex analysis
    sec_per_volume: mean inference time per volume

    this works for both models. predict_volume runs 2d model slice by slice and a 3d model on the complete volume, 
    so both are scored on identical voxels
    """
    model.eval()

    inter = np.zeros(NUM_CLASSES)
    psum = np.zeros(NUM_CLASSES)
    tsum = np.zeros(NUM_CLASSES)
    vol_dice, slice_dice, times = [], [], []

    for v, y in zip(vols, labels):
        x = torch.from_numpy(v).to(device)
        yt = torch.from_numpy(y.astype(np.int64)).to(device)

        # sync before and after GPU calls so the timer measures the actual computation
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device.type, enabled=amp):
            pred = predict_volume(model, x, dims).argmax(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        # accumulate overlap counts for each class (correct predictions)
        for c in range(NUM_CLASSES):
            p, t = pred == c, yt == c
            inter[c] += (p & t).sum().item()
            psum[c] += p.sum().item()
            tsum[c] += t.sum().item()

        # Prostate Dice for this patient
        p, t = pred == PROSTATE, yt == PROSTATE
        i, s = (p & t).sum().item(), p.sum().item() + t.sum().item()
        vol_dice.append(2 * i / s if s > 0 else float("nan"))

        # Prostate Dice 
        i_z = (p & t).sum((0, 1)).float()  # per slice (last axis)
        s_z = (p.sum((0, 1)) + t.sum((0, 1))).float()
        slice_dice.append(torch.where(s_z > 0, 2 * i_z / s_z, torch.nan).cpu().tolist())

    # ignore classes that are absent from every volume
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = 2 * inter / (psum + tsum)
        iou = inter / (psum + tsum - inter)

    # save as plain python types for JSON compatability
    return {"dice": dice.tolist(), "iou": iou.tolist(), "vol_prostate_dice": vol_dice,
            "slice_prostate_dice": slice_dice, "sec_per_volume": float(np.mean(times))}


def plot_history(hist, path, model_name=""):
    """Save training curves: loss (left) and validation Dice (right) per epoch.

    model_name (e.g. "unet2d") is shown in the overall figure title, so the 2D and
    3D curve images can be told apart in the README.
    """
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"], label="train loss")
    ax[0].set_title("Training loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("Dice + CE loss"); ax[0].legend()
    ax[1].plot(hist["val_prostate_dice"], label="val prostate Dice")
    ax[1].plot(hist["val_mean_dice"], label="val mean Dice (no background)")
    ax[1].set_title("Validation Dice")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("Dice"); ax[1].set_ylim(0, 1); ax[1].legend()
    if model_name:
        fig.suptitle(f"{model_name} training curves")  # title over both panels
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ## COMMAND LINE OPTIONS - this was mostly claude ##
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=RANGPUR_ROOT, help="HipMRI_Study_open folder (default: Rangpur path)")
    ap.add_argument("--dims", type=int, choices=(2, 3), required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=None, help="default 16 for 2D, 2 for 3D")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16, help="channels at the first U-Net level")
    ap.add_argument("--size", type=int, nargs=3, default=(128, 128, 64), help="resampled H W D")
    ap.add_argument("--flip-axis", type=int, default=None,
                    help="left-right axis for flip augmentation (verify with a plot first); off by default")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--early-stop", action="store_true", help="smoke test on a few volumes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()


    ## SETUP ##
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" # mixed precision for GPU
    # a 2D batch is 16 slices, 3D batch is 2 volumes. Should balance memory but the 3D batches are still bigger
    batch = args.batch or (16 if args.dims == 2 else 2)
    name = f"unet{args.dims}d"
    os.makedirs(args.out, exist_ok=True)
    print(f"device {device} | model {name} | batch {batch}")

    # Training data comes as a DataLoader (with 2D slices or 3D vols)
    train_loader, (vx, vy), (tx, ty), test_files = get_data_3d(
        args.root, args.dims, size=tuple(args.size), batch_size=batch,
        flip_axis=args.flip_axis, early_stop=args.early_stop)

    model = build(args.dims, args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")
    loss_fn = DiceCELoss(NUM_CLASSES)
    # AdamW - adam with decoupled weight decay (light regularisation)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # cosine learning rate scheduler 
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # gradscaler to ensure small float16 gradients do not underflow to 0. unscales before optimiser step
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats() # for measuring VRAM

    hist = {"train_loss": [], "val_prostate_dice": [], "val_mean_dice": [], "epoch_sec": []}
    best, ckpt = -1.0, os.path.join(args.out, f"{name}_best.pt")

    ### TRAINING LOOOOP ###
    for epoch in range(args.epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True) # clear last step's gradients
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(model(x), y) # fowrward pass + Dice/CE loss
            scaler.scale(loss).backward() #backprop on scaled loss
            scaler.step(opt) # unscale grads
            scaler.update() # adjust scale factor for next step
            # loss is weighted by batch size
            total += loss.item() * len(x)
            n += len(x)
        sched.step() # once per epoch


        # validation for the whole volume #
        val = evaluate(model, vx, vy, args.dims, device, amp)
        pd = val["dice"][PROSTATE]
        md = float(np.nanmean(val["dice"][1:]))
        hist["train_loss"].append(total / n)
        hist["val_prostate_dice"].append(pd)
        hist["val_mean_dice"].append(md)
        hist["epoch_sec"].append(time.time() - t0)

        # model selection with the validation set only
        flag = ""
        if pd > best:  # keep the weights with the best validation prostate Dice
            best = pd
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
            flag = " *saved*" # for debugging
        # logs to output file
        print(f"epoch {epoch + 1:3d}/{args.epochs} loss {total / n:.4f} "
              f"val prostate Dice {pd:.4f} mean Dice {md:.4f} ({hist['epoch_sec'][-1]:.0f}s){flag}")

        # redraw each epoch in case the run stops early
        plot_history(hist, os.path.join(args.out, f"{name}_curves.png"), name)

    # highest VRAM use during training
    peak_train_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")

    ## FINAL TEST w/ BEST CHECKPOINT ##
    # reload best validation weights
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    test = evaluate(model, tx, ty, args.dims, device, amp)
    peak_infer_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")

    print("\nTEST (best checkpoint)")
    for c, cname in enumerate(CLASS_NAMES):
        print(f"  {cname:10s} Dice {test['dice'][c]:.4f}  IoU {test['iou'][c]:.4f}")
    print(f"  per-volume prostate Dice: mean {np.nanmean(test['vol_prostate_dice']):.4f} "
          f"min {np.nanmin(test['vol_prostate_dice']):.4f}")
    print(f"  params {n_params:,} | peak VRAM train {peak_train_mb:.0f} MB, inference {peak_infer_mb:.0f} MB "
          f"| {test['sec_per_volume'] * 1000:.1f} ms/volume")

    # save required results to JSON
    results = {"model": name, "params": n_params, "peak_vram_train_mb": peak_train_mb,
               "peak_vram_infer_mb": peak_infer_mb, "mean_epoch_sec": float(np.mean(hist["epoch_sec"])),
               "test": test, "test_files": test_files, "history": hist, "args": vars(args)}
    with open(os.path.join(args.out, f"{name}_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved {ckpt}, curves and results json in {args.out}/")


if __name__ == "__main__":
    main()