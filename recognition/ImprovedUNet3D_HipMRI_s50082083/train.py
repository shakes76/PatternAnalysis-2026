"""
Train, validate and test a segmentation model on HipMRI.

Examples (Colab):
    # 2D U-Net baseline on full-resolution axial slices
    python train.py --data-root /content/hipmri3d --cache-dir /content/cache \
        --out-dir /content/drive/MyDrive/comp3710_runs/unet2d --model unet2d --epochs 30
    # 3D Improved U-Net on whole volumes downsampled x2
    python train.py --data-root /content/hipmri3d --cache-dir /content/cache \
        --out-dir /content/drive/MyDrive/comp3710_runs/iunet3d --model iunet3d \
        --epochs 150 --batch-size 2 --lr 5e-4

Each epoch: train on the training patients, evaluate prostate Dice on the
validation patients at full resolution, save `last.pt` (for resuming) and
`best.pt` (best validation prostate Dice). After the last epoch the best
checkpoint is evaluated ONCE on the test patients.
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from dataset import (HipMRISlices2D, HipMRIVolumes3D, case_ids_for, find_cases,
                     preprocess, split_patients)
from modules import DiceCELoss, build_model
from utils import (evaluate_cases, plot_history, print_summary, save_json,
                   seed_everything, summarise)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="folder containing the HipMRI .nii.gz files")
    p.add_argument("--cache-dir", default="cache", help="where preprocessed .npy volumes are stored")
    p.add_argument("--out-dir", required=True, help="checkpoints, curves and metrics go here")
    p.add_argument("--model", default="unet2d")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--depth", type=int, default=128, help="slices per volume after crop/pad")
    p.add_argument("--scale", type=int, default=2, help="3D models: downsampling factor per axis")
    p.add_argument("--prostate-fraction", type=float, default=0.5,
                   help="share of training slices sampled from slices containing prostate")
    p.add_argument("--n-val", type=int, default=6, help="validation patients")
    p.add_argument("--n-test", type=int, default=6, help="test patients")
    p.add_argument("--limit-train-patients", type=int, default=0,
                   help="use only N training patients (quick smoke tests); 0 = all")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--resume", action="store_true", help="continue from out-dir/last.pt")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}  |  mixed precision: {amp}")

    # ---- data: find, preprocess once, split by patient ----
    cases = find_cases(args.data_root)
    preprocess(cases, args.cache_dir, depth=args.depth)
    train_p, val_p, test_p = split_patients([c["patient"] for c in cases],
                                            n_val=args.n_val, n_test=args.n_test, seed=args.seed)
    if args.limit_train_patients:
        train_p = train_p[:args.limit_train_patients]
    train_ids, val_ids, test_ids = (case_ids_for(cases, p) for p in (train_p, val_p, test_p))
    save_json({"train": train_p, "val": val_p, "test": test_p,
               "train_cases": train_ids, "val_cases": val_ids, "test_cases": test_ids},
              os.path.join(args.out_dir, "split.json"))
    print(f"Patients train/val/test: {len(train_p)}/{len(val_p)}/{len(test_p)}  |  "
          f"scans: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    # ---- model and training data ----
    model, mode = build_model(args.model)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model {args.model} ({mode}): {n_params:,} parameters")

    if mode == "2d":
        train_ds = HipMRISlices2D(args.cache_dir, train_ids, augment=True)
        sampler = WeightedRandomSampler(train_ds.sample_weights(args.prostate_fraction),
                                        num_samples=len(train_ds), replacement=True)
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.workers, pin_memory=device.type == "cuda",
                            drop_last=True, persistent_workers=args.workers > 0)
    else:   # 3D: whole downsampled volumes, shuffled each epoch
        train_ds = HipMRIVolumes3D(args.cache_dir, train_ids, scale=args.scale, augment=True)
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                            pin_memory=device.type == "cuda", drop_last=True)
    print(f"Training samples per epoch: {len(train_ds):,}")

    criterion = DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    # ---- resume after a Colab disconnect ----
    start_epoch, best_dice, history = 1, -1.0, []
    last_path, best_path = (os.path.join(args.out_dir, n) for n in ("last.pt", "best.pt"))
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_dice, history = ckpt["epoch"] + 1, ckpt["best_dice"], ckpt["history"]
        print(f"Resumed from epoch {ckpt['epoch']} (best val prostate Dice {best_dice:.3f})")

    # ---- training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0, running, steps = time.time(), 0.0, 0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
            loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            steps += 1
        scheduler.step()
        train_time = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

        val = summarise(evaluate_cases(model, mode, args.cache_dir, val_ids, device, amp=amp, scale=args.scale))
        record = {"epoch": epoch, "train_loss": running / max(steps, 1),
                  "val_prostate_dice": val["prostate_dice"],
                  "val_mean_fg_dice": val["mean_foreground_dice"],
                  "epoch_time_s": round(train_time, 1), "peak_vram_gb": round(peak_gb, 2),
                  "lr": optimizer.param_groups[0]["lr"]}
        history.append(record)
        print(f"Epoch {epoch:3d} | loss {record['train_loss']:.4f} | "
              f"val prostate Dice {val['prostate_dice']:.3f} | val organs Dice {val['mean_foreground_dice']:.3f} | "
              f"{train_time:.0f}s | peak VRAM {peak_gb:.2f} GB")

        config = {"model": args.model, "depth": args.depth, "scale": args.scale}
        if val["prostate_dice"] > best_dice:
            best_dice = val["prostate_dice"]
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                        "val_prostate_dice": best_dice}, best_path)
            print(f"           new best -> {best_path}")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_dice": best_dice, "history": history, "config": config},
                   last_path)
        save_json(history, os.path.join(args.out_dir, "history.json"))
        plot_history(history, os.path.join(args.out_dir, "curves.png"))

    # ---- final test: best checkpoint, run once ----
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_results = evaluate_cases(model, mode, args.cache_dir, test_ids, device, amp=amp, scale=args.scale)
    test_summary = summarise(test_results)
    print_summary(test_summary, f"TEST — {args.model}, best epoch {ckpt['epoch']}")
    save_json({"summary": test_summary, "per_case": test_results, "best_epoch": ckpt["epoch"],
               "parameters": n_params}, os.path.join(args.out_dir, "test_metrics.json"))


if __name__ == "__main__":
    main()
