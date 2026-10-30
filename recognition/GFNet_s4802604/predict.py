"""Run a trained model on the val or test split and save per-scan probabilities.

Example:
  python predict.py --ckpt ~/adni_work/gfnet_clean_r5_stem_s42.pth --model gfnet --conv-stem --split test

The scan probability is the mean of the probabilities of its slices (same as train.py).
Also prints parameter count, inference time and peak GPU memory for the README resource table.
"""
import argparse
import json
import os
import time

import torch

from dataset import WORK, get_loaders
from modules import BaselineCNN, GFNet
from train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to a .pth state dict saved by train.py")
    ap.add_argument("--model", default="gfnet", choices=["baseline", "gfnet"])
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--conv-stem", action="store_true")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.path.expanduser(args.ckpt)
    name = os.path.basename(ckpt).replace(".pth", "")

    if args.model == "gfnet":
        model = GFNet(dim=args.dim, depth=args.depth, patch=args.patch, conv_stem=args.conv_stem)
    else:
        model = BaselineCNN()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    _, val_dl, test_dl = get_loaders(args.batch_size)
    loader = val_dl if args.split == "val" else test_dl

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    metrics, scan_p, labels = evaluate(model, loader, device)
    if device == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")

    out = f"{WORK}/{name}_{args.split}_preds.json"
    json.dump({"metrics": metrics, "scan_prob": scan_p, "scan_label": labels}, open(out, "w"))
    n_scans = len(scan_p)
    print(f"{name} [{args.split}] {json.dumps(metrics)}")
    print(f"params {n_params:,} | {n_scans} scans in {secs:.1f}s ({1000 * secs / n_scans:.0f} ms/scan, "
          f"incl. data loading) | peak GPU memory {peak_mb:.0f} MiB at batch {args.batch_size}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
