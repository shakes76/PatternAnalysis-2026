"""
Loads saved best.pt and computes test-set Dice, IoU, VRAM, and latency.

Usage:
    python evaluate.py --model unet2d --weights ./runs/unet2d_baseline/best.pt --out-json ./runs/unet2d_baseline/eval.json
"""
import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn

from dataset import build_dataloaders, NUM_CLASSES
from modules import UNet2D, ImprovedUNet2D

CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    
    dice_sum = np.zeros(NUM_CLASSES)
    iou_sum = np.zeros(NUM_CLASSES)
    total_slices = 0
    total_time = 0.0

    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        # init timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        logits = model(imgs)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
        total_slices += imgs.size(0)

        # netrics
        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3)
        inter = (probs * lbls).sum(dims).cpu().numpy()
        pred_sum = probs.sum(dims).cpu().numpy()
        gt_sum = lbls.sum(dims).cpu().numpy()
        
        dice_sum += ((2 * inter + 1e-6) / (pred_sum + gt_sum + 1e-6)) * imgs.size(0)
        iou_sum += ((inter + 1e-6) / (pred_sum + gt_sum - inter + 1e-6)) * imgs.size(0)

    mean_inference_ms = (total_time / total_slices) * 1000.0
    return dice_sum / total_slices, iou_sum / total_slices, mean_inference_ms

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model == "unet2d":
        model = UNet2D(in_channels=1, out_channels=NUM_CLASSES).to(device)
    elif args.model == "improved_unet2d":
        model = ImprovedUNet2D(in_channels=1, out_channels=NUM_CLASSES).to(device)
    else:
        raise ValueError(f"unknown model: {args.model}")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")
    
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Loaded weights from {args.weights}")

    loaders = build_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("\nEvaluating on test set...")
    test_dice, test_iou, latency_ms = evaluate(model, loaders["test"], device)

    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_vram_gb = 0.0

    print(f"\n{'='*55}")
    print(f"EVALUATION SUMMARY: {args.model}")
    print(f"{'='*55}")
    print(f"{'Class':12s} {'Test Dice':>12s} {'Test IoU':>12s}")
    print(f"{'-'*55}")
    for i, n in enumerate(CLASS_NAMES):
        print(f"{n:12s} {test_dice[i]:>12.4f} {test_iou[i]:>12.4f}")
    print(f"{'-'*55}")
    print(f"{'mean':12s} {test_dice.mean():>12.4f} {test_iou.mean():>12.4f}")
    
    print(f"\nOperational Constraints:")
    print(f"  Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"  Inference latency: {latency_ms:.2f} ms/slice")

    # save to JSON file
    out_data = {
        "model": args.model,
        "weights": args.weights,
        "test_dice": test_dice.tolist(),
        "test_iou": test_iou.tolist(),
        "peak_vram_gb": float(peak_vram_gb),
        "latency_ms": float(latency_ms)
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved metrics to {args.out_json}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["unet2d", "improved_unet2d"])
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-json", type=str, required=True)
    args = p.parse_args()
    main(args)