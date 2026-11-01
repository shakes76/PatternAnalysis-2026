"""Benchmark one model: parameters, peak GPU memory, training speed, inference time."""
import argparse
import json
import time

import torch

from dataset import get_dataloaders
from modules import DiceLoss
from train import MODELS, build_model, get_device, set_seed, train_one_epoch


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--model", choices=MODELS, default="unet")
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    device = get_device()
    set_seed(args.seed)
    train_loader, _, _ = get_dataloaders(args.root, batch_size=args.batch_size,
                                     num_workers=args.num_workers)
    model = build_model(args.model, args.base_channels).to(device)
    loss_fn = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Warm-up steps are not timed (first steps include one-off start-up costs).
    train_one_epoch(model, train_loader, optimizer, loss_fn, device, max_batches=args.warmup)
    sync(device)
    start = time.perf_counter()
    train_one_epoch(model, train_loader, optimizer, loss_fn, device, max_batches=args.batches)
    sync(device)
    sec_per_batch = (time.perf_counter() - start) / args.batches

    # Inference time for one slice.
    model.eval()
    x = torch.randn(1, 1, 256, 256, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        sync(device)
        start = time.perf_counter()
        for _ in range(100):
            model(x)
        sync(device)
    infer_ms = (time.perf_counter() - start) / 100 * 1000

    result = {
        "model": args.model,
        "base_channels": args.base_channels,
        "batch_size": args.batch_size,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_memory_gb": (torch.cuda.max_memory_allocated() / 1024**3
                           if device.type == "cuda" else None),
        "sec_per_batch": sec_per_batch,
        "est_min_per_epoch": sec_per_batch * len(train_loader) / 60,
        "inference_ms_per_slice": infer_ms,
    }
    print(json.dumps(result, indent=2))
    with open(f"benchmark_{args.model}_b{args.base_channels}.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()