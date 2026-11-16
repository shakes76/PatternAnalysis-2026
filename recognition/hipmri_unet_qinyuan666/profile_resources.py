"""Measure a specified tensor workload, not clinical performance or total VRAM.

Run each model in a fresh process for comparable CUDA allocator measurements.
Random tensors and synthetic labels isolate model compute from NIfTI loading.
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from metrics import segmentation_loss
from modules import build_model
from train import benchmark_inference, select_device, set_seed, synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("unet", "improved_unet"), required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.image_size < 32 or min(args.batch_size, args.base_channels) < 1:
        parser.error("image size >=32 and positive batch/width required")
    set_seed(3710)
    torch.set_num_threads(4)
    device = select_device(args.device)
    model = build_model(args.model, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    images = torch.randn(args.batch_size, 1, args.image_size, args.image_size, device=device)
    targets = torch.zeros(args.batch_size, args.image_size, args.image_size,
                          dtype=torch.long, device=device)
    targets[:, args.image_size // 3:2 * args.image_size // 3,
            args.image_size // 3:2 * args.image_size // 3] = 1

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = segmentation_loss(model(images), targets)
        loss.backward()
        optimizer.step()

    model.train()
    for _ in range(3):
        step()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(10):
        start = time.perf_counter()
        step()
        synchronize(device)
        durations.append(1000 * (time.perf_counter() - start))
    allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    result = {
        "data_kind": "random_tensor_resource_probe", "clinical_performance_claim": False,
        "model": args.model, "image_size": args.image_size, "batch_size": args.batch_size,
        "base_channels": args.base_channels, "num_classes": 2, "dtype": "float32",
        "parameters": sum(p.numel() for p in model.parameters()),
        "optimizer": "Adam, lr=0.001", "train_warmup": 3, "train_repeats": 10,
        "median_training_step_ms": statistics.median(durations),
        "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
        "memory_scope": "PyTorch allocator, model/Adam states/gradients/activations; not total device VRAM",
        "inference": benchmark_inference(model, images[:1], device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "limitations": "Tensor workload only. Two output classes. Real data I/O and multiclass cost not measured.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
