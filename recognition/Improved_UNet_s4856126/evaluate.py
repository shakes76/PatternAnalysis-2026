"""Evaluate Basic 2D U-Net or 2D CAN on identical HipMRI slices.

Outputs dataset-aggregated Dice/IoU by class, mean foreground Dice,
trainable parameter count, per-image inference time and peak allocated
CUDA memory. The independent Test split must not be used to tune models.

Usage on Rangpur (inside a Slurm GPU allocation):
    python evaluate.py --model baseline --split test --device cuda
    python evaluate.py --model can --split test --device cuda

Quick CPU sanity check (NOT final results):
    python evaluate.py --model can --split validate --max-samples 2 --device cpu
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import HipMRIDataset
from modules import CAN2D, UNet2D


NUM_CLASSES = 6
CHECKPOINTS = {
    "baseline": Path("checkpoints/baseline_unet.pt"),
    "can": Path("checkpoints/can_best.pt"),
}


def load_model(model_name, checkpoint_path, device):
    """Load a checkpoint saved by the COMP3710 baseline or CAN trainer.

    Bypasses predict.load_model(), which originally supported only the
    baseline / residual U-Net and might not recognise CAN.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError(f"Invalid checkpoint dictionary: {checkpoint_path}")
    if "model_state_dict" not in saved or "base_channels" not in saved:
        raise ValueError("Checkpoint must have model_state_dict and base_channels")

    # Old baseline checkpoints have no model_type field; CAN checkpoints do.
    declared_type = saved.get("model_type", "unet")
    expected_type = "unet" if model_name == "baseline" else "can"
    if declared_type != expected_type:
        raise ValueError(
            f"Wrong checkpoint architecture: checkpoint={declared_type}, "
            f"requested={expected_type}"
        )

    model_class = UNet2D if model_name == "baseline" else CAN2D
    model = model_class(
        in_channels=1,
        out_channels=NUM_CLASSES,
        base_channels=int(saved["base_channels"]),
    )
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def evaluate(model, loader, device):
    """Compute dataset-aggregated per-class Dice/IoU and resources.

    Inference timer retains the same behaviour as the earlier evaluation:
    times forward + argmax, synchronized for CUDA; excludes data loading.
    VRAM refers to peak torch CUDA *allocated* memory, not nvidia-smi use.
    """
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    total_inference_time = 0.0
    total_images = 0

    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for images, targets in loader:
        images = images.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        logits = model(images)
        predictions = logits.argmax(dim=1)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_inference_time += time.perf_counter() - start
        total_images += images.shape[0]

        predictions = predictions.cpu().long()
        targets = targets.cpu().long()
        flat = targets.reshape(-1) * NUM_CLASSES + predictions.reshape(-1)
        counts = torch.bincount(flat, minlength=NUM_CLASSES * NUM_CLASSES)
        confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES)

    results = []
    for label in range(NUM_CLASSES):
        tp = confusion[label, label].item()
        gt_pixels = confusion[label, :].sum().item()
        pred_pixels = confusion[:, label].sum().item()

        dice_denom = gt_pixels + pred_pixels
        iou_denom = gt_pixels + pred_pixels - tp
        results.append({
            "label": label,
            "ground_truth_pixels": gt_pixels,
            "predicted_pixels": pred_pixels,
            "dice": 2 * tp / dice_denom if dice_denom > 0 else None,
            "iou": tp / iou_denom if iou_denom > 0 else None,
        })

    foreground_dice = [
        row["dice"] for row in results[1:] if row["dice"] is not None
    ]
    mean_dice = (
        sum(foreground_dice) / len(foreground_dice)
        if foreground_dice else None
    )
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )
    return {
        "num_images": total_images,
        "mean_foreground_dice": mean_dice,
        "seconds_per_image": (
            total_inference_time / total_images if total_images else None
        ),
        "peak_gpu_allocated_mb": peak_memory_mb,
        "per_class": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["baseline", "can"], default="baseline"
    )
    parser.add_argument(
        "--split", choices=["validate", "test"], default="validate"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Override the default checkpoint for the selected model",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run in a Slurm GPU job")

    checkpoint_path = args.checkpoint or CHECKPOINTS[args.model]
    model = load_model(args.model, checkpoint_path, device)

    dataset = HipMRIDataset(split=args.split)
    full_dataset_size = len(dataset)
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print("Model:", args.model, flush=True)
    print("Checkpoint:", checkpoint_path, flush=True)
    print("Split:", args.split, flush=True)
    print("Samples:", len(dataset), flush=True)
    print("Device:", device, flush=True)

    result = evaluate(model, loader, device)
    result.update({
        "model": args.model,
        "split": args.split,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "full_dataset_size": full_dataset_size,
        "is_partial": len(dataset) != full_dataset_size,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    })

    print("\nEvaluation results:")
    for row in result["per_class"]:
        dice = row["dice"]
        iou = row["iou"]
        dice_text = f"{dice:.4f}" if dice is not None else "N/A"
        iou_text = f"{iou:.4f}" if iou is not None else "N/A"
        print(f"Label {row['label']}: Dice={dice_text}, IoU={iou_text}")

    print("Mean foreground Dice:", result["mean_foreground_dice"])
    print("Inference seconds/image:", result["seconds_per_image"])
    print("Peak allocated GPU memory (MB):", result["peak_gpu_allocated_mb"])
    print("Trainable parameters:", result["parameters"])
    if result["is_partial"]:
        print("PARTIAL CHECK ONLY: not a final evaluation.")

    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"{args.model}_{args.split}_{args.device}"
        + ("_partial" if result["is_partial"] else "")
    )
    json_path = output_dir / f"{suffix}.json"
    csv_path = output_dir / f"{suffix}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label", "ground_truth_pixels", "predicted_pixels", "dice", "iou"
            ],
        )
        writer.writeheader()
        writer.writerows(result["per_class"])
    print("Saved:", json_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
