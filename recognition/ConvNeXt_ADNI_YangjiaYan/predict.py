"""Show checkpoint predictions and measure batch-one GPU inference cost."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import ADNIDataset
from modules import build_model, count_parameters
from train import load_manifest


def profile_model(model, image, warmup=20, repeats=100):
    """Measure GPU forward plus softmax, excluding data loading and transfer."""
    with torch.inference_mode():
        for _ in range(warmup):
            probabilities = model(image).softmax(dim=1)
            del probabilities
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        timings = []

        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            probabilities = model(image).softmax(dim=1)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
            del probabilities

    return {
        "batch_size": 1,
        "input_shape": list(image.shape),
        "precision": "float32",
        "warmup_iterations": warmup,
        "timed_iterations": repeats,
        "timing_scope": "GPU forward and softmax; excludes image IO and transfer",
        "latency_mean_ms": float(np.mean(timings)),
        "latency_median_ms": float(np.median(timings)),
        "latency_p95_ms": float(np.percentile(timings, 95)),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "memory_scope": (
            "Inference only; allocated memory includes model, input and "
            "temporary tensors. Reserved memory includes allocator cache."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path.home() / "data/comp3710_a3_adni_split.csv",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    if args.examples < 1:
        parser.error("--examples must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Run predict.py inside a GPU Slurm job.")

    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True,
    )
    model = build_model(
        checkpoint["model_name"], num_classes=2, pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to("cuda").eval()
    del checkpoint

    records = load_manifest(args.manifest)
    dataset = ADNIDataset(records, args.split)
    validation = ADNIDataset(records, "val")
    if not len(dataset) or not len(validation):
        raise ValueError("Example or validation dataset is empty.")

    # Both models are benchmarked on the same validation input.
    image, _ = validation[0]
    image = image.unsqueeze(0).to("cuda")
    profile = profile_model(model, image)
    total, trainable = count_parameters(model)
    profile.update({
        "model": args.checkpoint.parent.parent.name,
        "parameters": total,
        "trainable_parameters": trainable,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "benchmark_input_split": "val",
        "tf32_enabled": False,
        "cudnn_benchmark": False,
    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "resource_profile.json", "w") as f:
        json.dump(profile, f, indent=2, allow_nan=False)
    print(json.dumps(profile, indent=2), flush=True)

    # Select examples in manifest order, independently of model predictions.
    selected, seen = [], set()
    for index, record in enumerate(dataset.records):
        if record["patient_id"] not in seen:
            selected.append(index)
            seen.add(record["patient_id"])
        if len(selected) == args.examples:
            break

    fig, axes = plt.subplots(
        1, len(selected), figsize=(3.3 * len(selected), 4),
        squeeze=False,
    )
    outputs = []
    names = {0: "NC", 1: "AD"}

    with torch.inference_mode():
        for number, (ax, index) in enumerate(
            zip(axes[0], selected), start=1
        ):
            tensor, label = dataset[index]
            score = model(
                tensor.unsqueeze(0).to("cuda")
            ).softmax(dim=1)[0, 1].item()
            prediction = int(score >= 0.5)
            record = dataset.records[index]

            with Image.open(record["path"]) as original:
                ax.imshow(original.convert("L"), cmap="gray")
            ax.set_title(
                f"Example {number}\nTrue: {names[label]} | "
                f"Predicted: {names[prediction]}\nP(AD) = {score:.3f}",
                fontsize=10,
            )
            ax.axis("off")
            outputs.append({
                "example": number,
                "true_label": names[label],
                "predicted_label": names[prediction],
                "prob_ad": score,
            })

    fig.suptitle(f"Checkpoint inference examples: {args.split} slices")
    fig.text(
        0.5, 0.02,
        "One slice per patient, selected in manifest order. "
        "These are slice-level predictions.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(
        args.output_dir / "prediction_examples.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    with open(args.output_dir / "prediction_examples.json", "w") as f:
        json.dump(outputs, f, indent=2, allow_nan=False)

    print(json.dumps(outputs, indent=2), flush=True)
    print("Saved to:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
