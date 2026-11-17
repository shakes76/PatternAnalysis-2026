"""Visualise HipMRI segmentation predictions from Basic 2D U-Net or 2D CAN.

Examples:
    python predict.py --model baseline --split test --index 0 --device cpu
    python predict.py --model can --split test --index 0 --device cpu

Use the original dataset and the trained, validation-selected checkpoints.
MRI visualisations may be subject to dataset redistribution restrictions.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import HipMRIDataset
from modules import CAN2D, UNet2D


NUM_CLASSES = 6
CHECKPOINTS = {
    "baseline": Path("checkpoints/baseline_unet.pt"),
    "can": Path("checkpoints/can_best.pt"),
}


def load_model(model_name, checkpoint_path, device="cpu"):
    """Load a matching architecture and its saved validation-selected weights.

    Compatible with the baseline checkpoint (which may lack model_type)
    and the CAN checkpoint produced by train_can.py.
    """
    if model_name not in CHECKPOINTS:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint is not a dictionary")

    # Both verified training pipelines save these fields.
    if "model_state_dict" not in saved:
        raise ValueError("Missing model_state_dict in checkpoint")

    expected_type = "unet" if model_name == "baseline" else "can"
    declared_type = saved.get("model_type", "unet")
    if declared_type != expected_type:
        raise ValueError(
            f"Checkpoint architecture is '{declared_type}', "
            f"but requested model is '{model_name}'"
        )

    model_class = UNet2D if model_name == "baseline" else CAN2D
    model = model_class(
        in_channels=1,
        out_channels=NUM_CLASSES,
        base_channels=int(saved.get("base_channels", 16)),
    )
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model = model.to(torch.device(device))
    model.eval()

    print("Model loaded:", model_name)
    print("Checkpoint:", checkpoint_path)
    if "epoch" in saved:
        print("Checkpoint epoch:", saved["epoch"])
    return model


def predict_sample(model, dataset, index):
    """Predict one MRI slice, returning image, GT mask and predicted labels."""
    image, ground_truth = dataset[index]
    device = next(model.parameters()).device

    model.eval()
    with torch.inference_mode():
        logits = model(image.unsqueeze(0).to(device))
        prediction = logits.argmax(dim=1).squeeze(0)

    return (
        image.squeeze(0).cpu().numpy(),
        ground_truth.cpu().numpy(),
        prediction.cpu().numpy(),
    )


def save_visualisation(image, ground_truth, prediction, path):
    """Save MRI, six-class GT, six-class prediction and prostate contours."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), constrained_layout=True)

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("MRI")

    axes[1].imshow(ground_truth, cmap="tab10", vmin=0, vmax=9)
    axes[1].set_title("Ground Truth (6 classes)")

    axes[2].imshow(prediction, cmap="tab10", vmin=0, vmax=9)
    axes[2].set_title("Prediction (6 classes)")

    axes[3].imshow(image, cmap="gray")
    gt = ground_truth == 5
    pred = prediction == 5
    if np.any(gt):
        axes[3].imshow(
            np.ma.masked_where(~gt, gt.astype(float)),
            cmap="Reds", vmin=0, vmax=1, alpha=0.45,
        )
        if not np.all(gt):
            axes[3].contour(
                gt.astype(float), levels=[0.5], colors="red", linewidths=1,
            )
    if np.any(pred) and not np.all(pred):
        axes[3].contour(
            pred.astype(float), levels=[0.5], colors="lime", linewidths=1,
        )
    axes[3].set_title("Prostate: GT red / Pred green")

    for ax in axes:
        ax.axis("off")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("Saved visualisation:", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["baseline", "can"], default="baseline")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["validate", "test"], default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/predictions")
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: use a Slurm GPU allocation or --device cpu")

    checkpoint_path = args.checkpoint or CHECKPOINTS[args.model]
    model = load_model(args.model, checkpoint_path, device=device)

    dataset = HipMRIDataset(split=args.split)
    if not 0 <= args.index < len(dataset):
        raise ValueError(
            f"Sample index {args.index} outside {args.split} split "
            f"(0 to {len(dataset) - 1})"
        )

    image, gt, pred = predict_sample(model, dataset, args.index)
    print("MRI shape:", image.shape)
    print("GT labels:", np.unique(gt).tolist())
    print("Predicted labels:", np.unique(pred).tolist())

    output_path = args.output_dir / f"{args.model}_{args.split}_{args.index}.png"
    save_visualisation(image, gt, pred, output_path)
    print("Prediction complete.")


if __name__ == "__main__":
    main()
