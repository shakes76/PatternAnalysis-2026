"""Reload a checkpoint and save input/target/prediction/error examples locally.

Synthetic runs require an explicit flag and are labelled as software checks.
The displayed slices are examples, not a complete patient-level evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import HipMRIDataset, SyntheticSegmentationDataset, validate_label_values
from modules import build_model


def _device(name: str) -> torch.device:
    if name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available()
                        else "cpu" if name == "auto" else name)


def _validate_checkpoint(checkpoint: dict, synthetic: bool) -> tuple[dict, int]:
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Expected a version-1 project checkpoint")
    config = checkpoint.get("config")
    required = {"model", "base_channels", "image_size", "num_classes", "label_values",
                "synthetic", "prostate_label", "seed", "samples"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("Checkpoint config is missing required model/data fields")
    config = dict(config)
    config.setdefault("resize_mode", "stretch")  # Before this option, all real slices were stretched.
    if config["resize_mode"] not in ("pad", "stretch"):
        raise ValueError("Checkpoint resize_mode must be pad or stretch")
    if not isinstance(config["synthetic"], bool) or config["synthetic"] != synthetic:
        raise ValueError("Checkpoint data mode and --synthetic must match exactly")
    if (not isinstance(checkpoint.get("model_state"), dict)
            or not checkpoint["model_state"]):
        raise ValueError("Checkpoint needs a nonempty model_state dictionary")
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("Checkpoint epoch must be a positive integer")
    if config["model"] not in ("unet", "improved_unet"):
        raise ValueError("Checkpoint model must be unet or improved_unet")
    if not isinstance(config["label_values"], (list, tuple)):
        raise ValueError("Checkpoint label_values must be a sequence of raw integer labels")
    labels = validate_label_values(config["label_values"])
    if (isinstance(config["num_classes"], bool)
            or not isinstance(config["num_classes"], int)
            or len(labels) != config["num_classes"]):
        raise ValueError("Checkpoint num_classes does not match label_values")
    for key, minimum in (("base_channels", 1), ("image_size", 32), ("samples", 1), ("seed", 0)):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Checkpoint {key} must be an integer >= {minimum}")
    if synthetic:
        if labels != (0, 1):
            raise ValueError("Synthetic checkpoints must use label_values [0, 1]")
        foreground_label = 1
    else:
        foreground_label = config["prostate_label"]
        if foreground_label is None:
            raise ValueError("Real-data checkpoint must specify a documented prostate_label")
        if (isinstance(foreground_label, bool) or not isinstance(foreground_label, int)
                or foreground_label == 0 or foreground_label not in labels):
            raise ValueError("prostate_label must be a non-background raw value in label_values")
    return config, labels.index(foreground_label)


def run_prediction(checkpoint: str | Path, output: str | Path = "runs/predictions",
                   device: str = "auto", manifest: str | Path | None = None,
                   split: str = "val", limit: int = 4, synthetic: bool = False) -> dict:
    """Save a PNG and JSON; never change the checkpoint or dataset files."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if split not in ("val", "test"):
        raise ValueError("Prediction split must be val or test")
    selected_device = _device(device)
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    config, foreground_channel = _validate_checkpoint(payload, synthetic)
    if synthetic:
        if manifest is not None:
            raise ValueError("Do not combine --synthetic with a real-data --manifest")
        prediction_seed = config["seed"] + 2000
        data = SyntheticSegmentationDataset(samples=config["samples"],
                                            image_size=config["image_size"], seed=prediction_seed)
    else:
        if manifest is None:
            raise ValueError("Real-data prediction requires --manifest")
        manifest_sha256 = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        if manifest_sha256 != config.get("manifest_sha256"):
            raise ValueError("Prediction manifest does not match the checkpoint manifest_sha256; "
                             "use the unchanged training CSV to preserve patient split assignments")
        prediction_seed = None
        data = HipMRIDataset(manifest, split, image_size=config["image_size"],
                            label_values=config["label_values"], augment=False,
                            resize_mode=config["resize_mode"])
    model = build_model(config["model"], in_channels=1, num_classes=config["num_classes"],
                        base_channels=config["base_channels"]).to(selected_device)
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("Checkpoint model_state is incompatible with its model configuration") from exc
    del payload["model_state"]
    model.eval()

    count = min(limit, len(data))
    figure, axes = plt.subplots(count, 4, figsize=(12, 3 * count), squeeze=False)
    examples = []
    kind = "SYNTHETIC SOFTWARE CHECK" if synthetic else "HipMRI manifest examples"
    subject = "synthetic foreground" if synthetic else "prostate"
    figure.suptitle(f"{kind} | {config['model']} | {subject} error overlay", fontsize=13)
    try:
        for index in range(count):
            sample = data[index]
            with torch.inference_mode():
                logits = model(sample["image"].unsqueeze(0).to(selected_device))
                if not torch.isfinite(logits).all():
                    raise ValueError("Checkpoint produced non-finite logits; prediction is invalid")
                prediction = logits.argmax(dim=1)[0].cpu()
            image = sample["image"][0].cpu().numpy()
            target = sample["mask"].cpu()
            predicted_foreground = prediction == foreground_channel
            target_foreground = target == foreground_channel
            true_positive = predicted_foreground & target_foreground
            false_positive = predicted_foreground & ~target_foreground
            false_negative = ~predicted_foreground & target_foreground
            denominator = int(predicted_foreground.sum() + target_foreground.sum())
            dice = 2.0 * int(true_positive.sum()) / denominator if denominator else None
            overlay = np.zeros((*target.shape, 4), dtype=np.float32)
            overlay[false_positive.numpy()] = (1.0, 0.1, 0.1, 0.7)
            overlay[false_negative.numpy()] = (0.1, 0.35, 1.0, 0.7)
            axes[index, 0].imshow(image, cmap="gray")
            axes[index, 1].imshow(target.numpy(), cmap="viridis", vmin=0,
                                  vmax=config["num_classes"] - 1, interpolation="nearest")
            axes[index, 2].imshow(prediction.numpy(), cmap="viridis", vmin=0,
                                  vmax=config["num_classes"] - 1, interpolation="nearest")
            axes[index, 3].imshow(image, cmap="gray")
            axes[index, 3].imshow(overlay, interpolation="nearest")
            titles = (f"Input | example {index + 1}", "Ground truth", "Prediction",
                      f"FP red / FN blue | Dice {dice:.3f}" if dice is not None
                      else "FP red / FN blue | Dice undefined")
            for axis, title in zip(axes[index], titles):
                axis.set_title(title, fontsize=10)
                axis.axis("off")
            examples.append({"sample_index": index, "patient_id": sample["patient_id"],
                             "foreground_dice": dice,
                             "false_positive_pixels": int(false_positive.sum()),
                             "false_negative_pixels": int(false_negative.sum())})
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination / "prediction_grid.png", dpi=140)
    finally:
        plt.close(figure)
    summary = {
        "data_kind": "synthetic" if synthetic else "real_manifest",
        "interpretation": "Software check only; not HipMRI performance evidence." if synthetic
                          else "Example slices only; not a complete patient-level evaluation.",
        "checkpoint": str(Path(checkpoint).resolve()),
        "model": config["model"], "epoch": payload["epoch"], "device": str(selected_device),
        "split": split if not synthetic else "synthetic_holdout",
        "manifest_sha256": None if synthetic else manifest_sha256,
        "prediction_seed": prediction_seed,
        "foreground_raw_label": config["label_values"][foreground_channel],
        "foreground_channel": foreground_channel,
        "resize_mode": config["resize_mode"],
        "label_values": config["label_values"], "examples_shown": count,
        "dice_empty_case": "Undefined (null) when both target and prediction are empty.",
        "examples": examples,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", default=Path("runs/predictions"), type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    try:
        summary = run_prediction(**vars(args))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output.resolve()), "examples_shown": summary["examples_shown"],
                      "data_kind": summary["data_kind"]}, indent=2))


if __name__ == "__main__":
    main()
