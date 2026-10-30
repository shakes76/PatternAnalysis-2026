"""Inspect a frozen U-Net checkpoint on an explicitly selected data split."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

from dataset import HipMRIDataset
from metrics import positive_slice_stats, summarize_stats
from modules import UNet2D, ImprovedUNet2D


def accumulate(total, current):
    for key, value in current.items():
        total[key] = total.get(key, 0) + value


def plot_failure(sample, prediction, dice, output, split="validate"):
    image = sample["image"][0].numpy()
    target = sample["mask"].numpy() == 5
    predicted = prediction.numpy() == 5
    fp = predicted & ~target
    fn = target & ~predicted

    overlay = np.zeros((*target.shape, 4), dtype=np.float32)
    overlay[fp] = [1.0, 0.0, 0.0, 0.8]
    overlay[fn] = [0.0, 1.0, 1.0, 0.8]

    low, high = np.percentile(image, [1, 99])
    if high <= low:
        high = low + 1

    fig, axes = plt.subplots(
        1, 4, figsize=(10, 4), constrained_layout=True
    )
    axes[0].imshow(image, cmap="gray", vmin=low, vmax=high)
    axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(predicted, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(image, cmap="gray", vmin=low, vmax=high)
    axes[3].imshow(overlay, interpolation="nearest")

    for axis, title in zip(
        axes, ["MRI", "True prostate", "Predicted prostate", "Prostate errors"]
    ):
        axis.set_title(title)
        axis.set_axis_off()

    axes[3].legend(
        handles=[
            Patch(color="red", label="False positive"),
            Patch(color="cyan", label="False negative"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8,
    )
    fig.suptitle(
        f"{split.capitalize()} case {sample['case_id']:03d}, "
        f"week {sample['week']}, slice {sample['slice_id']}\n"
        f"Prostate Dice: {dice:.4f}; stored array orientation"
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validate", "test"), default="validate")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Run this analysis inside a GPU job.")
    torch.set_num_threads(2)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    model_name = checkpoint.get("model_name")
    if checkpoint.get("num_classes") != 6:
        raise ValueError("Expected six output classes.")
    if model_name == "UNet2D":
        model = UNet2D(base_channels=checkpoint["base_channels"])
    elif model_name == "ImprovedUNet2D":
        model = ImprovedUNet2D(
            base_channels=checkpoint["base_channels"],
            dropout=checkpoint["dropout"],
        )
    else:
        raise ValueError(f"Unsupported checkpoint model: {model_name!r}")
    model = model.cuda()
    print("Checkpoint model:", model_name, flush=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    saved_dice = checkpoint["validation_prostate_dice"]
    epoch = checkpoint["epoch"]
    print("Checkpoint epoch:", epoch, flush=True)
    print("Saved validation prostate Dice:", saved_dice, flush=True)

    data_root = checkpoint["config"]["data_root"]
    dataset = HipMRIDataset(data_root, split=args.split)
    args.output.mkdir(parents=True, exist_ok=False)

    total = {}
    per_case = defaultdict(dict)
    worst = []
    fields = [
        "case_id", "week", "slice_id", "height", "width",
        "true_prostate_pixels", "predicted_prostate_pixels",
        "false_positive_pixels", "false_negative_pixels",
        "eligible", "prostate_dice", "prostate_iou",
    ]

    with (args.output / "slice_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()

        for index in range(len(dataset)):
            sample = dataset[index]
            image = sample["image"].unsqueeze(0).cuda()
            prediction = model(image).argmax(dim=1).cpu()
            target = sample["mask"].unsqueeze(0)

            stats = positive_slice_stats(prediction, target, class_id=5)
            score = summarize_stats(stats)
            accumulate(total, stats)
            accumulate(per_case[sample["case_id"]], stats)

            actual = target == 5
            predicted = prediction == 5
            writer.writerow({
                "case_id": sample["case_id"],
                "week": sample["week"],
                "slice_id": sample["slice_id"],
                "height": target.shape[1],
                "width": target.shape[2],
                "true_prostate_pixels": int(actual.sum()),
                "predicted_prostate_pixels": int(predicted.sum()),
                "false_positive_pixels": int((predicted & ~actual).sum()),
                "false_negative_pixels": int((actual & ~predicted).sum()),
                "eligible": bool(stats["eligible_slices"]),
                "prostate_dice": score["mean_dice"],
                "prostate_iou": score["mean_iou"],
            })

            if stats["eligible_slices"]:
                worst.append((
                    score["mean_dice"], index, prediction[0].clone()
                ))
                worst.sort(key=lambda item: (item[0], item[1]))
                worst = worst[:3]

            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1}/{len(dataset)}", flush=True)

    overall = summarize_stats(total)
    summary = {
        "split": args.split,
        "model_name": model_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": epoch,
        "saved_validation_prostate_dice": saved_dice,
        "overall": overall,
        "per_case": {
            str(case): summarize_stats(stats)
            for case, stats in sorted(per_case.items())
        },
        "figure_selection": "Three lowest-Dice ground-truth-positive slices",
    }
    if args.split == "validate" and overall["mean_dice"] is not None:
        summary["dice_difference_from_training_log"] = (
            overall["mean_dice"] - saved_dice
        )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    for rank, (dice, index, prediction) in enumerate(worst, start=1):
        sample = dataset[index]
        name = (
            f"worst_{rank}_case_{sample['case_id']:03d}"
            f"_week_{sample['week']}_slice_{sample['slice_id']}.png"
        )
        plot_failure(sample, prediction, dice, args.output / name, split=args.split)

    print(json.dumps(summary, indent=2), flush=True)
    if args.split == "validate":
        print("Validation analysis complete. No test data were read.", flush=True)
    else:
        print("Frozen-checkpoint test inspection complete. Do not tune on these results.", flush=True)


if __name__ == "__main__":
    main()
