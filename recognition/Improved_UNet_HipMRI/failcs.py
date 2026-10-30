"""Identify difficult HipMRI test slices using foreground Dice scores."""

import csv
from pathlib import Path

import torch

from dataset import HipMRIDataset
from modules import ImprovedUNet


NUM_CLASSES = 6


def class_dice(prediction, target, label):
    """
    Calculate Dice for one segmentation class in a single MRI slice.

    Return None if the class is absent from both the prediction
    and the ground truth.
    """
    pred = prediction == label
    true = target == label

    denominator = pred.sum().item() + true.sum().item()

    if denominator == 0:
        return None

    intersection = (pred & true).sum().item()

    return 2 * intersection / denominator


def main():
    """Rank test slices by the Improved U-Net's mean foreground Dice."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Use the held-out test dataset for failure-case analysis.
    dataset = HipMRIDataset(split="test")

    # Load the best checkpoint from the final Improved U-Net run.
    checkpoint = torch.load(
        "runs/improved_v1/best_model.pth",
        map_location="cpu",
        weights_only=True,
    )

    model = ImprovedUNet(
        in_channels=1,
        num_classes=NUM_CLASSES,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    results = []

    # Evaluate each test slice separately.
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, mask = dataset[index]

            output = model(image.unsqueeze(0).to(device))
            prediction = output.argmax(dim=1).squeeze(0).cpu()

            row = {
                "index": index,
                "filename": dataset.pairs[index][0].name,
            }

            scores = []

            # Calculate Dice for foreground classes 1 to 5.
            # Background is excluded from this analysis.
            for label in range(1, NUM_CLASSES):
                dice = class_dice(prediction, mask, label)

                row[f"dice_class_{label}"] = dice

                if dice is not None:
                    scores.append(dice)

            # Average only the foreground classes present in
            # either the prediction or ground truth.
            row["mean_foreground_dice"] = (
                sum(scores) / len(scores) if scores else None
            )

            results.append(row)

    # Save the per-slice results for later analysis.
    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "slice_dice_results.csv"

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    # Rank slices from lowest to highest mean foreground Dice.
    ranked = [
        row for row in results
        if row["mean_foreground_dice"] is not None
    ]

    ranked.sort(
        key=lambda row: row["mean_foreground_dice"]
    )

    print("\nWorst 10 test slices by foreground Dice:")

    for row in ranked[:10]:
        print(
            f"Index {row['index']:3d} | "
            f"Dice {row['mean_foreground_dice']:.4f} | "
            f"{row['filename']}"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

