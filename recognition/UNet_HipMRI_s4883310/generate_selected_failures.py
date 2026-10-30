"""Generate comparison figures for manually selected MRI failure cases."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import test_dataset
from evaluate import (
    PROSTATE_CLASS,
    calculate_prostate_metrics,
    load_model,
)

from predict import (
    CHECKPOINTS,
    error_counts,
    metric_text,
    plot_error_overlay,
    plot_mask,
    predict_one,
)


# Manually selected cases for further investigation.
# These descriptions must be verified against the current predictions.
SELECTED_CASES = {
    "case_040_week_0_slice_36.nii.gz": "Possible boundary failure",
    "case_040_week_4_slice_17.nii.gz": "Possible disconnected prediction",
}

OUTPUT_DIR = Path("results/selected_failures")


def find_selected_indices():
    """Find the selected MRI files in the test dataset."""
    filename_to_index = {
        Path(path).name: index
        for index, path in enumerate(test_dataset.image_paths)
    }

    missing = [
        filename
        for filename in SELECTED_CASES
        if filename not in filename_to_index
    ]

    if missing:
        raise ValueError(
            f"Selected files not found in test dataset: {missing}"
        )

    return {
        filename: filename_to_index[filename]
        for filename in SELECTED_CASES
    }


def generate_selected_failures(models):
    """Generate six-panel comparisons for manually selected MRI slices."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case_indices = find_selected_indices()
    results = []

    for filename, index in case_indices.items():
        reason = SELECTED_CASES[filename]

        image_tensor, mask_tensor = test_dataset[index]

        image = image_tensor.squeeze().cpu().numpy()
        ground_truth = mask_tensor.cpu().numpy()

        # Generate predictions using both trained models.
        predictions = {
            name: predict_one(model, image_tensor)
            for name, model in models.items()
        }

        for name, prediction in predictions.items():
            if prediction.shape != ground_truth.shape:
                raise ValueError(
                    f"{name} prediction shape does not match "
                    f"ground truth for {filename}"
                )

        # Calculate segmentation metrics and error counts.
        metrics = {
            name: calculate_prostate_metrics(
                prediction,
                ground_truth
            )
            for name, prediction in predictions.items()
        }

        errors = {
            name: error_counts(prediction, ground_truth)
            for name, prediction in predictions.items()
        }

        # Create the six-panel comparison.
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        axes[0, 0].imshow(image, cmap="gray")
        axes[0, 0].set_title("Original MRI")
        axes[0, 0].axis("off")

        plot_mask(
            axes[0, 1],
            image,
            ground_truth,
            "Greens",
            "Ground Truth"
        )

        plot_mask(
            axes[0, 2],
            image,
            predictions["baseline"],
            "Reds",
            "Baseline Prediction"
        )

        plot_mask(
            axes[1, 0],
            image,
            predictions["improved"],
            "Blues",
            "Improved Prediction"
        )

        # Highlight false positives and false negatives.
        plot_error_overlay(
            axes[1, 1],
            image,
            ground_truth,
            predictions["baseline"],
            "Baseline Errors (FP red / FN blue)"
        )

        plot_error_overlay(
            axes[1, 2],
            image,
            ground_truth,
            predictions["improved"],
            "Improved Errors (FP red / FN blue)"
        )

        # Add evaluation metrics to the figure title.
        descriptions = []

        for name in CHECKPOINTS:
            fp, fn = errors[name]

            descriptions.append(
                f"{name.title()}: {metric_text(metrics[name])}, "
                f"FP={fp}, FN={fn}"
            )

        fig.suptitle(
            f"{filename}\n"
            f"Selection: {reason}\n"
            + "\n".join(descriptions),
            fontsize=11
        )

        fig.tight_layout(rect=[0, 0, 1, 0.87])

        # Save the figure using the MRI filename.
        case_name = filename.replace(".nii.gz", "")

        output_path = OUTPUT_DIR / f"{case_name}.png"

        fig.savefig(output_path, dpi=200)
        plt.close(fig)

        print(f"Saved: {output_path}")

        # Record the selected case and its evaluation results.
        row = {
            "filename": filename,
            "index": index,
            "selection_reason": reason,
            "figure": str(output_path),
        }

        for name in CHECKPOINTS:
            model_metrics = metrics[name]
            fp, fn = errors[name]

            row[f"{name}_dice"] = (
                model_metrics["dice"]
                if model_metrics is not None else ""
            )

            row[f"{name}_iou"] = (
                model_metrics["iou"]
                if model_metrics is not None else ""
            )

            row[f"{name}_hd95_pixels"] = (
                model_metrics["hd95_pixels"]
                if model_metrics is not None else ""
            )

            row[f"{name}_fp"] = fp
            row[f"{name}_fn"] = fn

        results.append(row)

    # Save a summary of the manually selected cases.
    if results:
        with (OUTPUT_DIR / "selected_cases.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(results[0].keys())
            )

            writer.writeheader()
            writer.writerows(results)

    print(f"Generated {len(results)} selected failure comparisons.")


def main():
    """Load both trained models and generate selected case figures."""
    if not SELECTED_CASES:
        raise ValueError("No MRI cases have been selected.")

    models = {}

    # Load the best checkpoint for each model.
    for name, checkpoint in CHECKPOINTS.items():
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint}"
            )

        models[name] = load_model(name, checkpoint)
        models[name].eval()

    generate_selected_failures(models)


if __name__ == "__main__":
    main()