
import csv
import os

import numpy as np
import torch
from scipy.ndimage import label

from dataset import test_loader
from evaluate import PROSTATE_CLASS, device, load_model


def count_components(prediction):
    """Count the number of disconnected prostate regions in a 2D prediction."""
    
    target = prediction == PROSTATE_CLASS

    # Include diagonal neighbours as connected.
    structure = np.ones((3, 3), dtype=bool)

    # Label each connected prostate region and count the components
    _, count = label(target, structure=structure)
    
    return int(count)


def analyse_model(model, model_name):
    """Measure disconnected predictions across the test set."""
    
    rows = []
    index = 0

    with torch.no_grad():
        # Process each batch of MRI images and ground-truth masks
        for images, masks in test_loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu().numpy()
            ground_truths = masks.numpy()

            # Compare predicted and actual components for each MRI slice
            for prediction, ground_truth in zip(
                predictions, ground_truths
            ):
                # Count connected prostate regions in both masks
                predicted_count = count_components(prediction)
                actual_count = count_components(ground_truth)

                # Record component counts and any excess predicted components
                rows.append({
                    "index": index,
                    "gt_components": actual_count,
                    "predicted_components": predicted_count,
                    "extra_components": max(
                        0, predicted_count - actual_count
                    ),
                })
                index += 1

    return rows


def save_results(rows, model_name):
    """
    Save component analysis results and print summary statistics.
    """
    
    os.makedirs("results", exist_ok=True)

    path = f"results/{model_name}_artifacts.csv"

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "gt_components",
                "predicted_components",
                "extra_components",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Only include slices where the ground truth contains prostate tissue
    relevant = [
        row for row in rows
        if row["gt_components"] > 0
    ]

    # Count slices where the prediction has more components than ground truth
    failures = sum(
        row["extra_components"] > 0
        for row in relevant
    )
    
    # Calculate the average number of excess components per relevant slice
    mean_extra_components = (
        np.mean([row["extra_components"] for row in relevant])
        if relevant else 0.0
        )

    print(f"\n{model_name}")
    print(f"Relevant slices: {len(relevant)}")
    print(f"Slices with extra components: {failures}")
    print(
        f"Mean extra components: {mean_extra_components:.4f}"
    )

    # Percentage of relevant slices with excess components
    if relevant:
        print(
            f"Extra-component rate: "
            f"{100 * failures / len(relevant):.2f}%"
        )


if __name__ == "__main__":
    models = {
        "baseline": "checkpoints/unet_baseline_best.pth",
        "improved": "checkpoints/unet_improved_best.pth",
    }

    # Analyse each model and save its component results
    for name, checkpoint in models.items():
        model = load_model(name, checkpoint)
        rows = analyse_model(model, name)
        save_results(rows, name)
