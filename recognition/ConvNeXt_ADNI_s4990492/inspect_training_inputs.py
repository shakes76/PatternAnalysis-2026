"""Inspect fixed training samples and actual affine augmentation; CPU only."""
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import ADNISliceDataset
from dataset_aug import AugmentedADNISliceDataset


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    rng = random.Random(42)
    root = Path(__file__).resolve().parents[2] / "local_data"
    manifest = root / "splits_local" / "train.csv"
    original = ADNISliceDataset(manifest)
    augmented = AugmentedADNISliceDataset(manifest)
    patients = {0: defaultdict(list), 1: defaultdict(list)}
    for index, row in enumerate(original.rows):
        patients[int(row["target"])][row["patient_id"]].append(index)
    selected = []
    for label in (0, 1):
        for patient in rng.sample(sorted(patients[label]), 4):
            selected.append(rng.choice(patients[label][patient]))
    out = root / "diagnostics" / datetime.now().strftime("training_inputs_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(8, 3, figsize=(9, 22))
    records = []
    for row_index, index in enumerate(selected):
        image, label = original[index]
        variants = [image, augmented[index][0], augmented[index][0]]
        for column, tensor in enumerate(variants):
            if not torch.isfinite(tensor).all():
                raise ValueError("Non-finite image")
            array = tensor.squeeze(0).numpy()
            axes[row_index, column].imshow(array, cmap="gray", vmin=0, vmax=1)
            axes[row_index, column].set_axis_off()
            name = "NC" if label.item() == 0 else "AD"
            variant = ["Original", "Affine A", "Affine B"][column]
            axes[row_index, column].set_title(f"{name} sample {row_index % 4 + 1}: {variant}")
            records.append({
                "sample": row_index + 1, "class": name, "variant": variant,
                "manifest_row": index, "height": tensor.shape[-2],
                "width": tensor.shape[-1], "minimum": tensor.min().item(),
                "maximum": tensor.max().item(), "mean": tensor.mean().item(),
                "std_population": tensor.std(unbiased=False).item(),
                "zero_fraction": (tensor == 0).float().mean().item(),
            })
    fig.suptitle("Training images only: fixed patients, original vs affine augmentation", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / "training_inputs.png", dpi=140)
    plt.close(fig)
    with (out / "image_statistics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "split": "train", "seed": 42, "patients_sampled": 8,
        "images_sampled": 8, "augmented_versions_per_image": 2,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "sampling": "four randomly selected distinct patients per class, one random slice per patient",
        "limits": "Small visual sample; not an exhaustive data or orientation audit.",
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    main()
