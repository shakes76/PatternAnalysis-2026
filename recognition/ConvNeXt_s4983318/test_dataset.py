"""COMP3710 ADNI stage-2 data loader smoke test (no model training).

Usage (from your project folder):
  python test_dataset.py --data-root <AD_NC> --manifest-dir <prepared_seed42>

This script tests real CSV manifests, verifies patient separation, and loads
one mini-batch per split. It will NOT re-run the expensive image audit or move
any source images. AI-assisted initial scaffold; verify on your own data.
"""

import argparse
from collections import Counter
from pathlib import Path

import torch

from dataset import make_dataloaders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    loaders = make_dataloaders(
        data_root=args.data_root,
        manifest_dir=args.manifest_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=0,  # Safe default for Windows PowerShell.
        seed=42,
        augment_train=False,  # Reproducible, unaugmented inspection.
    )

    print("MANIFESTS (label 0=NC, label 1=AD)")
    for split, loader in loaders.items():
        dataset = loader.dataset
        patients = {row["patient_id"] for row in dataset.records}
        counts = Counter(row["label"] for row in dataset.records)
        print(f"{split}: images={len(dataset)}, patients={len(patients)}, "
              f"NC={counts[0]}, AD={counts[1]}")

        # This is intentionally just ONE batch rather than scanning all images.
        batch = next(iter(loader))
        expected = (min(args.batch_size, len(dataset)), 3,
                    args.image_size, args.image_size)
        actual = tuple(batch["image"].shape)
        assert actual == expected, f"Incorrect image batch shape: {actual}"
        assert batch["image"].dtype == torch.float32, "Expected float32 images"
        assert batch["label"].dtype == torch.int64, "Expected int64 labels"
        assert torch.isfinite(batch["image"]).all(), "Invalid image values"
        assert set(batch["label"].tolist()).issubset({0, 1})
        print(f"  batch image={actual}, labels={tuple(batch['label'].shape)}, "
              f"first_patient={batch['patient_id'][0]}")

    print("\nPASS: CSVs are patient-disjoint and all three batches loaded.")
    print("No model trained; original MRI images and CSVs are unchanged.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, AssertionError) as error:
        raise SystemExit(f"ERROR: {error}") from error
