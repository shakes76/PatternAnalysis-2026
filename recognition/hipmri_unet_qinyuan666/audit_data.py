"""Audit real HipMRI slices before training; writes only aggregate metadata."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from dataset import (SPLITS, load_nifti_pair, map_labels, normalize_slice,
                     read_manifest, validate_label_values)


def audit_manifest(manifest: str | Path, label_values: tuple[int, ...],
                   prostate_label: int) -> dict:
    values = validate_label_values(label_values)
    if prostate_label == 0 or prostate_label not in values:
        raise ValueError("prostate_label must be a documented nonzero raw label in label_values.")
    records = read_manifest(manifest)
    shapes: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    empty_prostate = 0
    constant_images = 0
    for record in records:
        image, raw_mask = load_nifti_pair(record)
        map_labels(raw_mask, values)
        shapes["x".join(map(str, image.shape))] += 1
        observed, observed_counts = np.unique(raw_mask, return_counts=True)
        for raw, count in zip(observed, observed_counts):
            counts[str(int(raw))] += int(count)
        empty_prostate += int(not np.any(raw_mask == prostate_label))
        constant_images += int(np.std(image, dtype=np.float64) < 1e-8)
    split_summary = {}
    for split in SPLITS:
        rows = [record for record in records if record["split"] == split]
        split_summary[split] = {"slices": len(rows),
                                "patients": len({record["patient_id"] for record in rows})}
    warnings = []
    if any(summary["slices"] == 0 for summary in split_summary.values()):
        warnings.append("At least one split is empty; complete the split manifest before final evaluation.")
    if constant_images:
        warnings.append("Constant-intensity slices found; inspect them before training.")
    if counts[str(prostate_label)] == 0:
        warnings.append("No voxels contain the declared prostate label; verify the label mapping.")
    return {"data_source": "user_supplied_nifti_manifest", "audit_passed": True,
            "checks": ["patient_split_isolation", "file_path_split_isolation", "2d_slice_shapes",
                       "image_mask_affine_alignment", "finite_values", "global_integer_label_mapping"],
            "limitation": "Identity checks use supplied metadata and resolved paths, not content hashes.",
            "slices": len(records), "patients": len({row["patient_id"] for row in records}),
            "splits": split_summary, "original_shapes": dict(shapes),
            "raw_label_voxels": {str(raw): counts[str(raw)] for raw in values},
            "raw_label_to_class": {str(raw): i for i, raw in enumerate(values)},
            "declared_prostate_raw_label": prostate_label,
            "prostate_class_index": values.index(prostate_label),
            "slices_without_prostate": empty_prostate, "constant_intensity_slices": constant_images,
            "warnings": warnings}


def save_sample(manifest: str | Path, label_values: tuple[int, ...], output: Path) -> None:
    """Write an optional de-identified slice/annotation visual for manual checking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first_record = read_manifest(manifest)[0]
    image, raw_mask = load_nifti_pair(first_record)
    mapped = map_labels(raw_mask, label_values)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(normalize_slice(image), cmap="gray")
    axes[0].set_title("MRI slice")
    axes[1].imshow(normalize_slice(image), cmap="gray")
    axes[1].imshow(np.ma.masked_where(mapped == 0, mapped), cmap="tab10", alpha=0.5,
                   vmin=0, vmax=max(1, len(label_values) - 1), interpolation="nearest")
    axes[1].set_title("Ground-truth overlay")
    for axis in axes:
        axis.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label-values", nargs="+", type=int, required=True,
                        help="Documented raw labels, background 0 first; order fixes class indices.")
    parser.add_argument("--prostate-label", type=int, required=True,
                        help="Documented raw label for the prostate; do not guess this value.")
    parser.add_argument("--output", type=Path, required=True, help="Aggregate JSON report path.")
    parser.add_argument("--sample-output", type=Path, help="Optional de-identified PNG overlay.")
    args = parser.parse_args()
    try:
        values = validate_label_values(args.label_values)
        report = audit_manifest(args.manifest, values, args.prostate_label)
        if args.sample_output:
            save_sample(args.manifest, values, args.sample_output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
