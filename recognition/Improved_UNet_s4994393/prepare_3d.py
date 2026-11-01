"""Optional paired downsampling with a shared grid and prostate-preservation audit."""

import argparse
import csv
import json
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

from dataset import discover_pairs, load_volume


def sha256(path):
    """Hash file contents in bounded chunks for preprocessing provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison_panel(raw, labels, image, mask, affine, output):
    """Show corresponding physical slices; terminal slices are only anatomical proxies."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis = next(i for i, code in enumerate(nib.aff2axcodes(affine)) if code in ("S", "I"))
    occupied = np.flatnonzero(np.any(labels == 5, axis=tuple(i for i in range(3) if i != axis)))
    if not len(occupied):
        raise ValueError("Cannot create prostate panel for a case without prostate")
    slices = [occupied[0], occupied[len(occupied) // 2], occupied[-1]]
    if nib.aff2axcodes(affine)[axis] == "I":
        slices.reverse()
    ratio = raw.shape[axis] / image.shape[axis]
    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    for row, (name, original) in enumerate(zip(("apex proxy", "middle", "base proxy"), slices)):
        reduced = int(np.clip(round((original + 0.5) / ratio - 0.5), 0, image.shape[axis] - 1))
        for col, (volume, target, index) in enumerate(
            ((raw, labels, original), (image, mask, reduced))
        ):
            ax = axes[row, col]
            ax.imshow(np.take(volume, index, axis=axis).T, cmap="gray", origin="lower")
            region = np.take(target, index, axis=axis).T == 5
            ax.imshow(
                np.ma.masked_where(~region, region),
                cmap="autumn",
                alpha=0.5,
                origin="lower",
                vmin=0,
                vmax=1,
            )
            ax.set_title(f'{name}: {"source" if col == 0 else "downsampled"} slice {index}')
            ax.axis("off")
    fig.suptitle(output.stem + "\nNearest corresponding slices; label overlay, not a prediction")
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    """Resample paired scans onto one grid, audit every pair and write a manifest.

    A missing completion manifest means the output is incomplete. Any invalid
    pair or disappearing label aborts preparation instead of excluding a scan.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument(
        "--reference-split",
        type=Path,
        help="Recorded split; verify names and use training cases for panels",
    )
    args = parser.parse_args()
    if not 0 < args.scale < 1:
        parser.error("scale must be between zero and one")
    if args.output_root.exists():
        parser.error("Use a new output directory to preserve original scans")
    pairs = discover_pairs(args.data_root)
    panel_cases = set()
    if args.reference_split:
        split = json.loads(args.reference_split.read_text())
        names = [Path(pair[0]).name for group in split.values() for pair in group]
        if len(names) != len(set(names)) or set(names) != {Path(p[0]).name for p in pairs}:
            raise ValueError("Reference split must contain exactly the source scans, once each")
        # Choose one deterministic case from each of three training patients only.
        selected_patients = set()
        for pair in sorted(split["train"]):
            name = Path(pair[0]).name
            patient = name.split("_")[0]
            if patient not in selected_patients:
                selected_patients.add(patient)
                panel_cases.add(name)
            if len(panel_cases) == 3:
                break
    (args.output_root / "semantic_MRs").mkdir(parents=True)
    (args.output_root / "semantic_labels_only").mkdir()
    rows = []
    records = []
    for pair in pairs:
        _, labels = load_volume(pair, 6)  # Validate the complete original pair first.
        source = nib.load(pair[0])
        raw = source.get_fdata(dtype=np.float32)
        shape = tuple(max(1, round(s * args.scale)) for s in raw.shape)
        # align_corners=False maps new voxel centers to old voxel centers.
        ratio = np.array(raw.shape) / np.array(shape)
        transform = np.eye(4)
        transform[:3, :3] = np.diag(ratio)
        # Include the half-voxel offset: scaling spacing alone shifts the physical grid.
        transform[:3, 3] = (ratio - 1) / 2
        affine = source.affine @ transform
        image = F.interpolate(
            torch.from_numpy(raw)[None, None], size=shape, mode="trilinear", align_corners=False
        )[0, 0].numpy()
        # nearest-exact matches the same voxel-center mapping as trilinear.
        mask = (
            F.interpolate(
                torch.from_numpy(labels.astype(np.float32))[None, None],
                size=shape,
                mode="nearest-exact",
            )[0, 0]
            .numpy()
            .astype(np.int16)
        )
        # Retaining labels is necessary, but does not prove boundaries were preserved.
        lost = set(np.unique(labels)) - set(np.unique(mask))
        if lost:
            raise ValueError(
                f"{pair[0]} loses labels {lost}; use a larger scale and a fresh output folder"
            )
        before, after = int(np.count_nonzero(labels == 5)), int(np.count_nonzero(mask == 5))
        # Voxel counts alone shrink with downsampling; account for larger voxel volume.
        volume_ratio = after * float(np.prod(ratio)) / before if before else None
        record = {
            "case": Path(pair[0]).name,
            "prostate_voxels_before": before,
            "prostate_voxels_after": after,
            "prostate_physical_volume_ratio": volume_ratio,
            "original_shape": list(raw.shape),
            "output_shape": list(shape),
            "original_spacing": list(map(float, nib.affines.voxel_sizes(source.affine))),
            "output_spacing": list(map(float, nib.affines.voxel_sizes(affine))),
            "original_affine": source.affine.tolist(),
            "output_affine": affine.tolist(),
            "labels_before": np.unique(labels).tolist(),
            "labels_after": np.unique(mask).tolist(),
            "source_image_sha256": sha256(pair[0]),
            "source_mask_sha256": sha256(pair[1]),
        }
        for array, original, folder in (
            (image, pair[0], "semantic_MRs"),
            (mask, pair[1], "semantic_labels_only"),
        ):
            header = nib.load(original).header.copy()
            header.set_data_dtype(array.dtype)
            nib.save(
                nib.Nifti1Image(array, affine, header),
                args.output_root / folder / Path(original).name,
            )
        prepared = tuple(
            args.output_root / folder / Path(original).name
            for original, folder in zip(pair, ("semantic_MRs", "semantic_labels_only"))
        )
        load_volume(prepared, 6)  # Verify serialized output, not just in-memory arrays.
        actual_image, actual_mask = (nib.load(p) for p in prepared)
        if not np.allclose(actual_image.affine, affine) or not np.allclose(
            actual_mask.affine, affine
        ):
            raise ValueError("Saved affine differs from intended shared grid")
        record.update(
            alignment_verified=True,
            output_image_sha256=sha256(prepared[0]),
            output_mask_sha256=sha256(prepared[1]),
        )
        records.append(record)
        rows.append({k: json.dumps(v) if isinstance(v, list) else v for k, v in record.items()})
        if Path(pair[0]).name in panel_cases:
            comparison_panel(
                raw,
                labels,
                image,
                mask,
                source.affine,
                args.output_root / (Path(pair[0]).name.replace(".nii.gz", "_preprocessing.png")),
            )
        print(
            f"{Path(pair[0]).name}: {raw.shape} -> {shape}; prostate volume ratio={volume_ratio}",
            flush=True,
        )
    with (args.output_root / "downsampling_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Written last: absence of this marker means preprocessing did not complete.
    manifest = {
        "complete": True,
        "scale": args.scale,
        "cases": len(records),
        "source_root": str(args.data_root.resolve()),
        "image_interpolation": "trilinear",
        "mask_interpolation": "nearest-exact",
        "align_corners": False,
        "torch_version": str(torch.__version__),
        "panel_training_cases": sorted(panel_cases),
        "reference_split_sha256": sha256(args.reference_split) if args.reference_split else None,
        "records": records,
    }
    (args.output_root / "preprocessing_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Inspect prostate volume ratios and boundaries before choosing the training grid.")


if __name__ == "__main__":
    main()
