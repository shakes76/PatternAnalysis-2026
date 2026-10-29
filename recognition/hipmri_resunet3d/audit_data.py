"""Inspect HipMRI Nifti pairs and label IDs before choosing the prostate label."""

import argparse
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from dataset import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/data_audit.json"))
    args = parser.parse_args()
    cases = read_manifest(args.manifest)
    rows = []
    labels = Counter()
    for case in cases:
        image = nib.as_closest_canonical(nib.load(str(case.image)))
        mask = nib.as_closest_canonical(nib.load(str(case.mask)))
        image_shape = (
            image.shape[:3] if len(image.shape) == 4 and image.shape[3] == 1
            else image.shape
        )
        mask_shape = (
            mask.shape[:3] if len(mask.shape) == 4 and mask.shape[3] == 1
            else mask.shape
        )
        aligned = np.allclose(image.affine, mask.affine, atol=1e-3)
        if image_shape != mask_shape or len(image_shape) != 3 or not aligned:
            raise ValueError(f"Misaligned image and mask: {case.case_id}")
        values, counts = np.unique(np.asarray(mask.dataobj), return_counts=True)
        if not np.allclose(values, np.rint(values), atol=1e-3):
            raise ValueError(f"Non-integer mask labels: {case.case_id}")
        histogram = {str(int(v)): int(n) for v, n in zip(values, counts)}
        labels.update({int(v): int(n) for v, n in zip(values, counts)})
        rows.append({
            "case_id": case.case_id,
            "patient_id": case.patient_id,
            "shape": image_shape,
            "spacing_mm": nib.affines.voxel_sizes(image.affine).tolist(),
            "label_voxels": histogram,
        })
    report = {
        "cases": len(cases),
        "patients": len({case.patient_id for case in cases}),
        "total_label_voxels": dict(sorted(labels.items())),
        "case_audit": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("cases", "patients", "total_label_voxels")}, indent=2))


if __name__ == "__main__":
    main()
