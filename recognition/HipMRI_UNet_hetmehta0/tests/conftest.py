"""Shared test setup.

Markers
    requires_manifest  needs HIPMRI_METADATA=/path/to/metadata.csv (CSV only)
    requires_data      needs HIPMRI_METADATA and readable NIfTI files (Rangpur)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def pytest_collection_modifyitems(config, items):
    metadata = os.environ.get("HIPMRI_METADATA")
    have_manifest = bool(metadata) and Path(metadata).is_file()
    have_data = False
    if have_manifest:
        first = pd.read_csv(metadata, nrows=1)
        have_data = Path(first.loc[0, "image_path"]).is_file()
    for item in items:
        if "requires_manifest" in item.keywords and not have_manifest:
            item.add_marker(pytest.mark.skip(reason="HIPMRI_METADATA not set"))
        if "requires_data" in item.keywords and not have_data:
            item.add_marker(
                pytest.mark.skip(reason="HipMRI NIfTI files not accessible (run on Rangpur)")
            )


@pytest.fixture
def real_metadata() -> Path:
    return Path(os.environ["HIPMRI_METADATA"])


# -----------------------------------------------------------------------------
# Synthetic NIfTI data
# -----------------------------------------------------------------------------
def make_synthetic_volume(
    rng: np.random.Generator, shape=(32, 32, 6)
) -> tuple[np.ndarray, np.ndarray]:
    """Mask with all six labels in blocks; image encodes label and slice index.

    image[x, y, z] = 1 + 10 * mask[x, y, z] + 100 * z, so a correctly aligned
    (image, mask, z) triple satisfies that identity exactly.
    """
    mask = np.zeros(shape, dtype=np.int16)
    h, w, _ = shape
    mask[: h // 2, : w // 2] = 1
    mask[: h // 2, w // 2 :] = 2
    mask[h // 2 :, : w // 2] = 3
    mask[h // 2 :, w // 2 : 3 * w // 4] = 4
    mask[h // 2 :, 3 * w // 4 :] = 5
    mask[:2, :2] = 0
    # Random per-slice shuffle of foreground blocks so slices differ.
    for z in range(shape[2]):
        if rng.random() < 0.5:
            mask[:, :, z] = np.flip(mask[:, :, z], axis=0)
    mask[:, :, 0] = 0  # first slice background-only
    z_index = np.arange(shape[2])[None, None, :]
    image = (1 + 10 * mask + 100 * z_index).astype(np.float32)
    return image, mask


def write_nifti(path: Path, array: np.ndarray, affine: np.ndarray | None = None) -> None:
    nib.save(nib.Nifti1Image(array, np.eye(4) if affine is None else affine), str(path))


@pytest.fixture
def synthetic_manifest(tmp_path: Path) -> Path:
    """Three patients per split, two scans each, 32x32x6 volumes."""
    rng = np.random.default_rng(0)
    rows = []
    for split in ("train", "val", "test"):
        for p in range(3):
            patient = f"{split[:2].upper()}{p:02d}"
            for week in range(2):
                image, mask = make_synthetic_volume(rng)
                image_path = tmp_path / f"{patient}_Week{week}_image.nii.gz"
                mask_path = tmp_path / f"{patient}_Week{week}_mask.nii.gz"
                write_nifti(image_path, image)
                write_nifti(mask_path, mask)
                rows.append(
                    {
                        "patient_id": patient,
                        "scan_id": f"{patient}_Week{week}",
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "split": split,
                    }
                )
    path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
