"""Dataset tests: manifest/split checks, NIfTI loading, slice/mask alignment.

Synthetic tests run anywhere. ``requires_manifest`` tests read the real CSV
only; ``requires_data`` tests read real NIfTI volumes and must run on Rangpur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import make_synthetic_volume, write_nifti

from dataset import (
    EXPECTED_PATIENTS_PER_SPLIT,
    SliceDataset,
    VolumeSet,
    all_slice_indices,
    load_manifest,
    load_volume_pair,
    normalise_mri,
    select_overfit_slices,
    split_sha256,
)

SYNTHETIC_COUNTS = {"train": 3, "val": 3, "test": 3}


# -----------------------------------------------------------------------------
# Manifest and split checks (synthetic)
# -----------------------------------------------------------------------------
def test_synthetic_manifest_passes(synthetic_manifest):
    frame = load_manifest(synthetic_manifest, SYNTHETIC_COUNTS, expected_sha256=None)
    assert len(frame) == 18


def test_patient_in_two_splits_is_rejected(synthetic_manifest):
    frame = pd.read_csv(synthetic_manifest)
    frame.loc[frame["scan_id"] == "TR00_Week1", "split"] = "val"
    frame.to_csv(synthetic_manifest, index=False)
    with pytest.raises(ValueError, match="more than one split"):
        load_manifest(synthetic_manifest, None, None)


def test_wrong_patient_counts_rejected(synthetic_manifest):
    with pytest.raises(ValueError, match="Patients per split"):
        load_manifest(synthetic_manifest, EXPECTED_PATIENTS_PER_SPLIT, None)


def test_wrong_split_hash_rejected(synthetic_manifest):
    with pytest.raises(ValueError, match="Split hash"):
        load_manifest(synthetic_manifest, None, "0" * 64)


def test_duplicate_scan_and_bad_split_rejected(synthetic_manifest):
    frame = pd.read_csv(synthetic_manifest)
    bad = pd.concat([frame, frame.iloc[[0]]])
    bad.to_csv(synthetic_manifest, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(synthetic_manifest, None, None)
    frame.loc[0, "split"] = "holdout"
    frame.to_csv(synthetic_manifest, index=False)
    with pytest.raises(ValueError, match="Invalid split"):
        load_manifest(synthetic_manifest, None, None)


# -----------------------------------------------------------------------------
# Volume loading and slice alignment (synthetic NIfTI)
# -----------------------------------------------------------------------------
def test_slice_and_mask_alignment(synthetic_manifest):
    """image == 1 + 10 * mask + 100 * z holds for every returned slice."""
    frame = load_manifest(synthetic_manifest, None, None)
    volumes = VolumeSet.from_manifest(frame, "train", normalise=False)
    dataset = SliceDataset(volumes)
    assert len(dataset) == volumes.num_slices == 6 * 6
    for index in range(len(dataset)):
        image, mask = dataset[index]
        _, z = dataset.indices[index]
        assert image.shape == (1, 32, 32) and image.dtype == torch.float32
        assert mask.shape == (32, 32) and mask.dtype == torch.int64
        assert torch.equal(image[0], (1 + 10 * mask + 100 * z).float())


def test_alignment_survives_non_ras_affine(tmp_path):
    """Both image and mask are reoriented identically when the affine flips axes."""
    image, mask = make_synthetic_volume(np.random.default_rng(1))
    affine = np.diag([-1.0, 1.0, -1.0, 1.0])
    write_nifti(tmp_path / "img.nii.gz", image, affine)
    write_nifti(tmp_path / "msk.nii.gz", mask, affine)
    loaded_image, loaded_mask = load_volume_pair(
        tmp_path / "img.nii.gz", tmp_path / "msk.nii.gz", normalise=False
    )
    assert loaded_mask.dtype == np.uint8
    # x and z were flipped by canonicalisation; the identity uses the original z.
    np.testing.assert_array_equal(loaded_mask, mask[::-1, :, ::-1])
    depth = image.shape[2]
    original_z = (depth - 1 - np.arange(depth))[None, None, :]
    np.testing.assert_array_equal(
        loaded_image, 1 + 10 * loaded_mask.astype(np.float32) + 100 * original_z
    )


def test_labels_preserved_and_validated(tmp_path):
    image, mask = make_synthetic_volume(np.random.default_rng(2))
    write_nifti(tmp_path / "img.nii.gz", image)
    write_nifti(tmp_path / "msk.nii.gz", mask)
    _, loaded = load_volume_pair(tmp_path / "img.nii.gz", tmp_path / "msk.nii.gz")
    assert set(np.unique(loaded)) == {0, 1, 2, 3, 4, 5}

    bad = mask.copy()
    bad[5, 5, 3] = 7
    write_nifti(tmp_path / "bad.nii.gz", bad)
    with pytest.raises(ValueError, match="outside"):
        load_volume_pair(tmp_path / "img.nii.gz", tmp_path / "bad.nii.gz")

    fractional = mask.astype(np.float32)
    fractional[5, 5, 3] = 1.5
    write_nifti(tmp_path / "frac.nii.gz", fractional)
    with pytest.raises(ValueError, match="non-integer"):
        load_volume_pair(tmp_path / "img.nii.gz", tmp_path / "frac.nii.gz")


def test_shape_and_affine_mismatch_rejected(tmp_path):
    image, mask = make_synthetic_volume(np.random.default_rng(3))
    write_nifti(tmp_path / "img.nii.gz", image)
    write_nifti(tmp_path / "short.nii.gz", mask[:, :, :4])
    with pytest.raises(ValueError, match="shape mismatch"):
        load_volume_pair(tmp_path / "img.nii.gz", tmp_path / "short.nii.gz")
    shifted = np.eye(4)
    shifted[0, 3] = 5.0
    write_nifti(tmp_path / "shifted.nii.gz", mask, shifted)
    with pytest.raises(ValueError, match="affine mismatch"):
        load_volume_pair(tmp_path / "img.nii.gz", tmp_path / "shifted.nii.gz")


def test_normalise_mri_keeps_background_zero():
    rng = np.random.default_rng(4)
    image = np.zeros((16, 16, 4), dtype=np.float32)
    image[4:12, 4:12] = rng.normal(500, 50, size=(8, 8, 4))
    result = normalise_mri(image)
    assert result.dtype == np.float32
    assert np.all(result[image == 0] == 0)
    foreground = result[image != 0]
    assert abs(foreground.mean()) < 0.05 and abs(foreground.std() - 1) < 0.05


def test_depth_taken_from_each_volume(tmp_path):
    """Volumes with different depths (like the single 144-slice scan) are fully indexed."""
    volumes = VolumeSet(
        scan_ids=["a", "b"],
        patient_ids=["p", "q"],
        images=[np.zeros((16, 16, 4), np.float32), np.zeros((16, 16, 7), np.float32)],
        masks=[np.zeros((16, 16, 4), np.uint8), np.zeros((16, 16, 7), np.uint8)],
    )
    indices = all_slice_indices(volumes)
    assert len(indices) == 11
    assert indices[-1] == (1, 6)


def test_overfit_selection_is_seeded_and_foreground_only(synthetic_manifest):
    frame = load_manifest(synthetic_manifest, None, None)
    volumes = VolumeSet.from_manifest(frame, "train")
    first = select_overfit_slices(volumes, 8, seed=3710)
    assert first == select_overfit_slices(volumes, 8, seed=3710)
    assert len(set(first)) == 8
    for v, z in first:
        assert volumes.masks[v][:, :, z].any()
        assert z != 0  # slice 0 is background-only in the synthetic data


def test_seeded_loader_order_is_reproducible(synthetic_manifest):
    frame = load_manifest(synthetic_manifest, None, None)
    dataset = SliceDataset(VolumeSet.from_manifest(frame, "train"))

    def order(seed):
        g = torch.Generator().manual_seed(seed)
        return list(torch.utils.data.RandomSampler(dataset, generator=g))

    assert order(3710) == order(3710)
    assert order(3710) != order(1)


# -----------------------------------------------------------------------------
# Real manifest (CSV only, no images needed)
# -----------------------------------------------------------------------------
@pytest.mark.requires_manifest
def test_real_manifest_frozen_split(real_metadata):
    frame = load_manifest(real_metadata)  # frozen 26/6/6 counts and SHA-256
    assert frame["patient_id"].nunique() == 38
    assert len(frame) == 211
    patients = {
        s: set(frame.loc[frame["split"] == s, "patient_id"]) for s in ("train", "val", "test")
    }
    assert not patients["train"] & patients["val"]
    assert not patients["train"] & patients["test"]
    assert not patients["val"] & patients["test"]
    depth = frame["shape"].str.split("x").str[-1].astype(int)
    slices = depth.groupby(frame["split"]).sum().to_dict()
    assert slices == {"train": 17424, "val": 4352, "test": 5248}
    assert split_sha256(frame) == "4da182221410f102ad9282c9b1167ccc357273fe5703517da9a361f792dd7886"


# -----------------------------------------------------------------------------
# Real NIfTI data (Rangpur only)
# -----------------------------------------------------------------------------
@pytest.mark.requires_data
def test_real_masks_labels_and_geometry(real_metadata):
    """Check one train and one val scan: labels within 0..5, image/mask aligned, depth matches manifest."""
    frame = load_manifest(real_metadata)
    for split in ("train", "val"):
        row = frame[frame["split"] == split].iloc[0]
        image, mask = load_volume_pair(row.image_path, row.mask_path)
        assert image.shape == mask.shape
        assert "x".join(map(str, image.shape)) == row["shape"]
        assert set(np.unique(mask).tolist()) <= {0, 1, 2, 3, 4, 5}
        assert np.isfinite(image).all()
