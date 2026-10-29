"""Check fixed central cropping and native-grid reconstruction."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import Case, VolumeDataset, central_xy_slices, mirror_left_right  # noqa: E402
from train import native_case_metrics  # noqa: E402


def test_crop_slices_validate_size() -> None:
    assert central_xy_slices((20, 16), 8) == (slice(6, 14), slice(4, 12))
    assert central_xy_slices((20, 16), None) == (slice(0, 20), slice(0, 16))
    with pytest.raises(ValueError, match="positive even"):
        central_xy_slices((20, 16), 9)
    with pytest.raises(ValueError, match="no larger"):
        central_xy_slices((20, 16), 18)


def test_cropped_labels_align_and_restore_to_native(tmp_path: Path) -> None:
    image = np.zeros((16, 16, 16), dtype=np.float32)
    image[3:13, 3:13, :] = 1.0
    image[6:10, 6:10, 5:11] = 2.0
    labels = np.zeros_like(image)
    labels[6:10, 6:10, 5:11] = 5
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(labels, np.eye(4)), mask_path)
    case = Case("example", "patient_1", image_path, mask_path)

    sample = VolumeDataset([case], (16, 8, 8), 5, crop_xy=8)[0]
    assert sample["image"].shape == (1, 16, 8, 8)
    assert sample["mask"].shape == (16, 8, 8)
    assert sample["mask"].sum().item() == 4 * 4 * 6
    assert torch.equal(sample["mask"], (sample["anatomy"] == 5).long())

    score = native_case_metrics(
        sample["mask"].numpy(), case, 5, None, None, crop_xy=8
    )
    assert score["native_prostate_dice"] == 1.0
    assert score["native_hd95_mm"] == 0.0

    filled_crop = np.ones((16, 8, 8), dtype=bool)
    score = native_case_metrics(filled_crop, case, 5, None, None, crop_xy=8)
    assert score["native_false_positive_voxels"] == 8 * 8 * 16 - 4 * 4 * 6

    mirrored = mirror_left_right(sample)
    assert torch.equal(mirrored["mask"], (mirrored["anatomy"] == 5).long())
