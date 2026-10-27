"""NIfTI geometry regression checks using constructed objects, not MRI data."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dataset import AFFINE_ATOL, load_pair, validate_pair_geometry


class GeometryChecks(unittest.TestCase):
    @staticmethod
    def pair():
        image = np.arange(32, dtype=np.float32).reshape(8, 4)
        mask = np.arange(32, dtype=np.int16).reshape(8, 4) % 6
        affine = np.diag([0.75, 0.75, 1.0, 1.0])
        return nib.Nifti1Image(image, affine), nib.Nifti1Image(mask, affine)

    def test_matching_geometry_preserves_label_ids_and_preprocessing(self):
        image, mask = self.pair()
        with patch("dataset.nib.load", side_effect=[image, mask]):
            sample = load_pair("constructed_image", "constructed_mask")
        self.assertEqual(sample["original_size"], (8, 4))
        self.assertEqual(tuple(sample["image"].shape), (1, 16, 16))
        self.assertEqual(sample["valid_mask"].sum().item(), 32)
        torch.testing.assert_close(
            sample["target"][:8, :4], torch.from_numpy(mask.get_fdata()).long(),
            rtol=0, atol=0,
        )

    def test_same_shape_translation_mismatch_fails_before_pixel_reads(self):
        image, mask = self.pair()
        mask.affine[0, 3] += 100
        with patch("dataset.nib.load", side_effect=[image, mask]), \
                patch.object(image, "get_fdata", side_effect=AssertionError("pixel read")), \
                patch.object(mask, "get_fdata", side_effect=AssertionError("pixel read")):
            with self.assertRaisesRegex(ValueError, "affine mismatch"):
                load_pair("constructed_image", "constructed_mask")

    def test_nonfinite_affine_on_either_side_fails_before_pixel_reads(self):
        for side in (0, 1):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(side=side, value=value):
                    image, mask = self.pair()
                    (image, mask)[side].affine[0, 3] = value
                    with patch("dataset.nib.load", side_effect=[image, mask]), \
                            patch.object(image, "get_fdata", side_effect=AssertionError("pixel read")), \
                            patch.object(mask, "get_fdata", side_effect=AssertionError("pixel read")):
                        with self.assertRaisesRegex(ValueError, "finite 4x4"):
                            load_pair("constructed_image", "constructed_mask")

    def test_absolute_tolerance_accepts_roundoff_but_rejects_larger_difference(self):
        image, mask = self.pair()
        mask.affine[0, 3] = AFFINE_ATOL / 2
        self.assertAlmostEqual(validate_pair_geometry(image, mask), AFFINE_ATOL / 2)
        mask.affine[0, 3] = AFFINE_ATOL * 2
        with self.assertRaisesRegex(ValueError, "affine mismatch"):
            validate_pair_geometry(image, mask)

    def test_large_origin_does_not_enable_relative_tolerance(self):
        image, mask = self.pair()
        image.affine[0, 3] = 1e6
        mask.affine[0, 3] = 1e6 + 0.1
        with self.assertRaisesRegex(ValueError, "affine mismatch"):
            validate_pair_geometry(image, mask)

    def test_shape_mismatch_is_rejected_before_pixel_reads(self):
        image, _ = self.pair()
        mask = nib.Nifti1Image(np.zeros((4, 8), dtype=np.int16), np.eye(4))
        with patch("dataset.nib.load", side_effect=[image, mask]), \
                patch.object(image, "get_fdata", side_effect=AssertionError("pixel read")):
            with self.assertRaisesRegex(ValueError, "matching 2D"):
                load_pair("constructed_image", "constructed_mask")


if __name__ == "__main__":
    unittest.main()
