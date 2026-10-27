"""Constructed contract checks and optional real local training-data integration."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOCAL_DATA_ROOT = (
    PROJECT_DIR.parents[3]
    / "Final Report"
    / "HipMRI_2D_Improved_UNet_学习"
    / "data"
    / "keras_slices_data"
)
sys.path.insert(0, str(PROJECT_DIR))

from dataset import (
    IGNORE_INDEX,
    collate_padded,
    crop_to_original,
    load_pair,
    preprocess_pair,
    valid_pixels,
)


class PreprocessingChecks(unittest.TestCase):
    @staticmethod
    def sample(height=17, width=19):
        count = height * width
        image = torch.arange(1, count + 1, dtype=torch.float32).reshape(
            1, height, width
        )
        target = torch.arange(count, dtype=torch.int64).reshape(height, width) % 3
        return image, target

    def assert_padding_contract(self, sample):
        valid = sample["valid_mask"]
        self.assertEqual(valid.dtype, torch.bool)
        self.assertTrue(torch.all(sample["target"][~valid] == IGNORE_INDEX).item())
        self.assertTrue(torch.all(sample["target"][valid] >= 0).item())
        self.assertTrue(torch.all(sample["image"][:, ~valid] == 0).item())

    def test_padding_preserves_original_region_and_normalizes_before_padding(self):
        image, target = self.sample()
        original_image, original_target = image.clone(), target.clone()
        sample = preprocess_pair(image, target)

        self.assertEqual(tuple(sample["image"].shape), (1, 32, 32))
        self.assertEqual(tuple(sample["target"].shape), (32, 32))
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertEqual(sample["target"].dtype, torch.int64)
        self.assertEqual(tuple(sample["original_size"]), (17, 19))
        self.assertEqual(sample["valid_mask"].sum().item(), 323)
        self.assertTrue(sample["valid_mask"][:17, :19].all().item())
        self.assertTrue(torch.equal(sample["target"][:17, :19], target))
        # The input minimum is 1. Padding first would incorrectly make it 0.
        self.assertEqual(sample["image"][0, 0, 0].item(), 0.0)
        self.assertEqual(sample["image"][0, 8, 9].item(), 0.5)
        self.assertEqual(sample["image"][0, 16, 18].item(), 1.0)
        self.assertTrue(torch.equal(image, original_image))
        self.assertTrue(torch.equal(target, original_target))
        self.assert_padding_contract(sample)

    def test_constant_image_and_background_are_still_valid_pixels(self):
        image = torch.full((1, 17, 19), 5.0, dtype=torch.float32)
        target = torch.zeros((17, 19), dtype=torch.int64)
        sample = preprocess_pair(image, target)

        self.assertTrue(torch.equal(sample["image"], torch.zeros_like(sample["image"])))
        self.assertEqual(sample["valid_mask"].sum().item(), 323)
        self.assertTrue(torch.all(sample["target"][sample["valid_mask"]] == 0).item())
        self.assert_padding_contract(sample)

    def test_already_divisible_shape_does_not_gain_extra_rows_or_columns(self):
        image, target = self.sample(16, 32)
        sample = preprocess_pair(image, target)

        self.assertEqual(tuple(sample["image"].shape), (1, 16, 32))
        self.assertEqual(tuple(sample["target"].shape), (16, 32))
        self.assertTrue(sample["valid_mask"].all().item())
        self.assertTrue(torch.equal(sample["target"], target))

    def test_mixed_size_batch_preserves_each_samples_valid_region(self):
        first = preprocess_pair(*self.sample(17, 19))
        second = preprocess_pair(*self.sample(16, 32))
        batch = collate_padded([first, second])

        self.assertEqual(tuple(batch["image"].shape), (2, 1, 32, 32))
        self.assertEqual(tuple(batch["target"].shape), (2, 32, 32))
        self.assertEqual(tuple(batch["valid_mask"].shape), (2, 32, 32))
        self.assertEqual(batch["valid_mask"].dtype, torch.bool)
        self.assertEqual(batch["original_sizes"], [(17, 19), (16, 32)])
        self.assertEqual(batch["valid_mask"].sum(dim=(1, 2)).tolist(), [323, 512])
        self.assertTrue(torch.all(batch["target"][~batch["valid_mask"]] == IGNORE_INDEX).item())
        self.assertTrue(torch.all(batch["target"][batch["valid_mask"]] >= 0).item())
        self.assertTrue(torch.all(batch["image"].masked_select(~batch["valid_mask"].unsqueeze(1)) == 0).item())
        for index, sample in enumerate((first, second)):
            original_size = batch["original_sizes"][index]
            self.assertTrue(torch.equal(
                crop_to_original(batch["target"][index], original_size),
                crop_to_original(sample["target"], original_size),
            ))

    def test_empty_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            collate_padded([])

    def test_invalid_padding_multiples_are_rejected(self):
        for multiple in (0, -16, 2.5, 16.0, True, False):
            with self.subTest(multiple=multiple):
                image, target = self.sample(2, 3)
                with self.assertRaises(ValueError):
                    preprocess_pair(image, target, multiple=multiple)

    def test_negative_original_labels_are_not_accepted_as_padding(self):
        for bad_label in (-1, IGNORE_INDEX):
            with self.subTest(bad_label=bad_label):
                image, target = self.sample(2, 3)
                target[0, 0] = bad_label
                with self.assertRaises(ValueError):
                    preprocess_pair(image, target)

    def test_crop_restores_original_image_target_and_valid_mask(self):
        image, target = self.sample()
        sample = preprocess_pair(image, target)

        cropped_image = crop_to_original(sample["image"], sample["original_size"])
        cropped_target = crop_to_original(sample["target"], sample["original_size"])
        cropped_valid = crop_to_original(sample["valid_mask"], sample["original_size"])
        self.assertEqual(tuple(cropped_image.shape), (1, 17, 19))
        self.assertTrue(torch.equal(cropped_target, target))
        self.assertTrue(cropped_valid.all().item())

    def test_non_finite_image_values_are_rejected(self):
        for bad_value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad_value=bad_value):
                image, target = self.sample(2, 3)
                image[0, 0, 0] = bad_value
                with self.assertRaises(ValueError):
                    preprocess_pair(image, target)

    def test_fractional_or_non_finite_nifti_mask_is_rejected_before_casting(self):
        for bad_value in (1.7, float("nan"), float("inf")):
            with self.subTest(bad_value=bad_value):
                image = np.arange(6, dtype=np.float64).reshape(2, 3)
                target = np.zeros((2, 3), dtype=np.float64)
                target[0, 0] = bad_value
                fake_image = nib.Nifti1Image(image, np.eye(4))
                fake_target = nib.Nifti1Image(target, np.eye(4))
                with patch("dataset.nib.load", side_effect=[fake_image, fake_target]):
                    with self.assertRaises(ValueError):
                        load_pair(Path("case_constructed.nii.gz"), Path("seg_constructed.nii.gz"))

    def test_cross_entropy_and_selected_pixels_ignore_padding(self):
        _, original_target = self.sample()
        sample = preprocess_pair(*self.sample())
        logits = torch.linspace(-1.0, 1.0, 3 * 32 * 32).reshape(1, 3, 32, 32)
        padded_target = sample["target"].unsqueeze(0)
        padded_loss = F.cross_entropy(logits, padded_target, ignore_index=IGNORE_INDEX)
        original_loss = F.cross_entropy(logits[:, :, :17, :19], original_target.unsqueeze(0))
        torch.testing.assert_close(padded_loss, original_loss)

        changed_logits = logits.clone()
        invalid = ~sample["valid_mask"]
        changed_logits[0, 0][invalid] = 1000.0
        changed_logits[0, 1][invalid] = -1000.0
        changed_logits[0, 2][invalid] = -500.0
        changed_loss = F.cross_entropy(changed_logits, padded_target, ignore_index=IGNORE_INDEX)
        torch.testing.assert_close(changed_loss, original_loss)

        prediction = logits.argmax(dim=1)[0]
        changed_prediction = prediction.clone()
        changed_prediction[invalid] = 99
        selected_prediction, selected_target = valid_pixels(
            prediction, sample["target"], sample["valid_mask"]
        )
        changed_selected, changed_target = valid_pixels(
            changed_prediction, sample["target"], sample["valid_mask"]
        )
        self.assertEqual(selected_target.numel(), 323)
        self.assertTrue(torch.equal(selected_target, original_target.reshape(-1)))
        self.assertTrue(torch.equal(selected_prediction, changed_selected))
        self.assertTrue(torch.equal(selected_target, changed_target))


@unittest.skipUnless(
    LOCAL_DATA_ROOT.is_dir(),
    "Real local HipMRI training data is unavailable; integration check skipped.",
)
class RealLocalTrainingIntegrationChecks(unittest.TestCase):
    def test_real_training_mixed_width_batch_preserves_labels_and_ignores_padding(self):
        """Real train files only: this is preprocessing integration, not training."""
        image_dir = LOCAL_DATA_ROOT / "keras_slices_train"
        mask_dir = LOCAL_DATA_ROOT / "keras_slices_seg_train"
        stems = ("004_week_0_slice_12", "019_week_1_slice_0")
        mask_paths = [mask_dir / f"seg_{stem}.nii.gz" for stem in stems]
        samples = [
            load_pair(image_dir / f"case_{stem}.nii.gz", mask_path)
            for stem, mask_path in zip(stems, mask_paths)
        ]
        self.assertEqual(samples[0]["original_size"], (256, 128))
        self.assertEqual(samples[1]["original_size"], (256, 144))
        with self.assertRaisesRegex(RuntimeError, "stack expects each tensor"):
            torch.utils.data.default_collate(samples)

        batch = collate_padded(samples)
        self.assertEqual(tuple(batch["image"].shape), (2, 1, 256, 144))
        self.assertEqual(tuple(batch["target"].shape), (2, 256, 144))
        self.assertEqual(batch["original_sizes"], [(256, 128), (256, 144)])
        self.assertEqual(batch["valid_mask"].sum(dim=(1, 2)).tolist(), [32768, 36864])
        self.assertEqual((~batch["valid_mask"]).sum(dim=(1, 2)).tolist(), [4096, 0])
        self.assertTrue(torch.all(batch["target"][~batch["valid_mask"]] == IGNORE_INDEX).item())
        self.assertTrue(torch.all(batch["target"][batch["valid_mask"]] >= 0).item())
        self.assertTrue(torch.all(
            batch["image"].masked_select(~batch["valid_mask"].unsqueeze(1)) == 0
        ).item())
        for index, mask_path in enumerate(mask_paths):
            original_mask = torch.from_numpy(nib.load(mask_path).get_fdata()).to(torch.int64)
            cropped_target = crop_to_original(
                batch["target"][index], batch["original_sizes"][index]
            )
            self.assertTrue(torch.equal(cropped_target, original_mask))


if __name__ == "__main__":
    unittest.main()
