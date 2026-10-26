"""Synthetic checks for the separately identified cache appearance study."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments import appearance as ap


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        self.image = np.array([0, -4, -1, -0.1, 0.1, 1, 2, 4], dtype=np.float32).reshape(1, 2, 2, 2)

    def test_parameters_reproduce_and_gamma_matches_between_arms(self):
        first = ap.deterministic_params(42, 3, "B006/Week0", "gamma_only")
        np.random.seed(998)
        np.random.random(23)
        second = ap.deterministic_params(42, 3, "B006/Week0", "gamma_polarity")
        self.assertEqual(first["gamma"], second["gamma"])
        self.assertEqual(first["gamma_draw"], second["gamma_draw"])
        self.assertEqual(first, ap.deterministic_params(42, 3, "B006/Week0", "gamma_only"))
        self.assertFalse(first["polarity"])
        draws = [ap.deterministic_params(42, epoch, "B006/Week0", "gamma_polarity") for epoch in range(1, 41)]
        self.assertGreater(len({value["gamma_draw"] for value in draws}), 1)
        self.assertTrue(any(value["polarity"] for value in draws))
        self.assertTrue(any(not value["polarity"] for value in draws))
        self.assertTrue(all(0.7 <= value["gamma_draw"] <= 1.5 for value in draws))

    def test_transform_preserves_zero_support_shape_dtype_and_input(self):
        before = self.image.copy()
        result = ap.augment_image(self.image, gamma=1.5, polarity=True)
        np.testing.assert_array_equal(self.image, before)
        np.testing.assert_array_equal(result != 0, before != 0)
        self.assertEqual(result.shape, before.shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(float(np.abs(result).max()), 5.0)
        self.assertFalse(np.shares_memory(result, self.image))
        tiny = np.array([0, np.nextafter(np.float32(0), np.float32(1)), -1e-30], dtype=np.float32).reshape(1, 1, 3)
        np.testing.assert_array_equal(ap.augment_image(tiny, gamma=1.5) != 0, tiny != 0)

    def test_signed_gamma_preserves_order_and_only_polarity_reverses_it(self):
        values = np.array([-2, -1, 0, 1, 2], dtype=np.float32).reshape(1, 1, 5)
        for gamma in (0.7, 1.5):
            self.assertTrue(np.all(np.diff(ap.augment_image(values, gamma).ravel()) > 0))
        reversed_values = ap.augment_image(values, gamma=0.7, polarity=True)
        self.assertTrue(np.all(np.diff(reversed_values.ravel()) < 0))

    def test_stress_conditions_are_separate_and_spatial_gain_is_h_axis(self):
        for condition in ap.STRESS_CONDITIONS:
            result = ap.apply_stress(self.image, condition)
            np.testing.assert_array_equal(result != 0, self.image != 0)
            self.assertEqual(result.shape, self.image.shape)
            self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(ap.apply_stress(self.image, "clean"), self.image)
        image = np.ones((1, 2, 3, 4), dtype=np.float32)
        changed = ap.apply_stress(image, "spatial_gain")
        np.testing.assert_allclose(changed[0, 0, :, 0], [0.7, 1.0, 1.3])
        np.testing.assert_array_equal(changed[0, 0], changed[0, 1])
        with self.assertRaises(ValueError):
            ap.apply_stress(image, "unregistered")

    def test_invalid_inputs_are_rejected(self):
        for bad in (np.full((2, 2, 2), np.nan, dtype=np.float32), np.ones((2, 2, 2), dtype=np.float64)):
            with self.assertRaises(ValueError):
                ap.augment_image(bad)
        for epoch in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                ap.deterministic_params(42, epoch, "case", "gamma_only")
        with self.assertRaises(ValueError):
            ap.deterministic_params(42, 1, "case", "unknown")

    def test_wrapper_preserves_labels_spacing_and_calls_original_flipped_base(self):
        mask = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
        spacing = np.array([2, 1, 1], dtype=np.float32)
        sample = {"image": self.image, "mask": mask, "spacing": spacing, "patient": "B006", "visit": "Week0"}
        class Base:
            records = [{"patient": "B006"}]
            def __len__(self): return 1
            def __getitem__(self, index): return sample
        fake = types.SimpleNamespace(HipMRIVolumeDataset=mock.Mock(return_value=Base()))
        with mock.patch.dict(sys.modules, {"dataset": fake}):
            wrapper = ap.AppearanceTrainingDataset("unused", "gamma_polarity", seed=42)
        fake.HipMRIVolumeDataset.assert_called_once_with("unused", "train", augment=True)
        wrapper.set_epoch(4)
        result = wrapper[0]
        self.assertIs(result["mask"], mask)
        self.assertIs(result["spacing"], spacing)
        self.assertEqual((result["patient"], result["visit"]), ("B006", "Week0"))
        self.assertIsNot(result, sample)
        np.testing.assert_array_equal(sample["image"], self.image)
        self.assertEqual(len(wrapper), 1)
        self.assertEqual(wrapper.records, Base.records)

    def test_split_guards_reject_test_and_training_stress(self):
        for split in ("validation", "test"):
            with self.assertRaisesRegex(ValueError, "TRAIN-only"):
                ap.AppearanceTrainingDataset("unused", "gamma_only", split=split)
        for split, augment in (("test", False), ("train", False), ("validation", True)):
            with self.assertRaisesRegex(ValueError, "unaugmented validation"):
                ap.StressValidationDataset(types.SimpleNamespace(split=split, augment=augment), "clean")
        sample = {"image": self.image, "mask": np.ones((2, 2, 2), dtype=np.uint8)}
        class Validation:
            split, augment, records = "validation", False, ["one"]
            def __len__(self): return 1
            def __getitem__(self, index): return sample
        wrapper = ap.StressValidationDataset(Validation(), "polarity")
        self.assertIs(wrapper[0]["mask"], sample["mask"])
        np.testing.assert_array_equal(wrapper[0]["image"], -self.image)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch not installed; NumPy policy tests still run")
class TorchAppearanceTests(unittest.TestCase):
    def test_tensor_backend_matches_numpy_and_preserves_device_dtype(self):
        import torch
        values = np.array([0, -4, -0.01, 0.1, 2, 4], dtype=np.float32).reshape(1, 1, 2, 3)
        tensor = torch.from_numpy(values.copy())
        for condition in ap.STRESS_CONDITIONS:
            actual = ap.apply_stress(tensor, condition)
            self.assertEqual(actual.dtype, tensor.dtype)
            self.assertEqual(actual.device, tensor.device)
            self.assertNotEqual(actual.data_ptr(), tensor.data_ptr())
            np.testing.assert_allclose(actual.numpy(), ap.apply_stress(values, condition), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(tensor, torch.from_numpy(values))


if __name__ == "__main__":
    unittest.main()
