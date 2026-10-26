"""Synthetic policy-information checks; no native HipMRI pixels or model runs."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("source_inputs", ROOT / "tools/diagnose_source_inputs.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


class SourceInputStatisticsTests(unittest.TestCase):
    def test_quantiles_zero_and_clip_fractions(self):
        result = tool.describe(np.array([-5., 0., 0., 5.]), [-5, 5])
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["mean"], 0)
        self.assertEqual(result["zero_fraction"], .5)
        self.assertEqual(result["at_or_beyond_clip_fraction"], .5)
        self.assertAlmostEqual(result["quantiles"]["q25"], -1.25)

    def test_empty_class_is_null_and_raw_clip_is_not_applicable(self):
        stats = tool.class_statistics(np.ones((2, 2)), np.zeros((2, 2), dtype=np.uint8))
        self.assertEqual(stats["5"]["n"], 0)
        self.assertIsNone(stats["5"]["mean"])
        self.assertIsNone(stats["0"]["at_or_beyond_clip_fraction"])

    def test_nonfinite_and_fractional_labels_rejected(self):
        with self.assertRaises(ValueError):
            tool.describe([np.nan])
        for values in ([0, .5], [0, 6], [0, np.inf]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                tool.map_labels(values)

    def test_normalization_preserves_zero_support_not_a_gt_mask(self):
        values = np.array([0., 2., 4.], dtype=np.float32)
        normalized, support, stats = tool.normalize_nonzero(values)
        np.testing.assert_array_equal(support, [False, True, True])
        np.testing.assert_array_equal(normalized, [0., -1., 1.])
        self.assertEqual(stats["mean"], 3)
        self.assertEqual(stats["std"], 1)

    def test_support_forcing_distinguishes_outside_from_actual_changed(self):
        before = np.array([2., 0., 3., 4.])
        support = np.array([False, False, True, False])
        mask = np.array([5, 5, 5, 1], dtype=np.uint8)
        after, result = tool.support_effect(before, support, mask)
        np.testing.assert_array_equal(after, [0, 0, 3, 0])
        np.testing.assert_array_equal(before, [2, 0, 3, 4])
        self.assertEqual(result["target_outside_resampled_nonzero_support"], 2)
        self.assertEqual(result["forced_zero_nonzero_voxels_target"], 1)
        self.assertEqual(result["forced_zero_nonzero_voxels_all"], 2)

    def test_test_pixel_scope_requires_ack_and_only_m015(self):
        records = [{"patient": "M015", "visit": "Week0", "split": "test"},
                   {"patient": "B006", "visit": "Week0", "split": "train"},
                   {"patient": "B037", "visit": "Week0", "split": "test"}]
        with self.assertRaises(ValueError):
            tool.select_cases(records, ["M015/Week0"])
        self.assertEqual(len(tool.select_cases(records, ["M015/Week0"], True)), 1)
        self.assertEqual(len(tool.select_cases(records, ["B006/Week0"])), 1)
        for names in (["B037/Week0"], ["M015/Week0", "M015/Week0"], []):
            with self.subTest(names=names), self.assertRaises(ValueError):
                tool.select_cases(records, names, True)

    def test_two_exploratory_train_controls_extend_scope_without_new_test(self):
        self.assertEqual(len(tool.ALLOWED_CASES), 14)
        records = [{"patient": "R016", "visit": "Week7", "split": "train"},
                   {"patient": "T014", "visit": "Week3", "split": "train"}]
        self.assertEqual(len(tool.select_cases(records, ["R016/Week7", "T014/Week3"])), 2)
        records[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "retain their TRAIN"):
            tool.select_cases(records, ["R016/Week7"])

    def test_header_text_is_hashed_with_only_whitelisted_hints(self):
        result = tool.text_field_evidence({"descrip": np.array(b"PRIVATE_NAME_t2w_tse DWI")}, "descrip")
        self.assertEqual(result["sequence_hints"], ["T2", "DWI"])
        self.assertFalse(result["free_text_exported"])
        self.assertNotIn("PRIVATE_NAME", str(result))

    def test_median_contrast_and_body_iqr_normalization_keep_sign(self):
        image = np.array([1., 3., -2., -1., 0., .5])
        mask = np.array([1, 1, 3, 3, 5, 5], dtype=np.uint8)
        first = tool.class_contrasts(tool.class_statistics(image, mask))
        changed = tool.class_contrasts(tool.class_statistics(image * 7 + 100, mask))
        self.assertEqual(first["bladder_minus_body_median"], -3.5)
        self.assertEqual(first["bladder_minus_body_median_div_body_iqr"], -3.5)
        self.assertEqual(first["bladder_minus_body_median_div_body_iqr"],
                         changed["bladder_minus_body_median_div_body_iqr"])


@unittest.skipIf(nib is None, "NiBabel unavailable; real NIfTI synthetic fixtures skipped")
class NiftiSourceInputTests(unittest.TestCase):
    def test_saved_int16_proxy_scaling_is_recorded_after_header_consumption(self):
        raw = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        affine = np.diag([2., 3., 4., 1.])
        volume = nib.Nifti1Image(raw, affine)
        volume.header.set_slope_inter(2., 10.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_scaled.nii.gz"
            nib.save(volume, path)
            loaded = nib.load(path)
            # The loaded header consumes scaling; its proxy retains the values.
            self.assertEqual(loaded.header.get_slope_inter(), (None, None))
            self.assertEqual(loaded.dataobj.slope, 2.)
            self.assertEqual(loaded.dataobj.inter, 10.)
            np.testing.assert_array_equal(loaded.get_fdata(), raw.astype(np.float64) * 2 + 10)
            result = tool.inspect_header(path, {"shape_xyz": list(raw.shape),
                                               "affine_xyz": affine.tolist()}, nib)
            self.assertEqual(result["proxy_slope"], 2.)
            self.assertEqual(result["proxy_inter"], 10.)
            self.assertEqual(np.dtype(result["data_dtype"]), np.dtype(np.int16))
            self.assertTrue(result["geometry_matches_frozen_record"])

    def test_nonras_canonical_permutation_preserves_each_class_roi(self):
        image = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + .25
        mask = (np.arange(24).reshape(2, 3, 4) % 6).astype(np.uint8)
        # Input axes point A, L, I: canonicalization must reorder and flip.
        affine = np.array([[0., -2., 0., 20.], [3., 0., 0., -10.],
                           [0., 0., -4., 8.], [0., 0., 0., 1.]])
        with tempfile.TemporaryDirectory() as directory:
            image_path, mask_path = Path(directory) / "image.nii.gz", Path(directory) / "mask.nii.gz"
            nib.save(nib.Nifti1Image(image, affine), image_path)
            nib.save(nib.Nifti1Image(mask, affine), mask_path)
            original_image, original_mask = nib.load(image_path), nib.load(mask_path)
            self.assertEqual(nib.aff2axcodes(original_image.affine), ("A", "L", "I"))
            canonical_image = nib.as_closest_canonical(original_image)
            canonical_mask = nib.as_closest_canonical(original_mask)
            self.assertEqual(nib.aff2axcodes(canonical_image.affine), ("R", "A", "S"))
            self.assertEqual(canonical_image.shape, (3, 2, 4))
            np.testing.assert_array_equal(canonical_image.affine, canonical_mask.affine)
            before_image, before_mask = original_image.get_fdata(), np.asanyarray(original_mask.dataobj)
            after_image, after_mask = canonical_image.get_fdata(), np.asanyarray(canonical_mask.dataobj)
            for label in range(6):
                np.testing.assert_array_equal(np.sort(before_image[before_mask == label]),
                                              np.sort(after_image[after_mask == label]))
            self.assertEqual(tool.class_statistics(before_image, before_mask),
                             tool.class_statistics(after_image, after_mask))


if __name__ == "__main__":
    unittest.main()
