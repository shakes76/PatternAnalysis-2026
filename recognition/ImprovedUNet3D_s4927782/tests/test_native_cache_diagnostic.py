"""Lightweight numerical contracts; optional real synthetic NIfTI integration."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "tools" / "diagnose_native_cache.py"
SPEC = importlib.util.spec_from_file_location("diagnose_native_cache", PATH)
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


class NativeDiagnosticTests(unittest.TestCase):
    def test_nonzero_normalization_and_zero_support(self):
        image = np.array([0, 1, 2, 3], dtype=np.float32)
        normalized, support, stats = diag.zscore_nonzero(image)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(normalized[0], 0)
        self.assertEqual(support.tolist(), [False, True, True, True])
        self.assertEqual(stats["mean"], 2)
        self.assertAlmostEqual(float(normalized[1]), -1.2247449, places=6)
        empty, _, stats = diag.zscore_nonzero(np.zeros((2, 2), dtype=np.float32))
        self.assertFalse(empty.any())
        self.assertEqual(stats["standard_deviation"], 0)
        with self.assertRaises(ValueError):
            diag.zscore_nonzero(np.array([np.nan], dtype=np.float32))

    def test_raw_centroid_projects_through_ras_affines_and_dhw_reverse(self):
        raw = np.zeros((4, 4, 4), dtype=bool); raw[1, 2, 3] = True
        raw_affine = np.diag([-1.0, -2.0, 3.0, 1.0])
        raw_affine[:3, 3] = [3, 6, 0]
        cached = np.zeros((4, 4, 4), dtype=bool); cached[3, 1, 2] = True
        cache_affine = np.diag([1.0, 2.0, 3.0, 1.0])
        result = diag.project_centroid(raw, raw_affine, cached, cache_affine)
        self.assertEqual(result["native_projected_to_cache_voxel_xyz"], [2.0, 1.0, 3.0])
        self.assertEqual(result["centroid_distance_mm"], 0)

    def test_discrete_centroid_change_is_reported_without_alignment_verdict(self):
        raw = np.ones((2, 1, 1), dtype=bool)
        cached = np.ones((1, 1, 1), dtype=bool)
        result = diag.project_centroid(raw, np.eye(4), cached, np.eye(4))
        self.assertEqual(result["centroid_distance_mm"], 0.5)
        self.assertNotIn("alignment_pass", result)

    def test_exact_mask_differences_and_float_tolerance_are_separate(self):
        image = np.ones((2, 2, 2), dtype=np.float32)
        original = np.zeros(image.shape, dtype=np.uint8); original[0, 0, 0] = 1
        changed = original.copy(); changed[0, 0, 0] = 0; changed[1, 1, 1] = 1
        report = diag.compare_arrays(image + 1e-6, changed, image, original, 2, 1)
        self.assertFalse(report["mask_exact_match"])
        self.assertEqual(report["mask_different_voxels"], 2)
        self.assertEqual(report["target_mask_different_voxels"], 2)
        self.assertTrue(report["image_allclose"])

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for path in ("../x.nii.gz", "/x.nii.gz", "a/../b.npz", "a\\b.npz"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    diag.safe_path(Path(folder).resolve(), path)

    def test_default_selection_is_twelve_fixed_cases_with_depth_exception(self):
        self.assertEqual(len(diag.DEFAULT_CASES), 12)
        self.assertEqual(diag.DEFAULT_CASES[:8], [f"M015/Week{i}" for i in range(8)])
        self.assertIn("K019/Week1", diag.DEFAULT_CASES)
        self.assertEqual(len(set(diag.DEFAULT_CASES)), 12)


@unittest.skipUnless(importlib.util.find_spec("nibabel") and importlib.util.find_spec("scipy"),
                     "nibabel and scipy required for synthetic NIfTI integration")
class SyntheticNiftiIntegrationTests(unittest.TestCase):
    def test_independent_affine_resample_matches_declared_zoom_grid(self):
        import nibabel as nib
        from scipy.ndimage import zoom
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data = root / "data"; data.mkdir(); output = root / "out"; output.mkdir()
            affine = np.diag([-1.0, -1.0, 2.0, 1.0]); affine[:2, 3] = [7, 7]
            image = np.arange(512, dtype=np.float32).reshape(8, 8, 8) + 1
            mask = np.zeros((8, 8, 8), dtype=np.int16); mask[2:6, 2:6, 2:6] = 5
            image_path, mask_path = data / "image.nii.gz", data / "mask.nii.gz"
            nib.save(nib.Nifti1Image(image, affine), image_path)
            nib.save(nib.Nifti1Image(mask, affine), mask_path)
            canonical = nib.as_closest_canonical(nib.load(image_path))
            canonical_mask = np.asanyarray(nib.as_closest_canonical(nib.load(mask_path)).dataobj)
            normalized, support, _ = diag.zscore_nonzero(canonical.get_fdata(dtype=np.float32))
            cached_image = zoom(normalized, (0.5, 0.5, 0.5), order=1, mode="nearest", grid_mode=True, prefilter=False)
            cached_image[~zoom(support.astype(np.uint8), (0.5, 0.5, 0.5), order=0, mode="nearest", grid_mode=True, prefilter=False).astype(bool)] = 0
            cached_mask = zoom((canonical_mask == 5).astype(np.uint8), (0.5, 0.5, 0.5), order=0, mode="nearest", grid_mode=True, prefilter=False)
            transform = np.diag([2.0, 2.0, 2.0, 1.0]); transform[:3, 3] = 0.5
            target_affine = canonical.affine @ transform
            np.savez_compressed(root / "cache.npz", image=cached_image.transpose(2, 1, 0), mask=cached_mask.transpose(2, 1, 0), spacing=np.array([4, 2, 2], dtype=np.float32))
            record = {"source_image": "image.nii.gz", "source_mask": "mask.nii.gz", "cache_path": "cache.npz", "split": "train",
                      "resampled_affine_xyz": target_affine.tolist(), "class_voxel_counts": np.bincount(cached_mask.ravel(), minlength=2).tolist()}
            for key, volume in [("original_image", nib.load(image_path)), ("original_mask", nib.load(mask_path)),
                                ("canonical_image", canonical), ("canonical_mask", nib.as_closest_canonical(nib.load(mask_path)))]:
                record[key] = {"shape_xyz": list(volume.shape), "affine_xyz": volume.affine.tolist()}
            manifest = {"label_ids": [0, 5], "target_shape_dhw": [4, 4, 4]}
            with mock.patch.object(diag, "plot_alignment", return_value=[]):
                result = diag.diagnose_case("Synthetic/Week0", record, manifest, data, root, output)
            self.assertTrue(result["comparison"]["mask_exact_match"])
            self.assertTrue(result["comparison"]["image_allclose"])
            self.assertTrue(result["secondary_rounded_coefficient_comparison"]["mask_exact_match"])
            self.assertTrue(all(result["invariants"].values()))
            self.assertEqual(result["native_shape_xyz"], [8, 8, 8])
            self.assertEqual(result["native_orientation"], ["L", "P", "S"])


if __name__ == "__main__":
    unittest.main()
