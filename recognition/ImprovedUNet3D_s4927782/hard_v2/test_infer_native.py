"""Independent image-only geometry fixtures; no clinical accuracy assertions."""
import importlib.util
from itertools import product
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import infer_native as adapter

HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_NIB = importlib.util.find_spec("nibabel") is not None
REFERENCE_PROJECT = Path(os.environ.get("HIPMRI_V1_PROJECT_ROOT", str(adapter.DEFAULT_PROJECT)))


class AffineTests(unittest.TestCase):
    def test_finite_flipped_anisotropic_oblique_affine_is_accepted(self):
        angle = .23
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        affine = np.eye(4)
        affine[:3, :3] = rotation @ np.diag([-1.2, 2.3, 4.1])
        affine[:3, 3] = [17, -23, 4]
        np.testing.assert_allclose(adapter.validate_affine(affine), [1.2, 2.3, 4.1])

    def test_shear_singular_and_nonfinite_affines_are_rejected(self):
        bad = []
        shear = np.eye(4); shear[0, 1] = .05; bad.append(shear)
        singular = np.eye(4); singular[1, 1] = 0; bad.append(singular)
        nonfinite = np.eye(4); nonfinite[0, 3] = np.nan; bad.append(nonfinite)
        for affine in bad:
            with self.subTest(affine=affine), self.assertRaises(ValueError):
                adapter.validate_affine(affine)

    def test_center_and_outer_cell_edges_preserve_world_coordinates(self):
        source_shape = np.array([9, 7, 5])
        target_dhw = (3, 5, 7)
        target_xyz = np.array(target_dhw[::-1])
        affine = np.diag([1.2, 2.3, 4.1, 1.])
        affine[:3, 3] = [17, -23, 4]
        cache_affine = adapter.grid_transform(affine, source_shape, target_dhw)
        for index in [np.array([-.5] * 3), target_xyz - .5, (target_xyz - 1) / 2]:
            old_index = (index + .5) * source_shape / target_xyz - .5
            expected_world = affine @ np.r_[old_index, 1.]
            actual_world = cache_affine @ np.r_[index, 1.]
            np.testing.assert_allclose(actual_world, expected_world, rtol=0, atol=1e-12)

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required for nearest affine sampling")
    def test_native_restoration_matches_independent_world_coordinate_sampler(self):
        native_shape = (9, 7, 5)
        canonical_shape = np.array(native_shape)
        canonical_affine = np.diag([1.2, 2.3, 4.1, 1.])
        canonical_affine[:3, 3] = [-11, 19, -3]
        target_dhw = (3, 5, 7)
        cache_affine = adapter.grid_transform(canonical_affine, canonical_shape, target_dhw)
        # Seven distinct labels encode cache-X; their raw IDs are noncontiguous.
        prediction = np.broadcast_to(np.arange(7, dtype=np.uint8), target_dhw).copy()
        raw_ids = [0, 4, 8, 12, 16, 20, 24]
        for flips in product((False, True), repeat=3):
            orientation = np.eye(4)
            for axis, flip in enumerate(flips):
                if flip:
                    orientation[axis, axis] = -1
                    orientation[axis, 3] = native_shape[axis] - 1
            native_affine = canonical_affine @ orientation
            original = SimpleNamespace(shape=native_shape, affine=native_affine)
            restored, returned_transform = adapter.restore_prediction(
                prediction, original, cache_affine, raw_ids)
            expected = np.empty(native_shape, dtype=np.uint8)
            for index in np.ndindex(native_shape):
                world = native_affine @ np.r_[index, 1.]
                cache_coord = np.linalg.solve(cache_affine, world)[:3]
                sampled = np.floor(cache_coord + .5).astype(int)
                sampled = np.clip(sampled, 0, np.array(target_dhw[::-1]) - 1)
                expected[index] = raw_ids[int(prediction[sampled[2], sampled[1], sampled[0]])]
                # Verify inverse-affine world-coordinate identity as well.
                np.testing.assert_allclose(cache_affine @ returned_transform @ np.r_[index, 1.], world,
                                           rtol=0, atol=1e-11)
            with self.subTest(flips=flips):
                np.testing.assert_array_equal(restored, expected)

    def test_cli_refuses_existing_output_before_loading_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SystemExit) as caught:
                adapter.main(["--image", "missing_image.nii.gz", "--checkpoint", "missing.pt",
                              "--output-dir", temporary])
            self.assertEqual(caught.exception.code, 2)


@unittest.skipUnless(HAS_NIB and HAS_SCIPY, "nibabel and SciPy required for real NIfTI fixtures")
class NiftiImageOnlyTests(unittest.TestCase):
    def setUp(self):
        import nibabel as nib
        self.nib = nib
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_image(self, affine=None, shape=(9, 7, 5), zero=False):
        image = np.zeros(shape, dtype=np.float32) if zero else np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        image[0] = 0
        affine = np.diag([-1.2, -2.3, 4.1, 1.]) if affine is None else affine
        volume = self.nib.Nifti1Image(image, affine)
        volume.header.set_xyzt_units("mm")
        path = self.root / "image_only.nii.gz"
        self.nib.save(volume, path)
        return path

    def test_no_mask_exists_or_is_loaded_and_original_geometry_is_restored(self):
        image_path = self.write_image()
        self.assertEqual(list(self.root.iterdir()), [image_path])
        with mock.patch.object(self.nib, "load", wraps=self.nib.load) as load:
            image, original, geometry = adapter.preprocess_image(image_path, (3, 5, 7))
        self.assertEqual([str(c.args[0]) for c in load.call_args_list], [str(image_path)])
        self.assertEqual(image.shape, (3, 5, 7))
        prediction = (image > 0).astype(np.uint8)
        output = self.root / "outputs"
        adapter.save_outputs(output, prediction, original, geometry, [0, 5], {"fixture": True})
        restored = self.nib.load(output / "prediction_native.nii.gz")
        self.assertEqual(restored.shape, original.shape)
        np.testing.assert_allclose(restored.affine, original.affine, rtol=0, atol=1e-6)
        self.assertTrue(set(np.unique(np.asanyarray(restored.dataobj))).issubset({0, 5}))
        with self.assertRaises(FileExistsError):
            adapter.save_outputs(output, prediction, original, geometry, [0, 5], {})

    def test_permutation_flip_and_oblique_original_affines_roundtrip(self):
        angle = .17
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        linear = rotation @ np.array([[0., 0., -4.1], [2.3, 0., 0.], [0., -1.2, 0.]])
        affine = np.eye(4); affine[:3, :3] = linear; affine[:3, 3] = [31, -27, 5]
        path = self.write_image(affine=affine)
        image, original, geometry = adapter.preprocess_image(path, (3, 5, 7))
        self.assertEqual(image.shape, (3, 5, 7))
        canonical = self.nib.as_closest_canonical(original)
        self.assertEqual(self.nib.aff2axcodes(canonical.affine), ("R", "A", "S"))
        original_to_cache = np.linalg.solve(geometry["cache_affine_xyz"], original.affine)
        for point in [(0, 0, 0), (8, 6, 4), (4, 3, 2)]:
            np.testing.assert_allclose(np.asarray(geometry["cache_affine_xyz"]) @ original_to_cache @ np.r_[point, 1.],
                                       original.affine @ np.r_[point, 1.], rtol=0, atol=1e-10)

    def test_preprocessing_matches_frozen_v1_pair_function_elementwise(self):
        path = self.write_image()
        mask = self.root / "synthetic_mask_for_reference_only.nii.gz"
        original = self.nib.load(path)
        self.nib.save(self.nib.Nifti1Image(np.zeros(original.shape, dtype=np.uint8), original.affine), mask)
        source = REFERENCE_PROJECT / "tools" / "prepare_cache.py"
        # The production adapter never loads this reference or its mask.
        sys.path.insert(0, str(REFERENCE_PROJECT))
        spec = importlib.util.spec_from_file_location("frozen_v1_prepare_reference", source)
        reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
        expected, metadata = reference.preprocess_pair(path, mask, (3, 5, 7), [0, 5])
        actual, _, geometry = adapter.preprocess_image(path, (3, 5, 7))
        np.testing.assert_array_equal(actual, expected["image"])
        np.testing.assert_allclose(geometry["cache_affine_xyz"], metadata["resampled_affine_xyz"], rtol=0, atol=0)
        np.testing.assert_allclose(geometry["cache_spacing_dhw"], expected["spacing"], rtol=1e-7)

    def test_empty_image_has_finite_zero_normalization(self):
        path = self.write_image(zero=True)
        image, _, geometry = adapter.preprocess_image(path, (3, 5, 7))
        self.assertEqual(geometry["normalization"]["nonzero_count"], 0)
        np.testing.assert_array_equal(image, np.zeros((3, 5, 7), dtype=np.float32))

    def test_nonfinite_image_and_sheared_grid_are_rejected(self):
        path = self.write_image()
        original = self.nib.load(path)
        array = original.get_fdata(dtype=np.float32); array[1, 1, 1] = np.nan
        self.nib.save(self.nib.Nifti1Image(array, original.affine), path)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            adapter.preprocess_image(path, (3, 5, 7))
        shear = np.eye(4); shear[0, 1] = .1
        path = self.write_image(affine=shear)
        with self.assertRaisesRegex(ValueError, "Sheared"):
            adapter.preprocess_image(path, (3, 5, 7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
