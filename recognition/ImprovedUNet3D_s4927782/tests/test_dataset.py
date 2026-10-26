"""Check patient partition contracts, cached tensors and synthetic resampling."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import dataset

SPEC = importlib.util.spec_from_file_location("prepare_cache", PROJECT / "tools" / "prepare_cache.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def example_manifest():
    return {
        "num_classes": 2, "label_ids": [0, 7], "raw_label_to_index": {"0": 0, "7": 1},
        "target_shape_dhw": [4, 5, 6], "audit_sha256": "a" * 64,
        "patient_splits": {"train": ["B006"], "validation": ["B037"], "test": ["B100"]},
        "records": [
            {"patient": "B006", "visit": "Week0", "split": "train", "cache_path": "B006_Week0.npz"},
            {"patient": "B006", "visit": "Week12", "split": "train", "cache_path": "B006_Week12.npz"},
            {"patient": "B037", "visit": "Week0", "split": "validation", "cache_path": "B037_Week0.npz"},
            {"patient": "B100", "visit": "Week0", "split": "test", "cache_path": "B100_Week0.npz"},
        ],
        "partial": False, "development_only": False,
    }


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "manifest.json"

    def write(self, manifest):
        self.path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_distinct_visits_stay_with_their_patient(self):
        manifest = example_manifest()
        self.write(manifest)
        loaded = dataset.load_manifest(self.path)
        self.assertEqual(len(loaded["records"]), 4)
        self.assertEqual({record["split"] for record in loaded["records"] if record["patient"] == "B006"}, {"train"})

    def test_all_splits_are_checked_before_selecting_a_dataset(self):
        manifest = example_manifest()
        manifest["records"][-1]["patient"] = "B006"
        self.write(manifest)
        with self.assertRaisesRegex(ValueError, "split disagrees"):
            dataset.load_manifest(self.path)
        manifest = example_manifest()
        manifest["patient_splits"]["test"].append("B006")
        self.write(manifest)
        with self.assertRaisesRegex(ValueError, "overlap"):
            dataset.load_manifest(self.path)

    def test_noncontiguous_labels_require_exact_mapping(self):
        manifest = example_manifest()
        del manifest["raw_label_to_index"]
        self.write(manifest)
        with self.assertRaisesRegex(ValueError, "raw_label_to_index"):
            dataset.load_manifest(self.path)

    def test_duplicate_records_and_unsafe_cache_paths_are_rejected(self):
        manifest = example_manifest()
        manifest["records"].append(dict(manifest["records"][0]))
        self.write(manifest)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            dataset.load_manifest(self.path)
        for bad_path in ("../outside.npz", "/outside.npz"):
            manifest = example_manifest()
            manifest["records"][0]["cache_path"] = bad_path
            self.write(manifest)
            with self.subTest(path=bad_path), self.assertRaises(ValueError):
                dataset.load_manifest(self.path)

    def test_partial_manifest_is_explicitly_training_only_development_data(self):
        manifest = example_manifest()
        manifest.update(partial=True, development_only=True)
        manifest["records"] = manifest["records"][:1]
        self.write(manifest)
        self.assertTrue(dataset.load_manifest(self.path)["development_only"])
        manifest["development_only"] = False
        self.write(manifest)
        with self.assertRaises(ValueError):
            dataset.load_manifest(self.path)


@unittest.skipIf(dataset.torch is None or dataset.np is None, "PyTorch and NumPy required for cache tensor tests")
class DatasetTensorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = example_manifest()
        self.path = self.root / "manifest.json"
        self.path.write_text(json.dumps(self.manifest))
        for record in self.manifest["records"]:
            mask = (dataset.np.arange(120).reshape(4, 5, 6) % 2).astype(dataset.np.uint8)
            dataset.np.savez_compressed(
                self.root / record["cache_path"], image=mask.astype(dataset.np.float32),
                mask=mask, spacing=dataset.np.array([3.0, 2.0, 1.0], dtype=dataset.np.float32),
            )

    def test_volume_shapes_dtypes_spacing_and_metadata(self):
        volumes = dataset.HipMRIVolumeDataset(self.path, "train")
        self.assertEqual(len(volumes), 2)
        sample = volumes[0]
        self.assertEqual(tuple(sample["image"].shape), (1, 4, 5, 6))
        self.assertEqual(tuple(sample["mask"].shape), (4, 5, 6))
        self.assertEqual(sample["image"].dtype, dataset.torch.float32)
        self.assertEqual(sample["mask"].dtype, dataset.torch.long)
        self.assertEqual(sample["spacing"].tolist(), [3.0, 2.0, 1.0])
        self.assertEqual((sample["patient"], sample["visit"]), ("B006", "Week0"))

    def test_slice_indexing_covers_every_visit_at_matching_depth(self):
        slices = dataset.HipMRISliceDataset(self.path, "train")
        volumes = dataset.HipMRIVolumeDataset(self.path, "train")
        self.assertEqual(len(slices), 8)
        self.assertEqual(len(slices.records), 2)
        sample = slices[5]
        self.assertEqual((sample["visit"], sample["slice_index"]), ("Week12", 1))
        self.assertEqual(tuple(sample["image"].shape), (1, 5, 6))
        self.assertTrue(dataset.torch.equal(sample["mask"], volumes[1]["mask"][1]))

    def test_validation_and_test_augmentation_are_rejected(self):
        for cls in (dataset.HipMRIVolumeDataset, dataset.HipMRISliceDataset):
            for split in ("validation", "test"):
                with self.subTest(cls=cls, split=split), self.assertRaises(ValueError):
                    cls(self.path, split, augment=True)

    def test_random_flips_are_synchronised_for_image_and_mask(self):
        volumes = dataset.HipMRIVolumeDataset(self.path, "train", augment=True)
        unmodified = dataset.HipMRIVolumeDataset(self.path, "train")[0]
        with mock.patch.object(dataset.torch, "rand", side_effect=[dataset.torch.tensor(0.1), dataset.torch.tensor(0.9)]):
            sample = volumes[0]
        self.assertTrue(dataset.torch.equal(sample["image"][0], sample["mask"].float()))
        self.assertTrue(dataset.torch.equal(sample["mask"], dataset.torch.flip(unmodified["mask"], [-2])))

    def test_cache_rejects_undeclared_class(self):
        record = self.manifest["records"][0]
        dataset.np.savez_compressed(
            self.root / record["cache_path"], image=dataset.np.zeros((4, 5, 6), dtype=dataset.np.float32),
            mask=dataset.np.full((4, 5, 6), 2, dtype=dataset.np.uint8), spacing=dataset.np.ones(3),
        )
        with self.assertRaisesRegex(ValueError, "undeclared"):
            dataset.HipMRIVolumeDataset(self.path, "train")[0]

    def test_two_slices_from_same_volume_decompress_only_once(self):
        slices = dataset.HipMRISliceDataset(self.path, "train")
        with mock.patch.object(slices, "_load_volume", wraps=slices._load_volume) as load:
            first, second = slices[0], slices[1]
        self.assertEqual(load.call_count, 1)
        self.assertEqual((first["slice_index"], second["slice_index"]), (0, 1))
        expected_bytes = 120 * (4 + 8) + 3 * 4  # float32 image, int64 mask, spacing
        self.assertEqual(slices._volume_cache_size_bytes, expected_bytes)

    def test_slice_cache_evicts_old_volume_when_byte_budget_is_exceeded(self):
        one_volume_bytes = 120 * (4 + 8) + 3 * 4
        slices = dataset.HipMRISliceDataset(self.path, "train", volume_cache_bytes=one_volume_bytes)
        with mock.patch.object(slices, "_load_volume", wraps=slices._load_volume) as load:
            slices[0]   # Week0 occupies the budget.
            slices[4]   # Week12 evicts Week0.
            self.assertEqual(list(slices._volume_cache), [1])
            slices[1]   # Week0 must now be loaded again.
        self.assertEqual(load.call_count, 3)
        self.assertEqual(list(slices._volume_cache), [0])
        self.assertEqual(slices._volume_cache_size_bytes, one_volume_bytes)

    def test_volume_larger_than_slice_cache_budget_is_never_retained(self):
        one_volume_bytes = 120 * (4 + 8) + 3 * 4
        slices = dataset.HipMRISliceDataset(self.path, "train", volume_cache_bytes=one_volume_bytes - 1)
        with mock.patch.object(slices, "_load_volume", wraps=slices._load_volume) as load:
            slices[0]
            slices[1]
        self.assertEqual(load.call_count, 2)
        self.assertEqual(len(slices._volume_cache), 0)
        self.assertEqual(slices._volume_cache_size_bytes, 0)

    def test_augmented_slice_does_not_pollute_cached_volume(self):
        slices = dataset.HipMRISliceDataset(self.path, "train", augment=True)
        original = dataset.HipMRIVolumeDataset(self.path, "train")[0]["mask"][0]
        with mock.patch.object(slices, "_load_volume", wraps=slices._load_volume) as load:
            with mock.patch.object(dataset.torch, "rand", side_effect=[dataset.torch.tensor(0.9), dataset.torch.tensor(0.1)]):
                flipped = slices[0]
            self.assertTrue(dataset.torch.equal(flipped["mask"], dataset.torch.flip(original, [-1])))
            self.assertFalse(dataset.torch.equal(flipped["mask"], original))
            self.assertTrue(dataset.torch.equal(slices._volume_cache[0]["mask"][0], original))
            flipped["mask"].fill_(99)  # Returned tensors must not alias cache storage.
            with mock.patch.object(dataset.torch, "rand", side_effect=[dataset.torch.tensor(0.9), dataset.torch.tensor(0.9)]):
                unchanged = slices[0]
        self.assertEqual(load.call_count, 1)
        self.assertTrue(dataset.torch.equal(unchanged["mask"], original))


@unittest.skipIf(prepare.nib is None or prepare.np is None or prepare.zoom is None,
                 "nibabel, NumPy and SciPy required for resampling tests")
class PrepareCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        for folder in ("semantic_MRs", "semantic_labels_only"):
            (self.data_root / folder).mkdir(parents=True)
        self.affine = prepare.np.diag([-1.0, -2.0, 3.0, 1.0])
        image = prepare.np.arange(192, dtype=prepare.np.float32).reshape(8, 6, 4) + 1
        image[0] = 0
        self.image = image
        self.mask = prepare.np.zeros((8, 6, 4), dtype=prepare.np.int16)
        self.mask[6, 4, 2] = 7  # Rare raw class; never interpolate new classes.
        self.audit_records = []
        for patient in ("B006", "B037", "B100"):
            image_path, mask_path = self.write_pair(patient, image, self.mask)
            self.audit_records.append({
                "patient": patient, "visit": "Week0", "pairing_valid": True, "geometry_valid": True,
                "image": {"relative_path": image_path.relative_to(self.data_root).as_posix()},
                "mask": {"relative_path": mask_path.relative_to(self.data_root).as_posix()},
            })
        self.audit_path = self.root / "audit.json"
        self.audit_path.write_text(json.dumps({
            "status": "audit_complete", "blockers": [], "records": self.audit_records,
            "data_root": str(self.data_root),
            "patient_splits": {"train": ["B006"], "validation": ["B037"], "test": ["B100"]},
        }))

    def write_pair(self, patient, image, mask, mask_affine=None):
        image_path = self.data_root / "semantic_MRs" / f"{patient}_Week0_LFOV.nii.gz"
        mask_path = self.data_root / "semantic_labels_only" / f"{patient}_Week0_SEMANTIC.nii.gz"
        prepare.nib.save(prepare.nib.Nifti1Image(image, self.affine), image_path)
        prepare.nib.save(prepare.nib.Nifti1Image(mask, self.affine if mask_affine is None else mask_affine), mask_path)
        return image_path, mask_path

    def test_canonical_resampling_maps_sparse_raw_labels_without_new_classes(self):
        image_path, mask_path = self.write_pair("B006", self.image, self.mask)
        arrays, metadata = prepare.preprocess_pair(image_path, mask_path, (2, 3, 4), [0, 7])
        self.assertEqual(arrays["image"].shape, (2, 3, 4))
        self.assertEqual(arrays["mask"].shape, (2, 3, 4))
        self.assertEqual(arrays["image"].dtype, prepare.np.float32)
        self.assertEqual(arrays["mask"].dtype, prepare.np.uint8)
        self.assertTrue(set(prepare.np.unique(arrays["mask"])).issubset({0, 1}))
        self.assertTrue(prepare.np.allclose(arrays["spacing"], [6.0, 4.0, 2.0]))
        self.assertEqual(prepare.nib.aff2axcodes(prepare.np.asarray(metadata["resampled_affine_xyz"])), ("R", "A", "S"))
        self.assertFalse(metadata["native_resolution_accuracy_claimed"])

    def test_identity_shape_keeps_rare_label_and_zero_background_after_canonicalisation(self):
        image_path, mask_path = self.write_pair("B006", self.image, self.mask)
        arrays, metadata = prepare.preprocess_pair(image_path, mask_path, (4, 6, 8), [0, 7])
        canonical_image = prepare.nib.as_closest_canonical(prepare.nib.load(image_path)).get_fdata()
        self.assertEqual(int(prepare.np.count_nonzero(arrays["mask"] == 1)), 1)
        self.assertTrue(prepare.np.all(arrays["image"].transpose(2, 1, 0)[canonical_image == 0] == 0))
        self.assertGreaterEqual(float(arrays["image"].min()), -5.0)
        self.assertLessEqual(float(arrays["image"].max()), 5.0)
        expected = prepare.nib.as_closest_canonical(prepare.nib.load(image_path)).affine
        self.assertTrue(prepare.np.allclose(metadata["resampled_affine_xyz"], expected))

    def test_fractional_undeclared_and_misaligned_masks_are_rejected(self):
        fractional = self.mask.astype(prepare.np.float32)
        fractional[0, 0, 0] = 0.5
        image_path, mask_path = self.write_pair("B006", self.image, fractional)
        with self.assertRaisesRegex(ValueError, "fractional"):
            prepare.preprocess_pair(image_path, mask_path, (2, 3, 4), [0, 7])
        image_path, mask_path = self.write_pair("B006", self.image, self.mask)
        with self.assertRaisesRegex(ValueError, "undeclared"):
            prepare.preprocess_pair(image_path, mask_path, (2, 3, 4), [0, 1])
        shifted = self.affine.copy()
        shifted[0, 3] = 10
        image_path, mask_path = self.write_pair("B006", self.image, self.mask, shifted)
        with self.assertRaisesRegex(ValueError, "not aligned"):
            prepare.preprocess_pair(image_path, mask_path, (2, 3, 4), [0, 7])

    def test_full_manifest_roundtrip_and_limit_only_uses_train(self):
        with contextlib.redirect_stdout(io.StringIO()):
            path = prepare.prepare_cache(self.audit_path, self.root / "full", (2, 3, 4), [0, 7])
            partial_path = prepare.prepare_cache(self.audit_path, self.root / "partial", (2, 3, 4), [0, 7], limit=1)
        full = dataset.load_manifest(path)
        partial = dataset.load_manifest(partial_path)
        self.assertEqual(len(full["records"]), 3)
        self.assertEqual(full["raw_label_to_index"], {"0": 0, "7": 1})
        self.assertTrue(partial["partial"] and partial["development_only"])
        self.assertEqual({record["split"] for record in partial["records"]}, {"train"})
        with self.assertRaises(FileExistsError):
            prepare.prepare_cache(self.audit_path, self.root / "full", (2, 3, 4), [0, 7])

    def test_blocked_audit_is_rejected_before_cache_writes(self):
        source = json.loads(self.audit_path.read_text())
        source["status"] = "blocked"
        self.audit_path.write_text(json.dumps(source))
        output = self.root / "blocked"
        with self.assertRaisesRegex(ValueError, "audit_complete"):
            prepare.prepare_cache(self.audit_path, output, (2, 3, 4), [0, 7])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
