"""Check exact decoded identities and frozen-manifest cache reconstruction."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("NumPy is required for decoded cache identity checks.")

TOOL = Path(__file__).resolve().parents[1] / "tools" / "cache_identity.py"
SPEC = importlib.util.spec_from_file_location("cache_identity", TOOL)
identity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(identity)


class CacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original, self.data = self.root / "original", self.root / "data"
        (self.original / "volumes").mkdir(parents=True)
        self.data.mkdir()
        self.arrays = {"image": np.arange(64, dtype=np.float32).reshape(4, 4, 4),
                       "mask": (np.arange(64).reshape(4, 4, 4) % 2).astype(np.uint8),
                       "spacing": np.array([2, 1, 1], dtype=np.float32)}
        self.geometry = {"target_shape_dhw": [4, 4, 4], "target_shape_xyz": [4, 4, 4],
                         "spacing_dhw_mm": [2.0, 1.0, 1.0],
                         "resampled_affine_xyz": np.diag([1, 1, 2, 1]).tolist(),
                         "class_voxel_counts": [32, 32]}
        records = []
        for patient, split in (("A", "train"), ("B", "validation"), ("C", "test")):
            np.save(self.data / f"{patient}_image.npy", self.arrays["image"])
            np.save(self.data / f"{patient}_mask.npy", self.arrays["mask"])
            np.savez_compressed(self.original / "volumes" / f"{patient}.npz", **self.arrays)
            records.append({"patient": patient, "visit": "Week0", "split": split,
                            "cache_path": f"volumes/{patient}.npz", "source_image": f"{patient}_image.npy",
                            "source_mask": f"{patient}_mask.npy", **copy.deepcopy(self.geometry)})
        self.manifest = {"num_classes": 2, "label_ids": [0, 1], "raw_label_to_index": {"0": 0, "1": 1},
                         "target_shape_dhw": [4, 4, 4], "audit_sha256": "a" * 64,
                         "patient_splits": {"train": ["A"], "validation": ["B"], "test": ["C"]},
                         "records": records, "partial": False, "development_only": False}
        self.manifest_path = self.original / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.book_path = self.root / "identity.json"
        self.book = identity.record_identity(self.manifest_path, self.book_path)
        self.rebuilt = self.root / "rebuilt"
        self.rebuilt.mkdir()
        self.frozen = self.rebuilt / "manifest.json"
        shutil.copyfile(self.manifest_path, self.frozen)

    def fake_preprocess(self, image_path, mask_path, shape, labels):
        arrays = {"image": np.load(image_path, allow_pickle=False),
                  "mask": np.load(mask_path, allow_pickle=False), "spacing": self.arrays["spacing"].copy()}
        metadata = copy.deepcopy(self.geometry)
        metadata["class_voxel_counts"] = np.bincount(arrays["mask"].ravel(), minlength=len(labels)).tolist()
        return arrays, metadata

    def rebuild(self, **kwargs):
        with mock.patch.object(identity, "get_preprocessor", return_value=self.fake_preprocess):
            return identity.rebuild_caches(self.frozen, self.book_path, self.data, **kwargs)

    def save_book(self):
        self.book_path.write_text(json.dumps(self.book), encoding="utf-8")

    def test_record_exports_only_descriptors_and_array_hash_includes_dtype_shape(self):
        entry = self.book["records"]["volumes/A.npz"]["arrays"]
        self.assertEqual(set(entry), {"image", "mask", "spacing"})
        self.assertEqual(set(entry["image"]), {"dtype", "shape", "sha256"})
        self.assertEqual(entry["image"], identity.array_identity(np.asfortranarray(self.arrays["image"])))
        self.assertNotEqual(entry["image"]["sha256"], identity.array_identity(self.arrays["image"].astype(np.float64))["sha256"])
        self.assertNotEqual(entry["image"]["sha256"], identity.array_identity(self.arrays["image"].reshape(8, 8))["sha256"])

    def test_roundtrip_default_test_only_keeps_manifest_bytes_and_reuses_matching_cache(self):
        original_manifest = self.frozen.read_bytes()
        result = self.rebuild()
        self.assertEqual(result["written"], ["volumes/C.npz"])
        self.assertFalse((self.rebuilt / "volumes/A.npz").exists())
        self.assertEqual(self.frozen.read_bytes(), original_manifest)
        destination = self.rebuilt / "volumes/C.npz"
        expected = self.book["records"]["volumes/C.npz"]["arrays"]
        self.assertEqual(identity.arrays_identity(identity.read_arrays(destination)), expected)
        compressed_digest = identity.file_sha256(destination)
        # An alternate ZIP container with identical decoded arrays is compatible.
        np.savez(destination, **self.arrays)
        self.assertNotEqual(identity.file_sha256(destination), compressed_digest)
        with mock.patch.object(identity, "get_preprocessor", side_effect=AssertionError("Must reuse a verified cache")):
            reused = identity.rebuild_caches(self.frozen, self.book_path, self.data)
        self.assertEqual(reused["reused"], ["volumes/C.npz"])
        self.assertEqual(self.frozen.read_bytes(), original_manifest)

    def test_corrupted_source_never_writes_a_cache_or_changes_manifest(self):
        changed = self.arrays["image"].copy()
        changed[0, 0, 0] += 1
        np.save(self.data / "C_image.npy", changed)
        original_manifest = self.frozen.read_bytes()
        with self.assertRaisesRegex(ValueError, "Rebuilt array digest mismatch"):
            self.rebuild()
        self.assertFalse((self.rebuilt / "volumes").exists())
        self.assertEqual(self.frozen.read_bytes(), original_manifest)

    def test_corrupted_existing_cache_is_not_overwritten(self):
        destination = self.rebuilt / "volumes/C.npz"
        destination.parent.mkdir()
        changed = dict(self.arrays)
        changed["image"] = self.arrays["image"] + 1
        np.savez_compressed(destination, **changed)
        original = destination.read_bytes()
        with self.assertRaisesRegex(ValueError, "Existing cache digest mismatch"):
            self.rebuild()
        self.assertEqual(destination.read_bytes(), original)

    def test_manifest_and_preprocessor_hash_mismatch_fail_before_writes(self):
        for field in ("manifest_sha256", "preprocessor_sha256"):
            with self.subTest(field=field):
                original = self.book[field]
                self.book[field] = "0" * 64
                self.save_book()
                with self.assertRaisesRegex(ValueError, "digest does not match"):
                    self.rebuild()
                self.assertFalse((self.rebuilt / "volumes").exists())
                self.book[field] = original

    def test_array_digest_mismatch_is_rejected_before_write(self):
        self.book["records"]["volumes/C.npz"]["arrays"]["image"]["sha256"] = "0" * 64
        self.save_book()
        with self.assertRaisesRegex(ValueError, "Rebuilt array digest mismatch"):
            self.rebuild()
        self.assertFalse((self.rebuilt / "volumes").exists())

    def test_source_traversal_is_rejected_even_with_matching_manifest_digest(self):
        changed = copy.deepcopy(self.manifest)
        changed["records"][-1]["source_image"] = "../outside.npy"
        self.frozen.write_text(json.dumps(changed), encoding="utf-8")
        self.book["manifest_sha256"] = identity.file_sha256(self.frozen)
        self.save_book()
        with self.assertRaisesRegex(ValueError, "Unsafe relative path"):
            self.rebuild()
        self.assertFalse((self.rebuilt / "volumes").exists())

    def test_cache_path_traversal_is_rejected_before_writes(self):
        changed = copy.deepcopy(self.manifest)
        changed["records"][-1]["cache_path"] = "../outside.npz"
        self.frozen.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.rebuild()
        self.assertFalse((self.rebuilt / "volumes").exists())

    def test_rebuilt_geometry_mismatch_is_rejected(self):
        def changed_preprocess(*args):
            arrays, metadata = self.fake_preprocess(*args)
            metadata["resampled_affine_xyz"][0][3] = 10.0
            return arrays, metadata
        with mock.patch.object(identity, "get_preprocessor", return_value=changed_preprocess):
            with self.assertRaisesRegex(ValueError, "Rebuilt metadata differs.*resampled_affine"):
                identity.rebuild_caches(self.frozen, self.book_path, self.data)
        self.assertFalse((self.rebuilt / "volumes").exists())

    def test_train_limit_rebuilds_one_patient_cache_only(self):
        result = self.rebuild(split="train", limit=1)
        self.assertEqual(result["selected_records"], 1)
        self.assertEqual(result["written"], ["volumes/A.npz"])
        self.assertFalse((self.rebuilt / "volumes/C.npz").exists())


if __name__ == "__main__":
    unittest.main()
