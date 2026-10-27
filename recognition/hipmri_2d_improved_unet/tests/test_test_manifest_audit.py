"""Constructed CPU-only filename/header/byte checks, never real test images."""

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from audit_test_manifest import (
    SPLIT_FOLDERS, audit_test_headers_and_hashes, collect_checked_splits,
    pair_metadata, sha256_file, verify_test_content,
)
from train import manifest_hash


TEMP_ROOT = PROJECT.parents[1] / "outputs/test_tmp"


def save_pair(root, split, case, week=0, slice_id=1, mask_affine=None):
    image_folder, mask_folder = SPLIT_FOLDERS[split]
    for folder in (image_folder, mask_folder):
        (root / folder).mkdir(parents=True, exist_ok=True)
    suffix = f"{case:03}_week_{week}_slice_{slice_id}.nii.gz"
    image = root / image_folder / f"case_{suffix}"
    mask = root / mask_folder / f"seg_{suffix}"
    nib.save(nib.Nifti1Image(np.arange(256, dtype=np.float32).reshape(16, 16), np.eye(4)), image)
    nib.save(nib.Nifti1Image(np.zeros((16, 16), dtype=np.uint8), np.eye(4) if mask_affine is None else mask_affine), mask)
    return image, mask


def prior_for(pairs):
    return {"data": {split: {"pairs": 1, "manifest_sha256": manifest_hash([pair])}
                     for split, pair in pairs.items() if split != "test"}}


def dataset(root, test_case=36):
    pairs = {split: save_pair(root, split, case)
             for split, case in (("training", 4), ("validation", 35), ("test", test_case))}
    return pairs, prior_for(pairs)


class TestManifestAuditChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    def temporary(self):
        return tempfile.TemporaryDirectory(dir=TEMP_ROOT)

    def test_numeric_tuple_aliases_are_rejected(self):
        pairs = [(Path("case_004_week_0_slice_1.nii.gz"), Path("seg_004_week_0_slice_1.nii.gz")),
                 (Path("case_4_week_00_slice_01.nii.gz"), Path("seg_4_week_00_slice_01.nii.gz"))]
        with self.assertRaisesRegex(ValueError, "Duplicate numeric"):
            pair_metadata(pairs)

    def test_malformed_or_mismatched_names_are_rejected(self):
        for image, mask in (("case_4_week_0_slice_bad.nii.gz", "seg_4_week_0_slice_bad.nii.gz"),
                            ("case_4_week_0_slice_1.nii.gz", "seg_4_week_1_slice_1.nii.gz")):
            with self.subTest(image=image, mask=mask), self.assertRaisesRegex(ValueError, "basename"):
                pair_metadata([(Path(image), Path(mask))])

    def test_same_case_different_week_or_slice_is_still_leakage(self):
        with self.temporary() as folder:
            root = Path(folder)
            pairs, prior = dataset(root)
            image, mask = pairs["test"]
            image.unlink(); mask.unlink()
            save_pair(root, "test", 4, week=7, slice_id=9)
            with self.assertRaisesRegex(ValueError, "case-group leakage"):
                collect_checked_splits(root, prior)

    def test_missing_and_extra_mask_fail_shared_pair_check(self):
        for missing in (True, False):
            with self.subTest(missing=missing), self.temporary() as folder:
                root = Path(folder)
                pairs, prior = dataset(root)
                if missing:
                    pairs["test"][1].unlink()
                else:
                    extra = root / SPLIT_FOLDERS["test"][1] / "seg_099_week_0_slice_1.nii.gz"
                    extra.write_bytes(b"not opened as NIfTI")
                expected = FileNotFoundError if missing else ValueError
                with self.assertRaises(expected):
                    collect_checked_splits(root, prior)

    def test_prior_manifest_drift_is_rejected_without_opening_headers(self):
        with self.temporary() as folder:
            root = Path(folder)
            _, prior = dataset(root)
            prior["data"]["training"]["manifest_sha256"] = "0" * 64
            with patch("audit_test_manifest.nib.load", side_effect=AssertionError("Must not read headers")):
                with self.assertRaisesRegex(ValueError, "training filename manifest"):
                    collect_checked_splits(root, prior)

    def test_header_audit_and_hashes_never_materialize_voxel_arrays(self):
        with self.temporary() as folder:
            root = Path(folder)
            pairs, prior = dataset(root)
            all_pairs, summaries, overlaps = collect_checked_splits(root, prior)
            self.assertEqual(overlaps, {"training__validation": [], "training__test": [], "validation__test": []})
            self.assertEqual(summaries["test"]["case_proxy_ids"], [36])
            with patch.object(nib.Nifti1Image, "get_fdata", side_effect=AssertionError("No voxels")), \
                    patch.object(nib.arrayproxy.ArrayProxy, "__array__", side_effect=AssertionError("No proxy voxels")):
                geometry, files, digest = audit_test_headers_and_hashes(all_pairs["test"], 1)
            self.assertEqual(geometry["error_count"], 0)
            self.assertEqual(geometry["passed_count"], 1)
            self.assertEqual(geometry["shape_counts"], {"16x16": 1})
            self.assertEqual(geometry["max_affine_abs_difference"], 0)
            self.assertEqual(geometry["identity_affine_pairs"], 1)
            row = files[0]
            self.assertEqual(row["image_sha256"], hashlib.sha256(pairs["test"][0].read_bytes()).hexdigest())
            text = f"{row['image']}|{row['mask']}|{row['image_sha256']}|{row['mask_sha256']}"
            self.assertEqual(digest, hashlib.sha256(text.encode()).hexdigest())

    def test_bad_geometry_is_recorded_without_reading_pixels(self):
        with self.temporary() as folder:
            root = Path(folder)
            affine = np.eye(4); affine[0, 3] = 1
            pair = save_pair(root, "test", 36, mask_affine=affine)
            with patch.object(nib.Nifti1Image, "get_fdata", side_effect=AssertionError("No voxels")):
                geometry, files, digest = audit_test_headers_and_hashes([pair], 1)
            self.assertEqual(geometry["error_count"], 1)
            self.assertIn("affine mismatch", geometry["errors"][0]["message"])
            self.assertIsNone(digest)
            self.assertEqual(files[0]["status"], "failed")

    def test_compressed_recheck_uses_no_header_load_and_detects_byte_changes(self):
        with self.temporary() as folder:
            root = Path(folder)
            _, prior = dataset(root)
            pairs, summaries, _ = collect_checked_splits(root, prior)
            geometry, files, digest = audit_test_headers_and_hashes(pairs["test"], 1)
            audit = {"status": "passed_full", "splits": summaries, "test_files": files,
                     "test_content_manifest_sha256": digest}
            with patch("audit_test_manifest.nib.load", side_effect=AssertionError("No header reload")):
                actual = verify_test_content(pairs["test"], audit)
                self.assertEqual(actual["compressed_files_verified"], 2)
                self.assertFalse(actual["voxel_arrays_materialized"])
                path = pairs["test"][0][1]
                with path.open("ab") as handle:
                    handle.write(b"changed-compressed-stream")
                with self.assertRaisesRegex(ValueError, "compressed file bytes changed"):
                    verify_test_content(pairs["test"], audit)

    def test_content_recheck_rejects_incomplete_audit_order_and_digest_changes(self):
        with self.temporary() as folder:
            root = Path(folder)
            _, prior = dataset(root)
            pairs, summaries, _ = collect_checked_splits(root, prior)
            _, files, digest = audit_test_headers_and_hashes(pairs["test"], 1)
            audit = {"status": "passed_full", "splits": summaries, "test_files": files,
                     "test_content_manifest_sha256": digest}
            for key, value in (("status", "failed"), ("test_files", []), ("test_content_manifest_sha256", "0" * 64)):
                changed = deepcopy(audit); changed[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    verify_test_content(pairs["test"], changed)

    def test_test_folder_cannot_leave_data_root(self):
        with self.temporary() as folder:
            for folders in (("../outside", "masks"), ("images", ".."), (str(Path(folder).resolve()), "masks")):
                with self.subTest(folders=folders), self.assertRaisesRegex(ValueError, "basenames"):
                    collect_checked_splits(Path(folder), {}, folders)


if __name__ == "__main__":
    unittest.main()
