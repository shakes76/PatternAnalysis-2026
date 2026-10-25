"""Acquisition failures and filename grouping, with artificial NIfTI fixtures."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_course_download import audit_course


class CourseDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for split in ("train", "validate", "test"):
            for prefix in ("", "seg_"):
                (self.root / ("keras_slices_" + prefix + split)).mkdir()

    def add_pair(self, split, case, week=0):
        paths = []
        for folder_prefix, file_prefix in (("", "case"), ("seg_", "seg")):
            path = self.root / ("keras_slices_" + folder_prefix + split) / f"{file_prefix}_{case:03}_week_{week}_slice_0.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((4, 5), dtype=np.uint8), np.eye(4)), path)
            paths.append(path)
        return paths

    def test_valid_pairs_and_aggregate_privacy(self):
        for case, split in enumerate(("train", "validate", "test"), 1):
            self.add_pair(split, case)
        report, private = audit_course(self.root, 1)
        self.assertTrue(report["local_integrity_passed"])
        self.assertTrue(report["case_groups_disjoint_by_filename"])
        self.assertEqual(report["gzip_and_nifti_files_read"], 6)
        self.assertEqual(report["valid_pair_shapes"], {"4x5": 3})
        self.assertNotIn("case_001", json.dumps(report))
        self.assertIn("case_001", json.dumps(private))
        self.assertFalse(report["training_ready"])

    def test_missing_mask_is_not_a_successful_download(self):
        _, mask = self.add_pair("train", 1)
        mask.unlink()
        report, _ = audit_course(self.root, 1)
        self.assertFalse(report["local_integrity_passed"])
        self.assertEqual(report["inventory_issue_counts"]["unpaired_key"], 1)

    def test_truncated_gzip_trailer_is_detected(self):
        image, _ = self.add_pair("train", 1)
        image.write_bytes(image.read_bytes()[:-8])
        report, _ = audit_course(self.root, 1)
        self.assertFalse(report["local_integrity_passed"])
        self.assertEqual(report["pair_error_counts"]["image_read_error"], 1)

    def test_repeated_case_in_different_weeks_still_overlaps(self):
        self.add_pair("train", 1, week=0)
        self.add_pair("validate", 2, week=0)
        self.add_pair("test", 1, week=1)
        report, _ = audit_course(self.root, 1)
        self.assertTrue(report["local_integrity_passed"])
        self.assertFalse(report["case_groups_disjoint_by_filename"])
        self.assertEqual(report["case_group_overlap_counts"]["train_test"], 1)

    def test_affine_mismatch_is_detected(self):
        _, mask = self.add_pair("train", 1)
        nib.save(nib.Nifti1Image(np.ones((4, 5), dtype=np.uint8), np.diag([2., 1., 1., 1.])), mask)
        report, _ = audit_course(self.root, 1)
        self.assertFalse(report["local_integrity_passed"])
        self.assertEqual(report["pair_error_counts"]["affine_mismatch"], 1)

    def test_empty_expected_directories_are_not_complete(self):
        self.add_pair("train", 1)
        report, _ = audit_course(self.root, 1)
        self.assertFalse(report["local_integrity_passed"])
        self.assertEqual(report["inventory_issue_counts"]["empty_expected_directory"], 4)


if __name__ == "__main__":
    unittest.main()
