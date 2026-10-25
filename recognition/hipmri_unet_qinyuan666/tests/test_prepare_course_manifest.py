"""Freeze only audited, unchanged and case-disjoint course inputs."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_course_download import audit_course
from prepare_course_manifest import prepare_manifest
from dataset import read_manifest


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "source"
        self.root.mkdir()
        for case, split in enumerate(("train", "validate", "test"), 1):
            for directory_prefix, name_prefix in (("", "case"), ("seg_", "seg")):
                directory = self.root / ("keras_slices_" + directory_prefix + split)
                directory.mkdir()
                nib.save(nib.Nifti1Image(np.ones((8, 4), dtype=np.uint8), np.eye(4)),
                         directory / f"{name_prefix}_{case:03d}_week_0_slice_0.nii.gz")
        summary, private = audit_course(self.root, 1)
        self.summary = self.base / "summary.json"
        self.private = self.base / "private.json"
        self.labels = self.base / "labels.json"
        for path, data in ((self.summary, summary), (self.private, private),
                           (self.labels, {"label_values": [0, 1], "prostate_raw_label": 1})):
            path.write_text(json.dumps(data), encoding="utf-8")
        self.manifest = self.base / "manifests/course.csv"

    def prepare(self):
        return prepare_manifest(self.root, self.summary, self.private, self.labels, self.manifest)

    def test_groups_relative_paths_and_idempotence(self):
        first = self.prepare()
        self.assertEqual(first, self.prepare())
        self.assertEqual(first["source_file_hashes_rechecked"], 6)
        self.assertEqual(first["splits"]["val"], {"slices": 1, "case_groups": 1})
        text = self.manifest.read_text(encoding="utf-8")
        self.assertIn("../source/", text)
        self.assertNotIn(str(self.base), text)
        self.assertTrue(all(Path(row["image"]).is_file() for row in read_manifest(self.manifest)))
        self.assertNotIn("case_001", json.dumps(first))

    def test_changed_source_hash_rejected(self):
        path = next((self.root / "keras_slices_train").glob("*.nii.gz"))
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.prepare()
        self.assertFalse(self.manifest.exists())

    def test_failed_case_isolation_cannot_be_promoted(self):
        data = json.loads(self.summary.read_text(encoding="utf-8"))
        data["case_groups_disjoint_by_filename"] = False
        self.summary.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "case-disjoint"):
            self.prepare()

    def test_different_existing_manifest_is_not_overwritten(self):
        self.manifest.parent.mkdir()
        self.manifest.write_text("keep existing content", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(self.manifest.read_text(encoding="utf-8"), "keep existing content")


if __name__ == "__main__":
    unittest.main()
