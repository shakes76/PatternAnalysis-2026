"""Meaningful checks for split leakage, raw-label mapping and NIfTI alignment."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import (HipMRIDataset, SyntheticSegmentationDataset, map_labels,
                     read_manifest, validate_patient_splits)


class ManifestTests(unittest.TestCase):
    def record(self, patient, split, image, mask):
        return {"patient_id": patient, "split": split, "image": image, "mask": mask}

    def test_patient_leakage_is_rejected(self):
        rows = [self.record("p1", "train", "i1.nii", "m1.nii"),
                self.record("p1", "val", "i2.nii", "m2.nii")]
        with self.assertRaisesRegex(ValueError, "Patient leakage"):
            validate_patient_splits(rows)

    def test_file_reuse_across_splits_is_rejected(self):
        for duplicate_column in ("image", "mask"):
            rows = [self.record("p1", "train", "i1.nii", "m1.nii"),
                    self.record("p2", "test", "i2.nii", "m2.nii")]
            rows[1][duplicate_column] = rows[0][duplicate_column]
            with self.subTest(column=duplicate_column):
                with self.assertRaisesRegex(ValueError, "File leakage"):
                    validate_patient_splits(rows)

    def test_explicit_valid_split_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            manifest.write_text("image,mask,patient_id,split\ni.nii,m.nii,p1,train\n", encoding="utf-8")
            rows = read_manifest(manifest)
            self.assertEqual(rows[0]["image"], str((Path(directory) / "i.nii").resolve()))
            self.assertEqual(rows[0]["patient_id"], "p1")


class LabelMappingTests(unittest.TestCase):
    def test_mapping_stays_global_when_intermediate_class_absent(self):
        mask = np.array([[0, 7], [7, 0]], dtype=np.float64)
        np.testing.assert_array_equal(map_labels(mask, (0, 3, 7)), [[0, 2], [2, 0]])

    def test_unknown_and_fractional_labels_rejected(self):
        for mask, message in ((np.array([[0., 8.]]), "unknown"),
                              (np.array([[0., 1.5]]), "non-integer"),
                              (np.array([[0., np.nan]]), "non-finite")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    map_labels(mask, (0, 1))

    def test_background_must_be_zero(self):
        with self.assertRaisesRegex(ValueError, "background 0"):
            map_labels(np.array([[0, 1]]), (1, 0))

    def test_synthetic_shapes_determinism_and_explicit_origin(self):
        dataset = SyntheticSegmentationDataset(samples=3, image_size=32, seed=7)
        a, b = dataset[1], dataset[1]
        self.assertEqual(a["image"].shape, (1, 32, 32))
        self.assertEqual(a["mask"].dtype, torch.long)
        self.assertTrue(torch.equal(a["image"], b["image"]))
        self.assertEqual(set(torch.unique(a["mask"]).tolist()), {0, 1})
        self.assertTrue(a["image_path"].startswith("synthetic://"))

    def test_adjacent_seeds_do_not_reuse_shifted_samples(self):
        first = SyntheticSegmentationDataset(samples=2, image_size=32, seed=7)[1]
        second = SyntheticSegmentationDataset(samples=2, image_size=32, seed=8)[0]
        self.assertFalse(torch.equal(first["image"], second["image"]))


@unittest.skipUnless(importlib.util.find_spec("nibabel"), "nibabel unavailable: real NIfTI checks skipped")
class NiftiTests(unittest.TestCase):
    def setUp(self):
        import nibabel as nib
        self.nib = nib
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "manifest.csv"
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "mask", "patient_id", "split"])
            writer.writerow(["image.nii.gz", "mask.nii.gz", "p1", "train"])

    def tearDown(self):
        self.tmp.cleanup()

    def write_pair(self, image, mask, mask_affine=None):
        self.nib.save(self.nib.Nifti1Image(image.astype(np.float32), np.eye(4)), self.root / "image.nii.gz")
        affine = np.eye(4) if mask_affine is None else mask_affine
        self.nib.save(self.nib.Nifti1Image(mask.astype(np.float64), affine), self.root / "mask.nii.gz")

    def test_valid_singleton_2d_and_global_mapping(self):
        image = np.arange(64, dtype=np.float32).reshape(8, 8, 1)
        mask = np.zeros((8, 8, 1))
        mask[2:6, 2:6] = 7
        self.write_pair(image, mask)
        sample = HipMRIDataset(self.manifest, "train", image_size=16, label_values=(0, 3, 7))[0]
        self.assertEqual(sample["image"].shape, (1, 16, 16))
        self.assertEqual(sample["mask"].shape, (16, 16))
        self.assertEqual(set(torch.unique(sample["mask"]).tolist()), {0, 2})
        self.assertTrue(torch.isfinite(sample["image"]).all())

    def test_affine_mismatch_rejected(self):
        affine = np.eye(4)
        affine[0, 3] = 10
        self.write_pair(np.ones((8, 8)), np.zeros((8, 8)), affine)
        with self.assertRaisesRegex(ValueError, "affines disagree"):
            HipMRIDataset(self.manifest, "train")[0]

    def test_three_dimensional_volume_rejected(self):
        self.write_pair(np.ones((8, 8, 2)), np.zeros((8, 8, 2)))
        with self.assertRaisesRegex(ValueError, "must be 2D"):
            HipMRIDataset(self.manifest, "train")[0]

    def test_shape_mismatch_rejected(self):
        self.write_pair(np.ones((8, 8)), np.zeros((8, 9)))
        with self.assertRaisesRegex(ValueError, "shapes disagree"):
            HipMRIDataset(self.manifest, "train")[0]

    def test_image_nan_rejected(self):
        image = np.ones((8, 8))
        image[0, 0] = np.nan
        self.write_pair(image, np.zeros((8, 8)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            HipMRIDataset(self.manifest, "train")[0]

    def test_unknown_raw_label_rejected_at_load(self):
        self.write_pair(np.ones((8, 8)), np.full((8, 8), 3))
        with self.assertRaisesRegex(ValueError, "unknown raw labels"):
            HipMRIDataset(self.manifest, "train", label_values=(0, 1))[0]

    def test_audit_cli_produces_aggregate_metadata_and_overlay(self):
        mask = np.zeros((8, 8))
        mask[2:6, 2:6] = 3
        self.write_pair(np.arange(64).reshape(8, 8), mask)
        output, overlay = self.root / "audit.json", self.root / "sample.png"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] / "audit_data.py"),
             "--manifest", str(self.manifest), "--label-values", "0", "3",
             "--prostate-label", "3", "--output", str(output), "--sample-output", str(overlay)],
            text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["raw_label_voxels"], {"0": 48, "3": 16})
        self.assertEqual(report["prostate_class_index"], 1)
        self.assertEqual(report["splits"]["train"], {"slices": 1, "patients": 1})
        self.assertTrue(report["warnings"])  # Intentionally missing val/test fixtures.
        self.assertNotIn('"p1"', output.read_text(encoding="utf-8"))
        self.assertNotIn(str(self.root), completed.stdout)
        self.assertGreater(overlay.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
