"""Small synthetic unit fixtures test contracts, not medical/model quality."""

import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tools"))
from dataset import (
    FIELDS,
    ADNISliceDataset,
    defective_scans,
    foreground_share,
    read_manifest,
)
from prepare_data import assign_patients


class DataContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.csv"
        self.rows = []
        for i, split in enumerate(("train", "validation", "test")):
            scan = str(100 + i)
            relative = f"AD_NC/train/AD/{scan}_1.jpeg"
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (8, 4), 64 * (i + 1)).save(path)
            self.rows.append(
                dict(
                    zip(
                        FIELDS,
                        (
                            relative,
                            "train",
                            "AD",
                            "2",
                            f"001_S_{i:04d}",
                            scan,
                            "1",
                            split,
                            hashlib.sha256(path.read_bytes()).hexdigest(),
                        ),
                    )
                )
            )
        self.write_manifest()

    def write_manifest(self):
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def test_padding_and_normalization_preserve_geometry(self):
        values = ADNISliceDataset(self.root, self.manifest, "train", 8)[0][
            "pixel_values"
        ]
        self.assertEqual(tuple(values.shape), (3, 8, 8))
        self.assertTrue(torch.equal(values[0], values[2]))
        self.assertTrue(torch.all(values[:, :2] == -1))
        self.assertTrue(torch.all(values[:, 6:] == -1))
        self.assertTrue(
            torch.allclose(values[:, 2:6], torch.full((3, 4, 8), 64 / 127.5 - 1))
        )

    def test_cross_patient_overlap_rejected_before_split_filter(self):
        self.rows[1]["patient_id"] = self.rows[0]["patient_id"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "patient_id"):
            ADNISliceDataset(self.root, self.manifest, "train")

    def test_identical_content_crossing_splits_is_rejected(self):
        self.rows[1]["sha256"] = self.rows[0]["sha256"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "sha256"):
            read_manifest(self.manifest)

    def test_directory_escape_is_rejected(self):
        self.rows[0]["relative_path"] = "../outside.jpeg"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            read_manifest(self.manifest)

    def test_mislabeled_image_path_is_rejected(self):
        self.rows[0]["relative_path"] = "AD_NC/train/NC/100_1.jpeg"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Path and recorded identity"):
            read_manifest(self.manifest)

    def test_changed_image_is_rejected_when_hashes_are_verified(self):
        train_image = self.root / self.rows[0]["relative_path"]
        Image.new("L", (8, 4), 200).save(train_image)
        ADNISliceDataset(self.root, self.manifest, "train", 8)[0]
        with self.assertRaisesRegex(ValueError, "hash changed"):
            ADNISliceDataset(self.root, self.manifest, "train", 8, verify_hash=True)[0]

    def test_whole_scan_is_flagged_when_any_slice_is_near_empty(self):
        images = -torch.ones(3, 3, 20, 20)
        images[0, :, :10] = 1.0  # half bright
        images[2, :, :, :1] = 1.0  # 5% bright, at the threshold, so not empty
        shares = foreground_share(images).tolist()
        self.assertTrue(
            torch.allclose(torch.tensor(shares), torch.tensor([0.5, 0.0, 0.05]))
        )
        rows = [{"scan_id": "100"}, {"scan_id": "100"}, {"scan_id": "200"}]
        self.assertEqual(defective_scans(rows, shares), {"100"})

    def test_invalid_resolution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "multiple of eight"):
            ADNISliceDataset(self.root, self.manifest, "train", 255)

    def test_stratified_patient_counts_and_order_independence(self):
        patients = {
            f"patient-{label}-{i}": label for label in ("AD", "NC") for i in range(20)
        }
        assigned = assign_patients(patients, 3710)
        self.assertEqual(
            assigned, assign_patients(dict(reversed(list(patients.items()))), 3710)
        )
        for label in ("AD", "NC"):
            counts = {
                split: sum(
                    patients[p] == label and s == split for p, s in assigned.items()
                )
                for split in ("train", "validation", "test")
            }
            self.assertEqual(counts, {"train": 14, "validation": 3, "test": 3})


if __name__ == "__main__":
    unittest.main()
