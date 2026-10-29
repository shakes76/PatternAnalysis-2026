"""Course-facing dataset API behavior, using synthetic records only."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from dataset import ADNISliceDataset, AugmentedTrainingDataset, split_by_patient


class SubmissionDatasetTests(unittest.TestCase):
    def test_patient_split_keeps_every_slice_of_each_patient_together(self):
        scans, samples = [], []
        for n in range(680):
            patient, image = f"001_S_{n:04}", str(10000 + n)
            label = "AD" if n < 221 else "NC"
            scans.append(dict(patient_id=patient, selected_image_id=image, label=label))
            for i in range(20):
                samples.append(dict(file_path=f"/synthetic/{image}_{i}.jpeg",
                    original_split="train" if i % 2 else "test", image_id=image,
                    patient_id=patient, scan_date="", label=label))
        patients, slices, checks = split_by_patient(scans, samples)
        self.assertTrue(all(checks.values()))
        mapping = {row["patient_id"]: row["split"] for row in patients}
        self.assertEqual(len(mapping), 680)
        self.assertTrue(all(row["split"] == mapping[row["patient_id"]] for row in slices))
        self.assertEqual({s: sum(p["split"] == s for p in patients)
                          for s in ("train", "validation", "test")},
                         {"train": 408, "validation": 136, "test": 136})

    def test_intensity_augmentation_is_optional_and_training_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields = ("file_path", "original_split", "image_id", "patient_id", "scan_date", "label", "split")
            rows = []
            for n, split in enumerate(("train", "validation", "test")):
                path = root / f"{100+n}_0.png"
                Image.new("L", (256, 240), 128).save(path)
                rows.append(dict(zip(fields, (str(path), "train", str(100+n),
                    f"001_S_{n:04}", "", "NC", split))))
            manifest = root / "split_samples.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
            (root / "split_summary.json").write_text(json.dumps({"status": "passed",
                "output_sha256": {"split_samples": hashlib.sha256(manifest.read_bytes()).hexdigest()}}))
            plain = ADNISliceDataset(manifest, "train")[0]["image"]
            torch.manual_seed(42)
            changed = AugmentedTrainingDataset(manifest, "train")[0]["image"]
            torch.manual_seed(42)
            repeated = AugmentedTrainingDataset(manifest, "train")[0]["image"]
            self.assertFalse(torch.equal(plain, changed))
            self.assertTrue(torch.equal(changed, repeated))
            self.assertGreaterEqual(changed.min().item(), 0)
            self.assertLessEqual(changed.max().item(), 1)
            for split in ("validation", "test"):
                with self.subTest(split=split):
                    with self.assertRaisesRegex(ValueError, "training only"):
                        AugmentedTrainingDataset(manifest, split)


if __name__ == "__main__":
    unittest.main()
