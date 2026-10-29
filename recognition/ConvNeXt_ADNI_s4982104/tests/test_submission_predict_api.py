"""Saved-checkpoint test inference for the course-facing predict.py entry point."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from modules import SimpleCNN, build_model
from predict import run_test_prediction


class SubmissionPredictTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fields = ("file_path", "original_split", "image_id", "patient_id", "scan_date", "label", "split")
        rows = []
        for n, (split, label) in enumerate((("train", "NC"), ("validation", "AD"),
                                            ("test", "NC"), ("test", "AD"))):
            image_id, patient = str(100 + n), f"001_S_{n:04}"
            for slice_no in range(2):
                path = self.root / f"{image_id}_{slice_no}.png"
                if split == "test":
                    Image.new("L", (256, 240), 80 + n * 30).save(path)
                rows.append(dict(zip(fields, (str(path), "train", image_id,
                    patient, "", label, split))))
        self.manifest = self.root / "split_samples.csv"
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        self.split_summary = self.root / "split_summary.json"
        self.split_summary.write_text(json.dumps({"status": "passed", "output_sha256": {
            "split_samples": hashlib.sha256(self.manifest.read_bytes()).hexdigest()}}))
        self.checkpoint = self.root / "model.pt"
        torch.save({"model_state_dict": SimpleCNN().state_dict(), "config": {"model": "SimpleCNN"},
                    "epoch": 1}, self.checkpoint)

    def tearDown(self):
        self.temp.cleanup()

    def test_saved_cnn_predicts_only_test_images_and_writes_outputs(self):
        result = run_test_prediction(self.manifest, self.split_summary, self.checkpoint,
                                     self.root / "output", "simple_cnn", device="cpu", expected_slices=2)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["split"], "test")
        self.assertEqual(result["image_reads"], 4)
        self.assertEqual(result["patient_count"], 2)
        self.assertTrue((self.root / "output" / "confusion_matrix.png").is_file())
        with (self.root / "output" / "patient_predictions.csv").open() as handle:
            patients = list(csv.DictReader(handle))
        self.assertEqual(len(patients), 2)
        self.assertEqual({r["patient_id"] for r in patients}, {"001_S_0002", "001_S_0003"})
        self.assertEqual({r["slice_count"] for r in patients}, {"2"})

    def test_wrong_model_type_and_incomplete_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "model"):
            run_test_prediction(self.manifest, self.split_summary, self.checkpoint,
                                self.root / "wrong", "convnext_tiny", device="cpu", expected_slices=2)
        broken = self.root / "broken.pt"
        torch.save({"model_state_dict": {}, "config": {"model": "SimpleCNN"}}, broken)
        with self.assertRaises(RuntimeError):
            run_test_prediction(self.manifest, self.split_summary, broken,
                                self.root / "broken-output", "simple_cnn", device="cpu", expected_slices=2)

    def test_saved_convnext_predicts_a_test_case(self):
        with self.manifest.open(newline="") as handle:
            rows = [r for r in csv.DictReader(handle)
                    if r["split"] == "test" and r["patient_id"] == "001_S_0002"]
        fields = list(rows[0])
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        self.split_summary.write_text(json.dumps({"status": "passed", "output_sha256": {
            "split_samples": hashlib.sha256(self.manifest.read_bytes()).hexdigest()}}))
        checkpoint = self.root / "convnext.pt"
        torch.save({"model_state_dict": build_model("convnext_tiny").state_dict(),
                    "config": {"model": "convnext_tiny"}, "epoch": 1}, checkpoint)
        result = run_test_prediction(self.manifest, self.split_summary, checkpoint,
                                     self.root / "convnext-output", "convnext_tiny",
                                     device="cpu", expected_slices=2)
        self.assertEqual(result["model"], "convnext_tiny")
        self.assertEqual(result["image_reads"], 2)
        self.assertEqual(result["patient_count"], 1)


if __name__ == "__main__":
    unittest.main()
