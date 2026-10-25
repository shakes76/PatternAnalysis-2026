"""Verify aggregation, provenance and isolation of explicitly selected splits."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_cases import evaluate, summarize_cases
from modules import build_model


class CaseEvaluationTests(unittest.TestCase):
    def test_case_mean_differs_from_pixel_pooling(self):
        matrices = {"a": torch.tensor([[0, 0], [0, 100]]),
                    "b": torch.tensor([[0, 0], [1, 0]])}
        result = summarize_cases(matrices, {"a": 10, "b": 1}, 1)
        self.assertEqual(result["prostate_case_dice"]["mean"], .5)
        self.assertAlmostEqual(result["pixel_pooled"]["per_class"][1]["dice"], 200/201)
        self.assertEqual(result["slice_count"], 11)

    def test_both_empty_excluded_but_false_positive_is_zero(self):
        result = summarize_cases({"empty": torch.tensor([[10, 0], [0, 0]]),
                                  "fp": torch.tensor([[9, 1], [0, 0]])},
                                 {"empty": 1, "fp": 1}, 1)
        self.assertEqual(result["prostate_case_dice"]["undefined_cases"], 1)
        self.assertEqual(result["prostate_case_dice"]["mean"], 0)
        all_empty = summarize_cases({"a": torch.tensor([[10, 0], [0, 0]])}, {"a": 1}, 1)
        self.assertIsNone(all_empty["prostate_case_dice"]["mean"])

    def fixture(self, root, split="val"):
        rows = []
        for index, case in enumerate(("private-A", "private-B", "private-A")):
            image = np.ones((32, 16), dtype=np.float32)
            mask = np.zeros((32, 16), dtype=np.uint8)
            mask[:index+1, :] = 5
            for name, array in ((f"image{index}.nii.gz", image), (f"mask{index}.nii.gz", mask)):
                nib.save(nib.Nifti1Image(array, np.eye(4)), root/name)
            rows.append([f"image{index}.nii.gz", f"mask{index}.nii.gz", case, split])
        # Missing files in other splits prove evaluation does not open them.
        other = "test" if split == "val" else "val"
        rows += [["missing_train.nii.gz", "missing_train_mask.nii.gz", "train-case", "train"],
                 [f"missing_{other}.nii.gz", f"missing_{other}_mask.nii.gz", "other-case", other]]
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask", "patient_id", "split"])
            writer.writerows(rows)
        model = build_model("unet", num_classes=2, base_channels=2)
        for parameter in model.parameters():
            parameter.data.zero_()  # Deterministic all-background prediction.
        payload = {"format_version": 1, "model_state": model.state_dict(), "epoch": 1,
                   "config": {"model": "unet", "base_channels": 2, "image_size": 32,
                              "num_classes": 2, "label_values": [0, 5], "synthetic": False,
                              "prostate_label": 5, "seed": 17, "samples": 2,
                              "resize_mode": "pad", "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}}
        checkpoint = root / "best.pt"
        torch.save(payload, checkpoint)
        return checkpoint, manifest

    def test_real_nifti_validation_only_grouping_and_private_ids(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                checkpoint, manifest = self.fixture(root)
                result = evaluate(checkpoint, manifest, root/"out", "cpu", batch_size=2)
                self.assertEqual(result["case_count"], 2)
                self.assertEqual([c["slices"] for c in result["cases"]], [2, 1])
                self.assertEqual(result["prostate_case_dice"]["mean"], 0)
                self.assertEqual(result["pixel_pooled"]["per_class"][1]["support_pixels"], 96)
                self.assertEqual(result["prostate_class_index"], 1)
                self.assertNotIn("private-A", json.dumps(result))
                details = json.loads((root/"out/case_details_private.json").read_text())
                self.assertEqual(details["Case 01"]["patient_id"], "private-A")
                self.assertFalse(result["test_evaluated"])
                self.assertEqual(json.loads((root/"out/summary.json").read_text()), result)
        finally:
            torch.set_num_threads(previous)

    def test_explicit_test_split_does_not_read_validation_or_modify_checkpoint(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                checkpoint, manifest = self.fixture(root, split="test")
                before = checkpoint.read_bytes()
                result = evaluate(checkpoint, manifest, root/"test_out", "cpu", 2, split="test")
                self.assertEqual(result["split"], "test")
                self.assertTrue(result["test_evaluated"])
                self.assertEqual(result["slice_count"], 3)
                self.assertEqual(result["case_count"], 2)
                self.assertEqual(result["pixel_pooled"]["per_class"][1]["support_pixels"], 96)
                self.assertEqual(result["prostate_case_dice"]["mean"], 0)
                self.assertEqual(checkpoint.read_bytes(), before)
                self.assertEqual(result["checkpoint_sha256"], hashlib.sha256(before).hexdigest())
                self.assertNotIn("private-A", json.dumps(result))
        finally:
            torch.set_num_threads(previous)

    def test_training_or_unknown_split_is_rejected_before_io(self):
        for split in ("train", "all", "validate"):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "split"):
                evaluate(Path("missing"), Path("missing"), Path("missing"), split=split)

    def test_changed_manifest_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, manifest = self.fixture(root)
            with manifest.open("a") as f:
                f.write("\n")
            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                evaluate(checkpoint, manifest, root/"out", "cpu")
            self.assertFalse((root/"out").exists())

    def test_existing_output_and_invalid_batch_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileExistsError):
                evaluate(root/"missing", root/"missing", root, "cpu")
            with self.assertRaisesRegex(ValueError, "batch_size"):
                evaluate(root/"missing", root/"missing", root/"out", "cpu", 0)


if __name__ == "__main__":
    unittest.main()
