"""End-to-end software checks using artificial NIfTI fixtures, never HipMRI.

The raw labels and patient identities below are invented test data. No medical
performance result or real patient data is produced or committed by these tests.
"""

import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import build_model
from predict import run_prediction
from train import parse_args, train


class TrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        cls.previous_deterministic = torch.are_deterministic_algorithms_enabled()
        cls.previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cls.previous_cudnn_benchmark = torch.backends.cudnn.benchmark
        cls.previous_cudnn_deterministic = torch.backends.cudnn.deterministic
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        torch.use_deterministic_algorithms(cls.previous_deterministic,
                                          warn_only=cls.previous_warn_only)
        torch.backends.cudnn.benchmark = cls.previous_cudnn_benchmark
        torch.backends.cudnn.deterministic = cls.previous_cudnn_deterministic

    def _manifest(self, root):
        """Create train/val files; test files deliberately do not exist."""
        rows = []
        for index, split in enumerate(("train", "train", "val")):
            image = np.arange(32 * 40, dtype=np.float32).reshape(32, 40) + index
            mask = np.zeros((32, 40), dtype=np.int16)
            mask[4:9, 6:12] = 3
            mask[12:24, 16:30] = 7
            for name, array in (("image", image), ("mask", mask)):
                nib.save(nib.Nifti1Image(array, np.diag([0.8, 0.9, 1.0, 1.0])),
                         root / f"fixture_{index}_{name}.nii.gz")
            rows.append([f"fixture_{index}_image.nii.gz", f"fixture_{index}_mask.nii.gz",
                         f"invented-patient-{index}", split])
        rows.append(["unavailable_test_image.nii.gz", "unavailable_test_mask.nii.gz",
                     "invented-test-patient", "test"])
        manifest = root / "artificial_fixture_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "mask", "patient_id", "split"])
            writer.writerows(rows)
        return manifest

    def _synthetic_checkpoint(self):
        return {
            "format_version": 1, "epoch": 1,
            "model_state": build_model("unet", base_channels=2).state_dict(),
            "config": {"model": "unet", "base_channels": 2, "image_size": 32,
                       "num_classes": 2, "label_values": [0, 1], "synthetic": True,
                       "prostate_label": None, "seed": 17, "samples": 1},
        }

    def test_manifest_training_selection_reload_and_prediction_for_both_models(self):
        for model in ("unet", "improved_unet"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = self._manifest(root)
                output = root / "run"
                args = parse_args([
                    "--model", model, "--manifest", str(manifest),
                    "--label-values", "0", "3", "7", "--prostate-label", "7",
                    "--image-size", "32", "--base-channels", "2", "--batch-size", "1",
                    "--epochs", "2", "--threads", "1", "--device", "cpu",
                    "--output", str(output),
                ])
                with contextlib.redirect_stdout(io.StringIO()):
                    result = train(args)
                self.assertEqual(result["training_samples"], 2)
                self.assertEqual(result["validation_samples"], 1)
                self.assertFalse(result["test_used_for_selection"])
                self.assertTrue(result["checkpoint_reload_exact"])
                self.assertEqual(result["checkpoint_reload_max_abs_error"], 0)
                for row in result["history"]:
                    self.assertEqual(sum(c["support_pixels"] for c in row["train"]["per_class"]),
                                     2 * 32 * 32)
                    self.assertEqual(sum(c["support_pixels"] for c in row["val"]["per_class"]),
                                     32 * 32)
                expected_best = min(result["history"], key=lambda row: row["val"]["loss"])
                checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["epoch"], expected_best["epoch"])
                self.assertEqual(result["best_epoch"], expected_best["epoch"])
                self.assertEqual(result["best_validation"], expected_best["val"])
                self.assertEqual(checkpoint["config"]["manifest_sha256"],
                                 hashlib.sha256(manifest.read_bytes()).hexdigest())
                self.assertEqual(json.loads((output / "summary.json").read_text("utf-8")), result)
                self.assertGreater((output / "training_curves.png").stat().st_size, 1000)

                prediction = run_prediction(output / "best.pt", root / "predictions",
                                            manifest=manifest, device="cpu", split="val", limit=1)
                self.assertEqual(prediction["foreground_channel"], 2)
                self.assertEqual(prediction["foreground_raw_label"], 7)
                self.assertEqual(prediction["examples"][0]["patient_id"], "invented-patient-2")
                self.assertGreater((root / "predictions" / "prediction_grid.png").stat().st_size, 1000)
                # Training and validation prediction succeeded without opening these missing files.
                with self.assertRaises(FileNotFoundError):
                    run_prediction(output / "best.pt", root / "test_predictions",
                                   manifest=manifest, device="cpu", split="test", limit=1)
                self.assertFalse((root / "test_predictions").exists())
                with self.assertRaises(FileExistsError):
                    train(args)

                changed = root / "changed.csv"
                changed.write_text(manifest.read_text("utf-8").replace(",train", ",val"), "utf-8")
                with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                    run_prediction(output / "best.pt", root / "changed_predictions",
                                   manifest=changed, device="cpu", limit=1)
                self.assertFalse((root / "changed_predictions").exists())

    def test_nonfinite_learning_rates_are_rejected_before_training(self):
        for rate in ("nan", "inf", "-inf"):
            with self.subTest(rate=rate), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--model", "unet", "--synthetic", "--lr=" + rate,
                                "--output", "unused"])

    def test_malformed_checkpoint_configuration_is_rejected(self):
        payload = self._synthetic_checkpoint()
        invalid = (("model", "unknown", "model"),
                   ("num_classes", 2.0, "num_classes"),
                   ("label_values", None, "label_values"),
                   ("base_channels", True, "base_channels"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for key, value, message in invalid:
                with self.subTest(field=key):
                    changed = copy.deepcopy(payload)
                    changed["config"][key] = value
                    torch.save(changed, root / "invalid.pt")
                    with self.assertRaisesRegex(ValueError, message):
                        run_prediction(root / "invalid.pt", root / "out", device="cpu", synthetic=True)
            self.assertFalse((root / "out").exists())

    def test_checkpoint_weights_must_match_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self._synthetic_checkpoint()
            payload["config"]["base_channels"] = 3
            torch.save(payload, root / "invalid.pt")
            with self.assertRaisesRegex(ValueError, "incompatible"):
                run_prediction(root / "invalid.pt", root / "out", device="cpu", synthetic=True)
            self.assertFalse((root / "out").exists())

    def test_nonfinite_checkpoint_predictions_cannot_be_reported_as_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self._synthetic_checkpoint()
            payload["model_state"]["head.bias"].fill_(float("nan"))
            torch.save(payload, root / "invalid.pt")
            with self.assertRaisesRegex(ValueError, "non-finite logits"):
                run_prediction(root / "invalid.pt", root / "out", device="cpu", synthetic=True)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
