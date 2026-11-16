"""Checkpoint provenance guards and actual prediction artifact generation."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import build_model
from predict import run_prediction


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def _checkpoint(self, path, synthetic=True, prostate_label=None):
        payload = {
            "format_version": 1,
            "model_state": build_model("unet", base_channels=2).state_dict(),
            "epoch": 1,
            "config": {"model": "unet", "base_channels": 2, "image_size": 32,
                       "num_classes": 2, "label_values": [0, 1], "synthetic": synthetic,
                       "prostate_label": prostate_label, "seed": 17, "samples": 2},
        }
        torch.save(payload, path)

    def test_reload_writes_explicitly_synthetic_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._checkpoint(root / "model.pt")
            result = run_prediction(root / "model.pt", root / "out", device="cpu",
                                    limit=1, synthetic=True)
            self.assertEqual(result["data_kind"], "synthetic")
            self.assertEqual(result["prediction_seed"], 2017)
            self.assertEqual(result["examples_shown"], 1)
            self.assertIn("not HipMRI", result["interpretation"])
            self.assertGreater((root / "out" / "prediction_grid.png").stat().st_size, 1000)
            saved = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, result)

    def test_cannot_relabel_checkpoint_data_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for checkpoint_mode in (True, False):
                self._checkpoint(root / "model.pt", synthetic=checkpoint_mode, prostate_label=1)
                with self.assertRaisesRegex(ValueError, "must match"):
                    run_prediction(root / "model.pt", root / "out", device="cpu",
                                   synthetic=not checkpoint_mode)
            self.assertFalse((root / "out").exists())

    def test_real_prediction_requires_verified_label_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._checkpoint(root / "model.pt", synthetic=False)
            with self.assertRaisesRegex(ValueError, "prostate_label"):
                run_prediction(root / "model.pt", root / "out", device="cpu")
            self._checkpoint(root / "model.pt", synthetic=False, prostate_label=1)
            with self.assertRaisesRegex(ValueError, "requires --manifest"):
                run_prediction(root / "model.pt", root / "out", device="cpu")


if __name__ == "__main__":
    unittest.main()
