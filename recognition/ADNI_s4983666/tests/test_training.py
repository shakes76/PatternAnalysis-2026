"""Exercise actual CPU training and checkpoint inference on synthetic data."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import torch

import predict
import train
from modules import SmallCNN
import test_adni_splits as split_fixture


class BaselineTrainingTests(unittest.TestCase):
    def setUp(self):
        # Reuse only fixture construction; importing the module avoids
        # rediscovering its TestCase as another local test class.
        self.fixture = split_fixture.ADNISplitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare()
        self.output = self.fixture.base / "run"
        self.args = argparse.Namespace(
            data_root=self.fixture.root, splits_dir=self.fixture.out, output=self.output,
            fold=1, epochs=4, patience=1, batch_size=16, workers=0, threads=1,
            lr=0.001, weight_decay=0.0001, min_delta=0.0, seed=3710,
            image_height=32, image_width=32, device="cpu",
        )

    def test_training_selects_on_early_stop_then_predict_reproduces_outer_val(self):
        """Force an early-stop trajectory while running real forward/backward passes."""
        early_rows = split_fixture.read_csv(self.fixture.out / "fold_01/early_stop.csv")
        val_rows = split_fixture.read_csv(self.fixture.out / "fold_01/val.csv")
        train_rows = split_fixture.read_csv(self.fixture.out / "fold_01/train.csv")
        early_paths = split_fixture.paths(early_rows)
        val_paths = split_fixture.paths(val_rows)
        calls = []
        real_evaluate = train.evaluate

        def observed_evaluate(model, loader, device, expected_slices):
            paths = split_fixture.paths(loader.dataset.rows)
            scores, slices, scans = real_evaluate(model, loader, device, expected_slices)
            if paths == early_paths:
                self.assertNotIn("val", calls)
                calls.append("early_stop")
                scores["scan"]["log_loss"] = 0.6 if len(calls) == 1 else 0.8
            else:
                self.assertEqual(paths, val_paths)
                self.assertTrue((self.output / "best.pt").exists())
                calls.append("val")
            return scores, slices, scans

        with mock.patch("train.evaluate", side_effect=observed_evaluate), contextlib.redirect_stdout(io.StringIO()):
            result = train.run(self.args)
        self.assertEqual(calls, ["early_stop", "early_stop", "val"])
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["epochs_completed"], 2)
        self.assertEqual(result["evaluation_role"], "development_outer_validation")
        self.assertIsNone(result["resources"]["peak_cuda_allocated_mib"])
        self.assertTrue((self.output / "learning_curves.png").is_file())
        saved = torch.load(self.output / "best.pt", map_location="cpu", weights_only=True)
        counts = {label: sum(row["label"] == label for row in train_rows) for label in ("0", "1")}
        self.assertEqual(saved["config"]["train_slice_class_counts"], counts)
        self.assertEqual(saved["config"]["train_pos_weight"], counts["0"] / counts["1"])
        scan_rows = split_fixture.read_csv(self.output / "val_scan_predictions.csv")
        self.assertEqual(len(scan_rows), len({row["image_id"] for row in val_rows}))
        self.assertEqual(split_fixture.patients(scan_rows), split_fixture.patients(val_rows))
        self.assertTrue(all(row["num_slices"] == "2" for row in scan_rows))
        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in saved["model_state"].values()))

        # Another fold gets a new constructor; checkpoints load only for inference.
        torch.manual_seed(saved["config"]["seed"])
        initial = SmallCNN().state_dict()
        self.assertTrue(any(not torch.equal(initial[name], value) for name, value in saved["model_state"].items()))
        prediction_dir = self.fixture.base / "prediction"
        prediction_args = argparse.Namespace(
            checkpoint=self.output / "best.pt", data_root=self.fixture.root,
            splits_dir=self.fixture.out, output=prediction_dir,
            batch_size=16, workers=0, threads=1, device="cpu",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            predict.run(prediction_args)
        reproduced = json.loads((prediction_dir / "metrics.json").read_text())
        self.assertEqual(reproduced["metrics"]["scan"], result["metrics"]["scan"])
        self.assertEqual(split_fixture.read_csv(prediction_dir / "scan_predictions.csv"), scan_rows)
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            train.run(self.args)

        # A valid but different manifest set must not be accepted by a checkpoint.
        saved["config"]["manifest_sha256"] = "0" * 64
        torch.save(saved, self.output / "wrong_split.pt")
        prediction_args.checkpoint = self.output / "wrong_split.pt"
        prediction_args.output = self.fixture.base / "wrong_prediction"
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "different set"):
            predict.run(prediction_args)
        self.assertFalse(prediction_args.output.exists())

    def test_changed_manifest_stops_training_before_run_creation(self):
        path = self.fixture.out / "fold_01/train.csv"
        path.write_text(path.read_text() + "unexpected,row\n")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "modified"):
            train.run(self.args)
        self.assertFalse(self.output.exists())

    def test_cnn_accepts_native_grayscale_resolution_and_batch_of_one(self):
        torch.set_num_threads(1)
        model = SmallCNN().eval()
        with torch.inference_mode():
            logits = model(torch.zeros(1, 1, 240, 256))
        self.assertEqual(tuple(logits.shape), (1,))
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
