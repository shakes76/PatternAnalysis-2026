"""End-to-end CPU training, calibration and test prediction on synthetic data."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import torch

from engine import prediction as predict
from engine import training as train
import test_adni_splits as split_fixture


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        # Reuse the synthetic dataset/manifests built by the split tests.
        self.fixture = split_fixture.ADNISplitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare()
        self.output = self.fixture.base / "run"
        torch.set_num_threads(1)

    def train_args(self, *extra: str):
        return train.build_parser().parse_args([
            "--data-root", str(self.fixture.root), "--splits-dir", str(self.fixture.out),
            "--output", str(self.output), "--model", "cnn", "--epochs", "3",
            "--warmup-epochs", "0", "--batch-size", "8", "--workers", "0", "--threads", "1", "--lr", "1e-3",
            "--image-height", "32", "--image-width", "32", "--device", "cpu",
            "--skip-inference-profile", *extra])

    def predict(self, role: str, output_name: str, *extra: str) -> dict:
        args = predict.build_parser().parse_args([
            "--checkpoint", str(self.output / "final.pt"), "--data-root", str(self.fixture.root),
            "--splits-dir", str(self.fixture.out), "--output", str(self.fixture.base / output_name),
            "--role", role, "--workers", "0", "--threads", "1", "--device", "cpu",
            "--skip-inference-profile", *extra])
        with contextlib.redirect_stdout(io.StringIO()):
            return predict.run(args)

    def test_final_epoch_checkpoint_then_calibration_then_test(self):
        fold = self.fixture.out / "fold_01"
        train_paths = (split_fixture.paths(split_fixture.read_csv(fold / "train.csv"))
                       | split_fixture.paths(split_fixture.read_csv(fold / "early_stop.csv")))
        val_paths = split_fixture.paths(split_fixture.read_csv(fold / "val.csv"))
        scored, trained_on = [], []
        real_evaluate, real_loader = train.evaluate, train.make_loader

        def observed_evaluate(model, loader, device, expected_slices, controls):
            scored.append("val" if split_fixture.paths(loader.dataset.rows) == val_paths else "other")
            return real_evaluate(model, loader, device, expected_slices, controls)

        def observed_loader(rows, *args, **kwargs):
            if kwargs["role"] == "train":
                trained_on.append(split_fixture.paths(rows))
            return real_loader(rows, *args, **kwargs)

        with mock.patch("engine.training.evaluate", side_effect=observed_evaluate), \
                mock.patch("engine.training.make_loader", side_effect=observed_loader), \
                contextlib.redirect_stdout(io.StringIO()):
            result = train.run(self.train_args())
        # The early_stop manifest is part of training; val is only monitored, once per epoch.
        self.assertEqual(trained_on, [train_paths])
        self.assertTrue(train_paths.isdisjoint(val_paths))
        self.assertEqual(scored, ["val"] * 3)
        self.assertEqual(result["evaluation_role"], "development_validation")
        self.assertEqual(result["checkpoint_selection"], "final_epoch")
        self.assertTrue((self.output / "learning_curves.png").is_file())
        history = split_fixture.read_csv(self.output / "history.csv")
        self.assertIn("train_peak_cuda_mib", history[0])  # Per-epoch GPU memory column (empty on CPU).
        # Reported metrics are those of the last epoch, i.e. of the saved weights.
        self.assertAlmostEqual(float(history[-1]["val_scan_accuracy"]), result["metrics"]["scan"]["accuracy"])
        saved = torch.load(self.output / "final.pt", map_location="cpu", weights_only=True)
        self.assertEqual(saved["epoch"], 3)
        self.assertEqual(saved["config"]["execution_controls"]["context_slices"], 3)
        self.assertEqual(saved["config"]["augmentation_config"]["name"], "strong")
        self.assertEqual(saved["config"]["checkpoint_selection"], "final_epoch")
        self.assertEqual(saved["model_state"]["features.0.weight"].shape[1], 3)

        # Reproduce the val result from the saved checkpoint.
        reproduced = self.predict("val", "val_prediction")
        self.assertEqual(reproduced["metrics"]["scan"]["accuracy"], result["metrics"]["scan"]["accuracy"])

        # Test needs a calibration fitted for this exact checkpoint.
        with self.assertRaisesRegex(ValueError, "calibration-file"):
            self.predict("test", "test_without_calibration")
        calibration = self.predict("calibration", "calibration")["calibration"]
        self.assertIn("platt_intercept", calibration)
        calibration_file = self.fixture.base / "calibration" / "calibration.json"
        tested = self.predict("test", "test", "--calibration-file", str(calibration_file))
        self.assertEqual(tested["role"], "test")
        self.assertEqual(tested["calibration"]["platt_slope"], calibration["platt_slope"])
        self.assertTrue((self.fixture.base / "test" / "test_examples.png").is_file())
        for unit in ("slice", "scan", "patient"):
            self.assertIn("accuracy", tested["metrics"][unit])

        wrong = json.loads(calibration_file.read_text())
        wrong["checkpoint_sha256"] = "0" * 64
        wrong_file = self.fixture.base / "wrong_calibration.json"
        wrong_file.write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            self.predict("test", "test_wrong", "--calibration-file", str(wrong_file))

        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            train.run(self.train_args())

    def test_convnext_with_ema_and_mixup(self):
        args = self.train_args("--model", "convnext_lite", "--ema-decay", "0.9", "--mixup-alpha", "0.2",
                               "--context-slices", "1", "--epochs", "1")
        with contextlib.redirect_stdout(io.StringIO()):
            train.run(args)
        saved = torch.load(self.output / "final.pt", map_location="cpu", weights_only=True)
        self.assertEqual(saved["config"]["ema_decay"], 0.9)

    def test_selection_options_are_gone(self):
        for option in ("--selection-metric", "--patience", "--inner-only"):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                self.train_args(option, "1")

    def test_plan_runner_trains_and_verifies_checkpoint_replay(self):
        spec = importlib.util.spec_from_file_location("run_plan", Path(train.__file__).parents[1] / "slurm/run_plan.py")
        run_plan = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_plan)
        plan = {"name": "tiny_plan", "common": {"epochs": 1, "warmup_epochs": 0, "batch_size": 8, "workers": 0,
                                                 "threads": 1, "image_height": 32, "image_width": 32, "device": "cpu",
                                                 "skip_inference_profile": True},
                "cases": [{"id": "C1", "purpose": "test", "args": {"model": "cnn", "lr": 0.001}}]}
        plan_file = self.fixture.base / "plan.json"
        plan_file.write_text(json.dumps(plan))
        runs = self.fixture.base / "plan_runs"
        argv = [str(plan_file), "--data-root", str(self.fixture.root), "--splits-dir", str(self.fixture.out),
                "--runs-root", str(runs)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_plan.main(argv), 0)
            self.assertEqual(run_plan.main(argv), 0)  # Second call resumes: nothing retrained.
        check = json.loads((runs / "tiny_plan/C1/replay_check.json").read_text())
        self.assertTrue(check["match"])
        self.assertEqual(check["role"], "val")

        spec = importlib.util.spec_from_file_location(
            "summarize_runs", Path(train.__file__).parents[1] / "slurm/summarize_runs.py")
        summarize = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(summarize)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(summarize.main([str(runs / "tiny_plan"), "--output", str(runs / "summary")]), 0)
        row = split_fixture.read_csv(runs / "summary.csv")[0]
        metrics = json.loads((runs / "tiny_plan/C1/metrics.json").read_text())["metrics"]
        self.assertEqual(row["group"], "C1")
        self.assertAlmostEqual(float(row["patient_accuracy"]), metrics["patient"]["accuracy"])
        self.assertEqual(row["replay_match"], "True")

    def test_changed_manifest_stops_training_before_run_creation(self):
        path = self.fixture.out / "fold_01/train.csv"
        path.write_text(path.read_text() + "unexpected,row\n")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "modified"):
            train.run(self.train_args())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
