"""Real-checkpoint inference, artifact validation and fresh-process CLI checks."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dataset import load_facebook
from evaluate import PROJECT
from predict import DEFAULT_POLICY, EXAMPLE_SELECTION_NODES, read_policy, run_prediction, select_nodes


class RealPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_facebook()
        cls.checkpoint = PROJECT / "outputs/gcn_seed42/best.pt"

    def test_predictions_use_no_targets_and_match_frozen_failure_report(self):
        expected = json.loads((PROJECT / "outputs/confidence_seed42/confidence_summary.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(self.dataset, "targets", side_effect=AssertionError("Inference requested labels")):
                with redirect_stdout(io.StringIO()):
                    rows, summary = run_prediction(self.dataset, self.checkpoint, DEFAULT_POLICY,
                                                   Path(directory), device="cpu")
            self.assertFalse(summary["labels_accessed"])
            self.assertEqual(summary["nodes"], 5)
            for row, case in zip(rows, expected["failure_cases"]):
                self.assertEqual(row["node_id"], case["node_id"])
                self.assertEqual(row["predicted_class"], case["GCN"]["predicted_class"])
                self.assertAlmostEqual(row["calibrated_confidence"], case["GCN"]["calibrated_confidence"], places=6)
                self.assertEqual(row["decision"] == "refer", case["GCN"]["referred"])
                self.assertAlmostEqual(sum(v for k, v in row.items() if k.startswith("probability_")), 1.)
            for file in ("predictions.csv", "prediction_summary.json", "predictions.png", "predictions.svg"):
                self.assertGreater((Path(directory) / file).stat().st_size, 0)
            with self.assertRaises(FileExistsError):
                run_prediction(self.dataset, self.checkpoint, DEFAULT_POLICY, Path(directory))

    def test_node_selection_rejects_wrong_partition_duplicate_and_invalid_ids(self):
        np.testing.assert_array_equal(select_nodes(self.dataset, "selection"), EXAMPLE_SELECTION_NODES)
        # Find a training node using its mask only; no test labels or predictions.
        train_node = int(np.flatnonzero(self.dataset.masks["train"].numpy())[0])
        for ids in ([train_node], [8505, 8505], [-1], [22470], []):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                select_nodes(self.dataset, "selection", ids)

    def test_policy_rejects_changed_checkpoints_roles_and_invalid_parameters(self):
        original = json.loads(DEFAULT_POLICY.read_text())
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "invalid_policy.json"
            for field, value in (("temperature", 0), ("threshold", 1.1), ("checkpoint_sha256", "bad")):
                altered = json.loads(json.dumps(original))
                altered["models"]["GCN"][field] = value
                file.write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    read_policy(file, self.checkpoint, self.dataset, "GCN")
            original["temperature_fitting_partition"] = "selection"
            file.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "fitting roles"):
                read_policy(file, self.checkpoint, self.dataset, "GCN")

    def test_fresh_process_runs_for_both_models_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("mlp", "gcn"):
                output = Path(directory) / name
                result = subprocess.run(
                    [sys.executable, str(PROJECT / "predict.py"), "--model", name,
                     "--device", "cpu", "--output-dir", str(output)],
                    cwd=directory, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("node 8505:", result.stdout)
                summary = json.loads((output / "prediction_summary.json").read_text())
                self.assertEqual(summary["model"], name.upper())
                self.assertFalse(summary["labels_accessed"])

    def test_missing_weights_and_policy_have_clear_errors_before_data_load(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_data = Path(directory) / "no-data"
            for extra, message in ((["--checkpoint", str(Path(directory) / "missing.pt")], "Weights not found"),
                                   (["--policy", str(Path(directory) / "missing.json")], "Frozen policy not found")):
                result = subprocess.run(
                    [sys.executable, str(PROJECT / "predict.py"), "--data-root", str(missing_data), *extra],
                    cwd=directory, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(missing_data.exists())


if __name__ == "__main__":
    unittest.main()
