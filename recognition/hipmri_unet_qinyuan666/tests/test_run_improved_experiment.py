"""Background orchestration must preserve results and stop on subprocess failure."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_improved_experiment import run_experiment


class BackgroundExperimentTests(unittest.TestCase):
    def test_training_failure_records_phase_and_never_evaluates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root / "manifest.csv"
            manifest.write_text("fixture")
            with patch("run_improved_experiment.subprocess.run", side_effect=subprocess.CalledProcessError(1, "train")) as child:
                with self.assertRaises(subprocess.CalledProcessError):
                    run_experiment(root/"run", manifest)
            self.assertEqual(child.call_count, 1)
            status = json.loads((root/"run/status.json").read_text())
            self.assertEqual((status["status"], status["failed_phase"]), ("failed", "training"))

    def test_existing_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root / "manifest.csv"
            manifest.write_text("fixture")
            with patch("run_improved_experiment.subprocess.run") as child:
                with self.assertRaises(FileExistsError):
                    run_experiment(root, manifest)
            child.assert_not_called()
            self.assertEqual(manifest.read_text(), "fixture")

    def test_success_waits_for_validation_and_records_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root / "manifest.csv"
            manifest.write_text("fixture")
            output = root / "run"

            def child(command, **kwargs):
                status = json.loads((output/"status.json").read_text())
                if "--epochs" in command:
                    self.assertEqual(command[command.index("--epochs")+1], "20")
                    self.assertEqual(status["status"], "training")
                    (output/"summary.json").write_text(json.dumps({"status":"completed", "checkpoint_reload_exact":True,
                        "history":[{}]*20, "best_epoch":3}))
                else:
                    self.assertEqual(status["status"], "evaluating")
                    (output/"case_validation").mkdir()
                    (output/"case_validation/summary.json").write_text(json.dumps({"split":"val", "test_evaluated":False,
                        "prostate_case_dice":{"mean":0.8}}))
                return subprocess.CompletedProcess(command, 0)

            with patch("run_improved_experiment.subprocess.run", side_effect=child) as process:
                run_experiment(output, manifest)
            self.assertEqual(process.call_count, 2)
            status = json.loads((output/"status.json").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["best_epoch"], 3)
            self.assertEqual(status["case_mean_prostate_dice"], .8)


if __name__ == "__main__":
    unittest.main()
