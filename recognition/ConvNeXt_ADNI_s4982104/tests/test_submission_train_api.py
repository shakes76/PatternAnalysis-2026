"""Recorded-history plot and course-facing train.py CLI modes."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "train.py"


class SubmissionTrainTests(unittest.TestCase):
    def test_plot_mode_renders_recorded_losses_and_patient_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "epochs.jsonl"
            rows = [dict(epoch=n, train={"unweighted_loss": 0.7 - 0.1*n},
                         validation={"loss": 0.8 - 0.05*n},
                         patient_metrics={"macro_f1": 0.4 + 0.1*n})
                    for n in (1, 2, 3)]
            history.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
            output = root / "learning_curves.png"
            run = subprocess.run([sys.executable, str(SCRIPT), "--plot-history",
                "--history", str(history), "--plot-output", str(output)],
                capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            with Image.open(output) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 300)

    def test_final_training_and_test_modes_are_exposed(self):
        for mode, required in (("--formal-convnext", "--n5-reference"),
                               ("--test-mode", "--checkpoint")):
            with self.subTest(mode=mode):
                run = subprocess.run([sys.executable, str(SCRIPT), mode, "--help"],
                    capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn(required, run.stdout)


if __name__ == "__main__":
    unittest.main()
