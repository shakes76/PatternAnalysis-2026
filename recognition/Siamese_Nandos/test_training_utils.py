"""Check recovery guards, progress output and preservation of saved checkpoints."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from training_utils import atomic_save, capture_rng, restore_rng, prepare_output, Progress


class RecoveryTests(unittest.TestCase):
    def test_rng_round_trip(self):
        state = capture_rng()
        expected = (random.random(), np.random.random(), torch.rand(3))
        restore_rng(state)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        self.assertTrue(torch.equal(torch.rand(3), expected[2]))

    def test_failed_save_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_save({"epoch": 1}, path)
            with patch("training_utils.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    atomic_save({"epoch": 2}, path)
            self.assertEqual(torch.load(path, weights_only=False)["epoch"], 1)

    def test_output_guard_and_legacy_best_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            prepare_output(output, False)
            (output / "best.pt").write_bytes(b"existing weights")
            with self.assertRaises(FileExistsError):
                prepare_output(output, False)
            with self.assertRaises(FileNotFoundError):
                prepare_output(output, True)

    def test_progress_has_batch_phase_and_eta(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            progress = Progress("train", 2, 10, 100, 25)
            progress.update(1, 0.5)
            progress.update(25, 0.4)
            progress.update(100, 0.3)
        output = stream.getvalue()
        self.assertIn("Epoch 2/10 train", output)
        self.assertIn("batch 25/100", output)
        self.assertIn("loss=0.40000", output)
        self.assertIn("phase_ETA=", output)


if __name__ == "__main__":
    unittest.main()
