"""Check empirical policy failure modes, malformed probabilities and evidence guards."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from policy97_experiment import accepted_mask, verify_policy_binding
from validation_study import class_decisions, fit_class_thresholds, role_split


class PolicyControlTests(unittest.TestCase):
    def test_equality_all_refer_and_invalid_probabilities(self):
        p = torch.tensor([[.75, .25], [.5, .5]], dtype=torch.float64)
        self.assertEqual(accepted_mask(p, .75).tolist(), [True, False])
        self.assertEqual(accepted_mask(p, 1.01).tolist(), [False, False])
        for invalid in [torch.tensor([[.8, .8]]), torch.tensor([[-.1, 1.1]]), torch.tensor([[float("nan"), .5]])]:
            with self.assertRaises(ValueError):
                accepted_mask(invalid, .7)

    def test_unqualified_predicted_class_refers_all(self):
        p = torch.tensor([[.8, .1, .05, .05]] * 60 + [[.1, .8, .05, .05]] * 60)
        labels = torch.tensor([0] * 60 + [1] * 58 + [0] * 2)
        thresholds, record = fit_class_thresholds(p, labels)
        self.assertEqual(thresholds[0], 0.)
        self.assertEqual(thresholds[1:], [1.01, 1.01, 1.01])
        decision = class_decisions(p, thresholds)
        self.assertTrue(bool(decision[:60].all()))
        self.assertFalse(bool(decision[60:].any()))
        self.assertFalse(record[1]["feasible_on_fitting_data"])

    def test_routes_depend_on_predicted_class(self):
        p = torch.tensor([[.7, .1, .1, .1], [.1, .7, .1, .1]])
        self.assertEqual(class_decisions(p, [.6, .8, 1.01, 1.01]).tolist(), [True, False])
        with self.assertRaises(ValueError):
            class_decisions(p, [.6, float("nan"), .9, .9])

    def test_five_label_roles_are_disjoint_and_reproducible(self):
        import numpy as np
        y = np.repeat(np.arange(4), 100)
        first, second = role_split(y), role_split(y)
        self.assertTrue(np.array_equal(np.sort(np.concatenate(list(first.values()))), np.arange(400)))
        for name in first:
            self.assertTrue(np.array_equal(first[name], second[name]))
            self.assertEqual(set(y[first[name]]), {0, 1, 2, 3})

    def test_wrong_checkpoint_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.bin"
            path.write_bytes(b"unrelated checkpoint")
            policy = {"checkpoint_sha256": "0" * 64, "dataset_sha256": "example", "initialization_seed": 42}
            with self.assertRaises(ValueError):
                verify_policy_binding(policy, path, "example", 42)

    def test_partial_experiment_folder_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "partial"
            path.mkdir()
            marker = path / "policy.json"
            marker.write_text("original evidence", encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("policy97_experiment.py")),
                                     "--output", str(path)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("partial experiment folders are preserved", result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8"), "original evidence")


if __name__ == "__main__":
    unittest.main()
