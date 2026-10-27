"""Independent selection and evidence-preservation checks for the new verifier."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from verify_id5_runs import compare_original_trajectory, compare_state_trees, independent_selections, protected_paths


TEMP_ROOT = Path(__file__).resolve().parents[3] / "outputs/test_tmp"


def row(epoch, dice, ce):
    return {"epoch": epoch, "validation": {"dice_per_class": [None] * 5 + [dice], "cross_entropy": ce}}


class ID5VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    def test_dice_primary_can_choose_higher_ce(self):
        chosen = independent_selections([row(1, .7, .01), row(2, .8, .2), row(3, .75, .03)])
        self.assertEqual(chosen["primary"]["epoch"], 2)
        self.assertEqual(chosen["id5_best"], chosen["primary"])
        self.assertEqual(chosen["ce_best"]["epoch"], 1)
        self.assertEqual(chosen["last"]["epoch"], 3)

    def test_exact_dice_tie_uses_lower_ce_then_earlier_epoch(self):
        chosen = independent_selections([row(7, .8, .1), row(3, .8, .1), row(1, .8, .2)])
        self.assertEqual(chosen["primary"]["epoch"], 3)
        self.assertEqual(chosen["ce_best"]["epoch"], 3)
        self.assertEqual(chosen["last"]["epoch"], 7)

    def test_near_dice_values_are_not_treated_as_a_tie(self):
        chosen = independent_selections([row(1, .8, .01), row(2, .8000000000000002, .5)])
        self.assertEqual(chosen["primary"]["epoch"], 2)

    def test_undefined_or_nonfinite_values_are_rejected(self):
        for dice in (None, float("nan"), float("inf"), -.01, 1.01, True):
            with self.subTest(dice=dice), self.assertRaises(ValueError):
                independent_selections([row(1, dice, .1)])
        for ce in (float("nan"), float("inf"), -.01, True):
            with self.subTest(ce=ce), self.assertRaises(ValueError):
                independent_selections([row(1, .8, ce)])

    def test_empty_duplicate_and_invalid_epoch_histories_are_rejected(self):
        for history in ([], [row(1, .8, .1), row(1, .9, .2)], [row(0, .8, .1)], [row(True, .8, .1)]):
            with self.subTest(history=history), self.assertRaises(ValueError):
                independent_selections(history)

    def test_protected_file_hash_detects_mutation(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root = Path(folder)
            path = root / "old_summary.json"
            path.write_bytes(b"original")
            hashes = {path.name: hashlib.sha256(b"original").hexdigest()}
            self.assertEqual(protected_paths({"protected_original_files_sha256": hashes}, root), hashes)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                protected_paths({"protected_original_files_sha256": hashes}, root)

    def test_protected_inventory_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root = Path(folder)
            for inventory in ({}, {"../outside.json": "0" * 64}, {str(root / "absolute.json"): "0" * 64}):
                with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                    protected_paths({"protected_original_files_sha256": inventory}, root)

    def test_trajectory_comparison_reports_actual_not_assumed_agreement(self):
        metric = {"confusion_matrix": [[2]], "dice_per_class": [1.0], "iou_per_class": [1.0],
                  "mean_dice_excluding_class0": 1.0, "mean_iou_excluding_class0": 1.0, "pixel_accuracy": 1.0,
                  "valid_pixels": 2, "samples": 1, "batches": 1, "optimizer_steps": 0,
                  "cross_entropy": .1, "elapsed_seconds": 10}
        original = {"epochs": [{"epoch": i, "training": deepcopy(metric), "validation": deepcopy(metric)} for i in range(1, 21)]}
        new = deepcopy(original)
        new["epochs"][0]["validation"]["elapsed_seconds"] = 100
        self.assertTrue(compare_original_trajectory(new, original)["all_epoch_metrics_equal_exactly"])
        new["epochs"][2]["validation"]["cross_entropy"] = .2
        result = compare_original_trajectory(new, original)
        self.assertFalse(result["all_epoch_metrics_equal_exactly"])
        self.assertEqual(result["maximum_ce_absolute_difference"], .1)
        self.assertEqual(result["differences"], [{"epoch": 3, "split": "validation", "field": "cross_entropy", "absolute_difference": .1}])

    def test_tensor_comparison_checks_all_adam_tensors_and_scalar_settings(self):
        original = {"state": {0: {"step": torch.tensor(10.), "exp_avg": torch.tensor([.1, .2])}},
                    "param_groups": [{"lr": .001, "betas": (.9, .999)}]}
        new = deepcopy(original)
        self.assertEqual(compare_state_trees(new, original), ([], 2))
        new["state"][0]["exp_avg"][1] += .01
        new["param_groups"][0]["lr"] = .002
        differences, count = compare_state_trees(new, original)
        self.assertEqual(count, 2)
        self.assertEqual(differences, ["state.param_groups[0].lr", "state.state.0.exp_avg"])

    def test_tensor_comparison_rejects_dtype_and_structure_changes(self):
        differences, count = compare_state_trees({"weight": torch.tensor([1.], dtype=torch.float64)},
                                                {"weight": torch.tensor([1.], dtype=torch.float32), "bias": torch.tensor([0.])})
        self.assertEqual(count, 1)
        self.assertEqual(differences, ["state.keys", "state.weight"])


if __name__ == "__main__":
    unittest.main()
