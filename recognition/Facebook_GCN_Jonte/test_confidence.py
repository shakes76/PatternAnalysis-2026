"""Analytic calibration/policy fixtures and real-checkpoint integrity checks."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from confidence import (ECE_BINS, MIN_ACCEPTED, TARGET_ERROR, acceptance_mask,
                        apply_policy, choose_threshold, collect_logits, failure_cases,
                        fit_temperature, probabilities, reliability_metrics, risk_curve,
                        run_analysis)
from dataset import load_facebook
from evaluate import PROJECT
from train import TrainConfig, configure_reproducibility


class CalibrationUnitTests(unittest.TestCase):
    def setUp(self):
        configure_reproducibility(TrainConfig(device="cpu"))

    def test_analytic_ece_nll_brier_and_empty_bins(self):
        probs = np.array([[.8, .2], [.4, .6]])
        logits = np.log(probs)
        result = reliability_metrics(logits, np.array([0, 0]), bins=2)
        self.assertAlmostEqual(result["ece"], .2)
        self.assertAlmostEqual(result["nll"], -np.log(.8 * .4) / 2)
        self.assertAlmostEqual(result["brier"], (.08 + .72) / 2)
        self.assertEqual(result["bins"][0]["count"], 0)
        self.assertIsNone(result["bins"][0]["accuracy"])
        self.assertAlmostEqual(result["confidence_groups"]["correct"]["mean_confidence"], .8)
        self.assertAlmostEqual(result["confidence_groups"]["incorrect"]["mean_confidence"], .6)

    def test_closed_last_bin_and_confident_wrong_prediction(self):
        result = reliability_metrics([[1000., -1000.]], np.array([1]))
        self.assertEqual(result["bins"][-1]["count"], 1)
        self.assertEqual(result["ece"], 1)
        self.assertEqual(result["nll"], 2000)
        self.assertEqual(result["brier"], 2)

    def test_temperature_known_optimum_and_argmax_invariance(self):
        # Eight correct / two incorrect at common logits: optimum p(true top)=.8.
        logits = np.tile([4., 0.], (10, 1))
        targets = np.array([0] * 8 + [1] * 2)
        result = fit_temperature(logits, targets)
        self.assertAlmostEqual(result["temperature"], 4 / np.log(4), places=8)
        self.assertLess(result["nll_after"], result["nll_before"])
        np.testing.assert_array_equal(probabilities(logits).argmax(1),
                                      probabilities(logits, result["temperature"]).argmax(1))
        self.assertEqual(result, fit_temperature(logits, targets))

    def test_temperature_boundary_flat_logits_and_validation(self):
        boundary = fit_temperature(np.tile([4., 0.], (8, 1)), np.zeros(8, dtype=np.int64))
        self.assertTrue(boundary["at_boundary"])
        self.assertLessEqual(boundary["nll_after"], boundary["nll_before"])
        flat = fit_temperature(np.zeros((4, 2)), np.array([0, 0, 1, 1]))
        self.assertEqual(flat["temperature"], 1)
        for logits, targets in (([[float("nan"), 0]], np.array([0])),
                                ([[1., 0]], np.array([-1])),
                                ([[1., 0]], np.array([0.]))):
            with self.assertRaises(ValueError):
                fit_temperature(logits, targets)
        with self.assertRaises(ValueError):
            probabilities([[1., 0]], temperature=0)

    def test_policy_highest_coverage_and_whole_confidence_ties(self):
        # Last tied pair introduces one error. Partial acceptance of the tie
        # would be invalid; only coverage 2/4 is feasible at error <= .1.
        probs = np.array([[.9, .1], [.8, .2], [.7, .3], [.7, .3]])
        targets = np.array([0, 0, 0, 1])
        curve = risk_curve(probs, targets)
        self.assertEqual([row["accepted"] for row in curve], [1, 2, 4])
        policy = choose_threshold(probs, targets, target_error=.1, min_accepted=2)
        self.assertEqual(policy["threshold"], .8)
        self.assertEqual(policy["accepted"], 2)
        self.assertEqual(apply_policy(probs, targets, .8)["coverage"], .5)
        self.assertEqual(acceptance_mask(probs, .7).sum(), 4)
        # Risk need not be monotone: a later correct case may restore feasibility.
        other = np.array([[.9, .1], [.8, .2], [.7, .3]])
        best = choose_threshold(other, np.array([0, 1, 0]), target_error=.34, min_accepted=1)
        self.assertEqual(best["accepted"], 3)

    def test_no_feasible_threshold_refers_all_and_reports_undefined_error(self):
        probs = np.array([[.9, .1], [.8, .2]])
        chosen = choose_threshold(probs, np.array([1, 1]), min_accepted=1)
        self.assertFalse(chosen["feasible"])
        self.assertIsNone(chosen["threshold"])
        report = apply_policy(probs, np.array([1, 1]), chosen["threshold"])
        self.assertEqual(report["coverage"], 0)
        self.assertEqual(report["referral_rate"], 1)
        self.assertIsNone(report["accepted_error"])
        with self.assertRaises(ValueError):
            choose_threshold(probs, np.array([1, 1]), min_accepted=0)


class RealConfidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_facebook()
        cls.mlp = PROJECT / "outputs/mlp_seed42/best.pt"
        cls.gcn = PROJECT / "outputs/gcn_seed42/best.pt"
        cls.device = configure_reproducibility(TrainConfig(device="cpu"))
        cls.logits = {name: collect_logits(cls.dataset, path, name, cls.device)
                      for name, path in (("MLP", cls.mlp), ("GCN", cls.gcn))}

    def test_case_ids_predictions_degrees_and_training_only_neighbors(self):
        selection = self.dataset.masks["selection"].numpy()
        logits = {name: values[selection] for name, values in self.logits.items()}
        cases = failure_cases(self.dataset, logits, {"MLP": 1., "GCN": 1.},
                              {"MLP": .9, "GCN": .9})
        self.assertEqual(len(cases), 5)
        self.assertEqual(len({row["node_id"] for row in cases}), 5)
        self.assertEqual(cases, failure_cases(self.dataset, logits, {"MLP": 1., "GCN": 1.},
                                              {"MLP": .9, "GCN": .9}))
        ids = np.flatnonzero(selection)
        truth = self.dataset.targets("selection").numpy()
        for row in cases:
            index = int(np.flatnonzero(ids == row["node_id"])[0])
            self.assertEqual(row["true_class_id"], truth[index])
            self.assertTrue(any(row[name]["predicted_class_id"] != truth[index] for name in ("MLP", "GCN")))
            for name in ("MLP", "GCN"):
                probs = probabilities(logits[name][index:index+1])[0]
                self.assertEqual(row[name]["predicted_class_id"], int(probs.argmax()))
                self.assertAlmostEqual(row[name]["raw_confidence"], float(probs.max()))
                self.assertEqual(row[name]["referred"], bool(probs.max() < .9))
            edge = self.dataset.graph.edge_index.numpy()
            neighbors = set(edge[1, edge[0] == row["node_id"]]) - {row["node_id"]}
            self.assertEqual(row["unique_nonself_degree"], len(neighbors))
            train = self.dataset.masks["train"].numpy()
            self.assertEqual(row["training_labeled_neighbors"], sum(train[node] for node in neighbors))
            self.assertEqual(sum(row["training_neighbor_class_counts"].values()), row["training_labeled_neighbors"])

    def test_real_fit_and_threshold_match_bruteforce_candidate_search(self):
        for name, logits in self.logits.items():
            fit = fit_temperature(logits[self.dataset.masks["calibration"]], self.dataset.targets("calibration"))
            self.assertLessEqual(fit["nll_after"], fit["nll_before"])
            probs = probabilities(logits[self.dataset.masks["policy"]], fit["temperature"])
            targets = self.dataset.targets("policy").numpy()
            chosen = choose_threshold(probs, targets)
            feasible = []
            for threshold in np.unique(probs.max(1)):
                result = apply_policy(probs, targets, threshold)
                if result["accepted"] >= MIN_ACCEPTED and result["accepted_error"] <= TARGET_ERROR:
                    feasible.append(result["accepted"])
            self.assertEqual(chosen["accepted"], max(feasible, default=0), name)

    def test_end_to_end_test_label_guard_exports_and_policy_freeze(self):
        poisoned = replace(self.dataset, _targets={
            name: (torch.full_like(values, -999) if name == "test" else values.clone())
            for name, values in self.dataset._targets.items()})
        access = poisoned.targets

        def permitted(name, **kwargs):
            if name not in ("train", "selection", "calibration", "policy"):
                raise AssertionError("Test labels accessed")
            return access(name, **kwargs)

        # Capture fitting/selection arguments to verify exact label roles rather
        # than merely trusting the pipeline's function names or report metadata.
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(poisoned, "targets", side_effect=permitted), \
                 patch("confidence.fit_temperature", wraps=fit_temperature) as fits, \
                 patch("confidence.choose_threshold", wraps=choose_threshold) as choices:
                with redirect_stdout(io.StringIO()):
                    report, policy = run_analysis(poisoned, self.mlp, self.gcn, Path(directory), "cpu")
            self.assertEqual(len(fits.call_args_list), 2)
            self.assertEqual(len(choices.call_args_list), 2)
            for call in fits.call_args_list:
                np.testing.assert_array_equal(call.args[1], self.dataset.targets("calibration").numpy())
            for call in choices.call_args_list:
                np.testing.assert_array_equal(call.args[1], self.dataset.targets("policy").numpy())
            self.assertFalse(report["test_evaluated"])
            self.assertFalse(policy["test_evaluated"])
            self.assertEqual(policy["ece_bins"], ECE_BINS)
            self.assertEqual(json.loads((Path(directory) / "referral_policy.json").read_text()), policy)
            for name in ("MLP", "GCN"):
                result = report["models"][name]
                self.assertEqual(result["metrics"]["selection"]["before"]["accuracy"],
                                 result["metrics"]["selection"]["after"]["accuracy"])
                self.assertEqual(policy["models"][name]["threshold"], result["policy"]["threshold"])
            for stem in ("reliability_selection", "confidence_selection", "risk_coverage_policy", "failure_neighbors_selection"):
                for extension in ("png", "svg"):
                    self.assertGreater((Path(directory) / f"{stem}.{extension}").stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                run_analysis(self.dataset, self.mlp, self.gcn, Path(directory))


if __name__ == "__main__":
    unittest.main()
