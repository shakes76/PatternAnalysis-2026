"""Hand-checked confidence, calibration, referral, EMA and resource reporting."""

import json
import math
import unittest
from unittest import mock

import torch

from engine.ema import ModelEMA
from evaluation.calibration import (apply_platt, choose_referral_threshold, fit_platt,
                                    referral_decision)
from evaluation.metrics import aggregate_scans
from evaluation.reporting import aggregate_patients, confidence_metrics, prediction_report
from evaluation.resources import profile_inference
from utils.evaluation_artifacts import select_failure_cases
from utils.training_controls import ExecutionControls
from test_metrics import slice_record


class ConfidenceReportingTests(unittest.TestCase):
    def test_ece_brier_referral_and_confidence_ties(self):
        report = confidence_metrics([0, 1, 0, 1], [0.1, 0.8, 0.8, 0.4], bins=2, reject_threshold=0.8)
        self.assertAlmostEqual(report["ece_predicted_class"], 0.275)
        self.assertAlmostEqual(report["brier_ad"], 0.2625)
        self.assertEqual([p["accepted"] for p in report["risk_coverage"]], [0, 1, 3, 4])
        self.assertAlmostEqual(report["risk_coverage"][-1]["risk"], 0.5)
        self.assertEqual(report["referral"]["accepted"], 3)
        self.assertAlmostEqual(report["referral"]["accepted_accuracy"], 2 / 3)
        self.assertNotIn("referral", confidence_metrics([0, 1], [0.2, 0.9]))
        json.dumps(report, allow_nan=False)

    def test_patient_average_over_all_scans(self):
        rows = [slice_record("a", 1, 0.1, 0), slice_record("a", 2, 0.3, 0),
                slice_record("b", 1, 0.9, 0), slice_record("b", 2, 0.7, 0)]
        patients = aggregate_patients(rows)
        self.assertEqual(patients[0]["num_scans"], 2)
        self.assertAlmostEqual(patients[0]["probability"], 0.5)
        report, _ = prediction_report(rows, aggregate_scans(rows, expected_slices=2))
        self.assertEqual(report["accuracy_target"]["units"], ["slice", "scan", "patient"])
        self.assertIn("patient", report["confidence"])

    def test_failure_examples_include_both_types_and_never_invent_errors(self):
        rows = [slice_record("a", 1, 0.99, 0, "a"), slice_record("b", 1, 0.01, 1, "b"),
                slice_record("c", 1, 0.95, 1, "c")]
        failures = select_failure_cases(rows)
        self.assertEqual({r["error_type"] for r in failures}, {"FP", "FN"})
        self.assertEqual(select_failure_cases([rows[-1]]), [])


class CalibrationTests(unittest.TestCase):
    def test_platt_scaling_softens_overconfidence_and_corrects_class_bias(self):
        # Logits that over-call AD: shifted by +1 and right only 75% of the time.
        logits = [6.0, -4.0, 6.0, -4.0] * 5
        labels = [1, 0, 0, 1, 1, 0, 1, 0] * 2 + [1, 0, 1, 0]
        slope, intercept = fit_platt(logits, labels)
        self.assertTrue(0 < slope < 1)            # Over-confident logits are shrunk.
        self.assertLess(intercept, 0)             # The bias towards AD is moved back.
        rows = [{"logit": z, "probability": 1 / (1 + math.exp(-z))} for z in logits]
        for before, after in zip(rows, apply_platt(rows, slope, intercept)):
            self.assertLess(abs(after["probability"] - 0.5), abs(before["probability"] - 0.5))
        with self.assertRaises(ValueError):
            fit_platt([1.0, 2.0], [1, 1])         # Needs both classes.

    def test_referral_threshold_maximises_coverage_at_target_accuracy(self):
        labels = [1, 0, 1, 0, 1]
        probabilities = [0.95, 0.1, 0.6, 0.55, 0.8]
        chosen = choose_referral_threshold(labels, probabilities, 0.9)
        self.assertEqual((chosen["threshold"], chosen["coverage"]), (0.6, 0.8))
        self.assertIsNone(choose_referral_threshold([1, 0], [0.4, 0.6], 0.9)["threshold"])
        self.assertEqual(referral_decision(0.58, 0.6), "REFER")
        self.assertEqual(referral_decision(0.9, 0.6), "AD")
        self.assertEqual(referral_decision(0.1, 0.6), "NC")
        self.assertEqual(referral_decision(0.99, None), "REFER")


class EMATests(unittest.TestCase):
    def test_average_moves_toward_weights_and_never_trains(self):
        model = torch.nn.Linear(2, 1)
        ema = ModelEMA(model, decay=0.9)
        self.assertFalse(any(p.requires_grad for p in ema.module.parameters()))
        with torch.no_grad():
            model.weight.add_(1.0)
        before = ema.module.weight.clone()
        ema.update(model)
        decay = min(0.9, 2 / 11)  # Warm-up decay at the first update.
        self.assertTrue(torch.allclose(ema.module.weight, decay * before + (1 - decay) * model.weight))
        with self.assertRaises(ValueError):
            ModelEMA(model, 1.0)


class ResourceProfileTests(unittest.TestCase):
    def test_cpu_timing_and_training_mode_restoration(self):
        class Toy(torch.nn.Module):
            def forward(self, images):
                return images.mean(dim=(1, 2, 3))
        model = Toy().train()
        controls = ExecutionControls()
        with mock.patch("evaluation.resources.time.perf_counter", side_effect=[0, 0.002, 1, 1.004]):
            report = profile_inference(model, (32, 32), torch.device("cpu"), controls,
                                       batch_sizes=(2,), warmup=1, repeats=2, on_cpu=True)
        measured = report["batches"][0]
        self.assertAlmostEqual(measured["mean_batch_ms"], 3)
        self.assertAlmostEqual(measured["mean_ms_per_slice"], 1.5)
        self.assertTrue(model.training)
        report = profile_inference(model, (32, 32), torch.device("cpu"), controls)
        self.assertEqual(report["status"], "not_measured_cpu")


if __name__ == "__main__":
    unittest.main()
