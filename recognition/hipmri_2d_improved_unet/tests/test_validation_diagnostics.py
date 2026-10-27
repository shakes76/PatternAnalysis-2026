"""Hand-computed diagnostics on constructed tensors; no MRI or model training."""

import math
from pathlib import Path
import sys
import unittest

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from validation_diagnostics import class5_overlap, classwise_ce_statistics


class ValidationDiagnosticChecks(unittest.TestCase):
    @staticmethod
    def example():
        # Three valid pixels: P(true) is 0.5, 0.25 and 0.75. Fourth is padding.
        probabilities = torch.tensor([
            [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.75],
            [1 / 6] * 6,
        ], dtype=torch.float32)
        logits = probabilities.log().T.reshape(1, 6, 1, 4)
        target = torch.tensor([[[0, 5, 5, -100]]], dtype=torch.int64)
        return logits, target, target != -100

    def test_hand_computed_class_nll_and_weighted_ce(self):
        logits, target, valid = self.example()
        logits.requires_grad_()
        result = classwise_ce_statistics(logits, target, valid)
        self.assertEqual(result["support"], [1, 0, 0, 0, 0, 2])
        self.assertEqual(result["valid_pixels"], 3)
        self.assertAlmostEqual(result["nll_sum"][0], -math.log(0.5), places=6)
        self.assertAlmostEqual(result["nll_sum"][5], -math.log(0.25) - math.log(0.75), places=6)
        self.assertEqual(result["nll_sum"][1:5], [0.0] * 4)
        expected = (-math.log(0.5) - math.log(0.25) - math.log(0.75)) / 3
        self.assertAlmostEqual(result["ce_reconstructed"], expected, places=6)
        self.assertAlmostEqual(result["ce_shared"], expected, places=6)
        self.assertLess(result["ce_abs_difference"], 1e-6)
        self.assertIsNone(logits.grad)
        self.assertTrue(torch.is_grad_enabled())

    def test_aggregation_uses_pixel_support_instead_of_batch_means(self):
        logits, target, valid = self.example()
        first = classwise_ce_statistics(logits[..., :1], target[..., :1], valid[..., :1])
        second = classwise_ce_statistics(logits[..., 1:3], target[..., 1:3], valid[..., 1:3])
        whole = classwise_ce_statistics(logits, target, valid)
        self.assertEqual(
            [a + b for a, b in zip(first["support"], second["support"])],
            whole["support"],
        )
        for a, b, expected in zip(first["nll_sum"], second["nll_sum"], whole["nll_sum"]):
            self.assertAlmostEqual(a + b, expected, places=12)
        pooled = (first["nll_sum_total"] + second["nll_sum_total"]) / 3
        self.assertAlmostEqual(pooled, whole["ce_reconstructed"], places=12)
        unweighted_batch_mean = (first["ce_reconstructed"] + second["ce_reconstructed"]) / 2
        self.assertGreater(abs(unweighted_batch_mean - pooled), 0.02)

    def test_padding_logits_cannot_change_the_breakdown(self):
        logits, target, valid = self.example()
        original = classwise_ce_statistics(logits, target, valid)
        changed = logits.clone()
        changed[..., 3] = torch.tensor([1000.0, -1000.0, 0, 500.0, -500.0, 1.0]).reshape(1, 6, 1)
        modified = classwise_ce_statistics(changed, target, valid)
        self.assertEqual(original, modified)
        self.assertNotIn("pixel_nll_cpu", original)
        optional = classwise_ce_statistics(changed, target, valid, return_pixel_nll=True)
        pixel_nll = optional.pop("pixel_nll_cpu")
        self.assertEqual(optional, modified)
        self.assertEqual(pixel_nll.device.type, "cpu")
        self.assertEqual(pixel_nll.dtype, torch.float64)
        self.assertEqual(tuple(pixel_nll.shape), tuple(target.shape))
        self.assertFalse(pixel_nll.requires_grad)
        self.assertTrue(torch.equal(pixel_nll[~valid], torch.zeros(1, dtype=torch.float64)))
        self.assertAlmostEqual(pixel_nll[valid].sum().item(), original["nll_sum_total"], places=12)
        for class_id, total in enumerate(original["nll_sum"]):
            self.assertAlmostEqual(pixel_nll[(target == class_id) & valid].sum().item(), total, places=12)

    def test_hand_computed_id5_overlap_and_padding_exclusion(self):
        target = torch.tensor([[5, 5, 1, 5, 0, 1, -100]])
        prediction = torch.tensor([[5, 1, 5, 5, 0, 1, 5]])
        result = class5_overlap(prediction, target, target != -100)
        self.assertEqual(
            {key: result[key] for key in ("tp", "fp", "fn", "reference_pixels", "predicted_pixels", "valid_pixels")},
            {"tp": 2, "fp": 1, "fn": 1, "reference_pixels": 3, "predicted_pixels": 3, "valid_pixels": 6},
        )
        for key in ("dice", "precision", "recall"):
            self.assertAlmostEqual(result[key], 2 / 3)

    def test_empty_reference_or_prediction_policy(self):
        cases = [
            ([[0, 1]], [[0, 1]], None, None, None),
            ([[5, 1]], [[0, 1]], 0.0, 0.0, None),
            ([[0, 1]], [[5, 1]], 0.0, None, 0.0),
        ]
        for predicted, actual, dice, precision, recall in cases:
            with self.subTest(prediction=predicted, target=actual):
                target = torch.tensor(actual)
                result = class5_overlap(torch.tensor(predicted), target, torch.ones_like(target, dtype=torch.bool))
                self.assertEqual((result["dice"], result["precision"], result["recall"]), (dice, precision, recall))
        result = class5_overlap(torch.tensor([[5]]), torch.tensor([[-100]]), torch.tensor([[False]]))
        self.assertEqual(result["valid_pixels"], 0)
        self.assertIsNone(result["dice"])

    def test_malformed_class_counts_and_valid_ids_are_rejected(self):
        logits, target, valid = self.example()
        with self.assertRaises(ValueError):
            classwise_ce_statistics(logits, target, valid, num_classes=5)
        with self.assertRaises(TypeError):
            classwise_ce_statistics(logits, target, valid, return_pixel_nll="yes")
        with self.assertRaises(TypeError):
            class5_overlap(torch.tensor([[5.0]]), torch.tensor([[5]]), torch.tensor([[True]]))
        with self.assertRaises(ValueError):
            class5_overlap(torch.tensor([[6]]), torch.tensor([[5]]), torch.tensor([[True]]))


if __name__ == "__main__":
    unittest.main()
