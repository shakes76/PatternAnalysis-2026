"""Check narrow feasible intervals, whole ties, weights and label-free audits."""
import math
import unittest

import torch

from exact_policy import exact_decisions, fit_exact_class_thresholds, random_audit_mask
from refinement_study import training_class_weights
from validation_study import fit_class_thresholds


def politician_probabilities(confidences, dtype=torch.float32):
    confidence = torch.tensor(confidences, dtype=dtype)
    other = (1 - confidence) / 3
    return torch.stack((other, other, other, confidence), dim=1)


class ExactPolicyTests(unittest.TestCase):
    def test_finds_feasible_interval_above_point99(self):
        p = politician_probabilities([.9997] * 50 + [.995] * 10)
        y = torch.tensor([3] * 50 + [0] * 10)
        coarse, _ = fit_class_thresholds(p, y)
        fine, details = fit_exact_class_thresholds(p, y)
        self.assertEqual(coarse[3], 1.01)
        self.assertGreater(fine[3], .99)
        self.assertLess(fine[3], 1.)
        self.assertEqual(details[3]["accepted"], 50)
        self.assertEqual(int(exact_decisions(p, fine).sum()), 50)

    def test_confidence_ties_cannot_be_partially_accepted(self):
        p = politician_probabilities([.9997] * 51)
        y = torch.tensor([3] * 50 + [0])
        fine, details = fit_exact_class_thresholds(p, y, target=1.)
        self.assertEqual(fine[3], 1.01)
        self.assertEqual(details[3]["accepted"], 0)

    def test_all_accept_tie_uses_zero_and_small_groups_refer(self):
        p = politician_probabilities([.8] * 60, dtype=torch.float64)
        fine, _ = fit_exact_class_thresholds(p, torch.full((60,), 3))
        self.assertEqual(fine, [1.01, 1.01, 1.01, 0.])
        fewer, _ = fit_exact_class_thresholds(p[:49], torch.full((49,), 3))
        self.assertEqual(fewer, [1.01] * 4)

    def test_malformed_values_and_labels_are_rejected(self):
        p = politician_probabilities([.8] * 60)
        with self.assertRaises(ValueError):
            fit_exact_class_thresholds(p, torch.full((60,), 4))
        with self.assertRaises(ValueError):
            exact_decisions(p * 2, [.5] * 4)
        with self.assertRaises(ValueError):
            exact_decisions(p, [.5, .5, .5, float("nan")])

    def test_class_weights_use_training_labels_only(self):
        y = torch.tensor([0, 0, 0, 0, 1, 1, 2, 3, 0, 0])
        mask = torch.tensor([True] * 8 + [False] * 2)
        counts, weights = training_class_weights(y, mask)
        self.assertEqual(counts.tolist(), [4, 2, 1, 1])
        torch.testing.assert_close(weights, torch.tensor([.5, 1., 2., 2.]))
        changed = y.clone()
        changed[~mask] = 3
        torch.testing.assert_close(training_class_weights(changed, mask)[1], weights)
        with self.assertRaises(ValueError):
            training_class_weights(torch.zeros(10, dtype=torch.long), mask)

    def test_audits_are_reproducible_subsets_without_labels(self):
        accepted = torch.tensor([True] * 65 + [False] * 10)
        first = random_audit_mask(accepted, .05)
        self.assertEqual(int(first.sum()), math.ceil(65 * .05))
        self.assertFalse(bool((first & ~accepted).any()))
        self.assertTrue(torch.equal(first, random_audit_mask(accepted, .05)))
        self.assertEqual(int(random_audit_mask(accepted, 0.).sum()), 0)


if __name__ == "__main__":
    unittest.main()
