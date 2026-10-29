"""Test disagreement rejection, shared errors and confidence group accounting."""
import unittest

import torch

from consensus_policy import (average_and_agreement, calibrated_average,
                              confidence_diagnostics, fit_rule, route)


class ConsensusPolicyTests(unittest.TestCase):
    def test_confident_disagreement_is_referred(self):
        members = torch.tensor([[[0., 0., 1., 0.]], [[0., 0., 0., 1.]]])
        p, agree, _ = calibrated_average(members, 1.)
        self.assertFalse(bool(agree[0]))
        self.assertFalse(bool(route(p, agree, [0.] * 4, True)[0]))
        self.assertTrue(bool(route(p, agree, [0.] * 4, False)[0]))

    def test_shared_confident_error_is_not_detected_by_unanimity(self):
        members = torch.tensor([[[.999, .0005, .0004, .0001]]] * 6)
        p, agree, _ = calibrated_average(members, 1.)
        accepted = route(p, agree, [.9] * 4, True)
        self.assertTrue(bool(accepted[0]))
        truth = torch.tensor([3])
        self.assertNotEqual(int(p.argmax(1)[0]), int(truth[0]))

    def test_averaging_is_member_permutation_invariant(self):
        members = torch.tensor([[[.7, .1, .1, .1], [.1, .7, .1, .1]],
                                [[.6, .2, .1, .1], [.1, .6, .2, .1]],
                                [[.8, .1, .05, .05], [.7, .1, .1, .1]]])
        first, agreement, _ = average_and_agreement(members)
        other, reverse_agreement, _ = average_and_agreement(members.flip(0))
        torch.testing.assert_close(first, other)
        self.assertTrue(torch.equal(agreement, reverse_agreement))
        torch.testing.assert_close(first.sum(1), torch.ones(2))
        self.assertEqual(agreement.tolist(), [True, False])

    def test_invalid_members_and_temperatures_are_rejected(self):
        with self.assertRaises(ValueError):
            average_and_agreement(torch.ones(1, 2, 4) / 4)
        invalid = torch.ones(2, 2, 4) / 4
        invalid[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            calibrated_average(invalid, 1.)
        for temperature in [0., -1., float("inf")]:
            with self.assertRaises(ValueError):
                calibrated_average(torch.ones(2, 2, 4) / 4, temperature)

    def test_temperature_preserves_mean_class_predictions(self):
        members = torch.tensor([[[.7, .1, .1, .1], [.1, .6, .1, .2]]] * 6)
        cool, _, _ = calibrated_average(members, .5)
        warm, _, _ = calibrated_average(members, 2.)
        self.assertTrue(torch.equal(cool.argmax(1), warm.argmax(1)))
        self.assertTrue(bool((cool.max(1).values > warm.max(1).values).all()))

    def test_fitting_excludes_ineligible_cases(self):
        p = torch.tensor([[.01, .01, .01, .97]] * 60)
        labels = torch.tensor([3] * 50 + [0] * 10)
        eligible = torch.tensor([True] * 50 + [False] * 10)
        thresholds, details = fit_rule(p, labels, eligible)
        self.assertEqual(details[3]["accepted"], 50)
        self.assertEqual(thresholds[3], 0.)
        self.assertEqual(int(route(p, eligible, thresholds, True).sum()), 50)

    def test_empty_eligible_group_rejects_all(self):
        p = torch.ones(4, 4) / 4
        thresholds, _ = fit_rule(p, torch.arange(4), torch.zeros(4, dtype=torch.bool))
        self.assertEqual(thresholds, [1.01] * 4)
        self.assertEqual(int(route(p, torch.zeros(4, dtype=torch.bool), thresholds, True).sum()), 0)

    def test_true_and_predicted_politician_counts_are_separate(self):
        p = torch.tensor([[.999, .0003, .0003, .0004], [.0003, .0003, .0004, .999], [.0003, .0003, .0004, .999]])
        truth = torch.tensor([3, 3, 0])
        result = confidence_diagnostics(p, truth)
        self.assertEqual(result["groups"]["predicted_politician"]["count"], 2)
        self.assertEqual(result["groups"]["predicted_politician"]["correct"], 1)
        self.assertEqual(result["groups"]["true_politician"]["errors_confidence_ge_099"], 1)
        self.assertEqual(result["missed_true_politicians"], 1)
        self.assertEqual(sum(row["count"] for row in result["politician_probability_bins"]), 3)


if __name__ == "__main__":
    unittest.main()
