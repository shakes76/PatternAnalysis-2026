"""Numerical checks for evaluation metrics and validation-only referral selection."""
import unittest
import numpy as np
from reporting import extra_metrics, choose_referral_band, referral_report
from predict import choose_threshold


class ReportingTests(unittest.TestCase):
    def test_all_negative_roc_cutoff_is_finite(self):
        labels = np.array([1, 0])
        scores = np.array([.1, .9])
        threshold, _ = choose_threshold(labels, scores, "val_specificity80_max_sensitivity")
        self.assertTrue(np.isfinite(threshold))
        self.assertFalse((scores >= threshold).any())
        low, high = choose_referral_band(scores, threshold, .2)
        self.assertTrue(np.isfinite([low, high]).all())

    def test_both_class_metrics(self):
        metrics = extra_metrics(np.array([0, 0, 1, 1]), np.array([.1, .8, .3, .9]), .5)
        self.assertEqual(metrics["accuracy"], .5)
        self.assertEqual(metrics["macro_f1"], .5)
        self.assertEqual(metrics["benign_recall"], .5)
        self.assertAlmostEqual(metrics["brier_score"], (.01 + .64 + .49 + .01) / 4)

    def test_referral_uses_frozen_validation_band(self):
        low, high = choose_referral_band(np.array([.1, .4, .6, .9]), .5, .5)
        labels = np.array([1, 1, 0, 0])
        referred, result = referral_report(labels, np.array([.1, .45, .55, .9]), .5, low, high)
        np.testing.assert_array_equal(referred, [False, True, True, False])
        self.assertEqual(result["unreferred_melanoma_misses"], 1)
        self.assertEqual(result["referral_rate"], .5)
        self.assertEqual(result["automatic_accuracy"], 0)

    def test_referral_disabled_and_all_referred(self):
        scores = np.array([.1, .5, .9])
        for fraction, expected in ((0, 0), (1, 3)):
            low, high = choose_referral_band(scores, .5, fraction)
            mask, _ = referral_report(np.array([0, 1, 1]), scores, .5, low, high)
            self.assertEqual(mask.sum(), expected)


if __name__ == "__main__":
    unittest.main()
