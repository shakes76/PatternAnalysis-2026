"""Check scan completeness, the AD label convention, and metric edge cases."""

import json
import math
import unittest

from metrics import aggregate_scans, binary_metrics


def slice_record(image_id, slice_index, probability, label=1, patient_id="001_S_0001"):
    """Build a small prediction record without loading any real patient data."""
    return {
        "patient_id": patient_id, "image_id": image_id, "slice_index": slice_index,
        "relative_path": f"images/{image_id}_{slice_index}.jpeg", "label": label,
        "probability": probability,
    }


class BinaryMetricsTests(unittest.TestCase):
    """Use hand-calculated results instead of duplicating metric algorithms."""

    def test_confusion_and_per_class_metrics(self):
        result = binary_metrics([0, 0, 0, 1, 1], [0.1, 0.2, 0.9, 0.4, 0.8])
        self.assertEqual(result["confusion_matrix"], [[2, 1], [1, 1]])
        self.assertEqual(result["n_samples"], 5)
        self.assertAlmostEqual(result["accuracy"], 0.6)
        self.assertAlmostEqual(result["balanced_accuracy"], (2 / 3 + 1 / 2) / 2)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 1 / 2) / 2)
        self.assertEqual(result["per_class"]["NC"]["support"], 3)
        self.assertEqual(result["per_class"]["AD"]["support"], 2)
        self.assertAlmostEqual(result["per_class"]["NC"]["precision"], 2 / 3)
        self.assertAlmostEqual(result["per_class"]["AD"]["recall"], 0.5)
        expected_loss = -math.log(0.9 * 0.8 * 0.1 * 0.4 * 0.8) / 5
        self.assertAlmostEqual(result["log_loss"], expected_loss)
        self.assertAlmostEqual(result["auroc"], 4 / 6)
        json.dumps(result, allow_nan=False)

    def test_auc_known_ranking_and_direction(self):
        self.assertAlmostEqual(binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])["auroc"], 0.75)
        self.assertEqual(binary_metrics([0, 1], [0.1, 0.9])["auroc"], 1.0)
        self.assertEqual(binary_metrics([0, 1], [0.9, 0.1])["auroc"], 0.0)

    def test_auc_ties_and_input_order(self):
        self.assertAlmostEqual(binary_metrics([0, 1, 0, 1], [0.1, 0.4, 0.4, 0.9])["auroc"], 0.875)
        self.assertAlmostEqual(binary_metrics([1, 0, 1, 0], [0.9, 0.4, 0.4, 0.1])["auroc"], 0.875)
        self.assertEqual(binary_metrics([0, 0, 1], [0.5, 0.5, 0.5])["auroc"], 0.5)

    def test_threshold_changes_decisions_but_not_auc_or_loss(self):
        usual = binary_metrics([0, 1], [0.5, 0.7])
        shifted = binary_metrics([0, 1], [0.5, 0.7], threshold=0.6)
        self.assertEqual(usual["confusion_matrix"], [[0, 1], [0, 1]])
        self.assertEqual(shifted["confusion_matrix"], [[1, 0], [0, 1]])
        self.assertEqual(usual["auroc"], shifted["auroc"])
        self.assertEqual(usual["log_loss"], shifted["log_loss"])

    def test_single_class_and_unpredicted_class_are_explicit(self):
        result = binary_metrics([0, 0], [0.1, 0.2])
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["balanced_accuracy"])
        self.assertEqual(result["macro_f1"], 0.5)
        self.assertEqual(result["per_class"]["AD"], {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
        all_negative = binary_metrics([0, 1], [0.1, 0.2])
        self.assertEqual(all_negative["per_class"]["AD"]["precision"], 0.0)
        self.assertEqual(all_negative["balanced_accuracy"], 0.5)
        json.dumps(result, allow_nan=False)

    def test_log_loss_handles_endpoints_without_nan(self):
        correct = binary_metrics([0, 1], [0.0, 1.0])
        incorrect = binary_metrics([0, 1], [1.0, 0.0])
        self.assertLess(correct["log_loss"], 1e-12)
        self.assertGreater(incorrect["log_loss"], 30)
        json.dumps(incorrect, allow_nan=False)

    def test_reject_invalid_labels_probabilities_and_lengths(self):
        for labels, probabilities in (([], []), ([0], []), ([2], [0.1]), (["1"], [0.1]),
                                      ([True], [0.1]), ([1.0], [0.1])):
            with self.subTest(labels=labels, probabilities=probabilities), self.assertRaises(ValueError):
                binary_metrics(labels, probabilities)
        for probability in (math.nan, math.inf, -0.01, 1.01, "0.5", True):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                binary_metrics([1], [probability])
        for threshold in (math.nan, -0.01, 1.01, "0.5"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                binary_metrics([1], [0.5], threshold=threshold)


class ScanAggregationTests(unittest.TestCase):
    """Reject incomplete or contradictory scans before reporting scan metrics."""

    def test_mean_probability_and_mixed_diagnosis_patient(self):
        rows = [slice_record("20", 79, 0.9), slice_record("10", 78, 0.1, label=0),
                slice_record("20", 78, 0.2), slice_record("10", 79, 0.2, label=0)]
        scans = aggregate_scans(rows, expected_slices=2)
        self.assertEqual([scan["image_id"] for scan in scans], ["10", "20"])
        self.assertEqual([scan["label"] for scan in scans], [0, 1])
        self.assertEqual(scans[0]["patient_id"], scans[1]["patient_id"])
        self.assertAlmostEqual(scans[0]["probability"], 0.15)
        self.assertAlmostEqual(scans[1]["probability"], 0.55)
        self.assertEqual([scan["prediction"] for scan in scans], [0, 1])
        self.assertEqual([scan["num_slices"] for scan in scans], [2, 2])

    def test_mean_probabilities_instead_of_majority_vote(self):
        rows = [slice_record("10", i, score) for i, score in enumerate((0.49, 0.49, 0.99))]
        self.assertEqual(aggregate_scans(rows, expected_slices=3)[0]["prediction"], 1)
        tie = [slice_record("10", 0, 0.2), slice_record("10", 1, 0.8)]
        self.assertEqual(aggregate_scans(tie, expected_slices=2)[0]["prediction"], 1)

    def test_default_requires_twenty_slices(self):
        rows = [slice_record("10", i, 0.3) for i in range(20)]
        self.assertEqual(aggregate_scans(rows)[0]["num_slices"], 20)
        for invalid in (rows[:-1], rows + [slice_record("10", 20, 0.3)], []):
            with self.subTest(count=len(invalid)), self.assertRaises(ValueError):
                aggregate_scans(invalid)

    def test_reject_duplicate_slice_identity_even_with_different_path(self):
        rows = [slice_record("10", 78, 0.1), slice_record("10", 78, 0.2)]
        rows[1]["relative_path"] = "other/10_78.jpeg"
        with self.assertRaisesRegex(ValueError, "Repeated slice index"):
            aggregate_scans(rows, expected_slices=2)

    def test_reject_duplicate_path_even_with_different_scan(self):
        rows = [slice_record("10", 78, 0.1), slice_record("20", 79, 0.2)]
        rows[1]["relative_path"] = rows[0]["relative_path"]
        with self.assertRaisesRegex(ValueError, "Repeated slice path"):
            aggregate_scans(rows, expected_slices=1)

    def test_reject_patient_and_label_conflicts_within_scan(self):
        for key, value in (("patient_id", "002_S_0002"), ("label", 0)):
            rows = [slice_record("10", 78, 0.1), slice_record("10", 79, 0.2)]
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Inconsistent"):
                aggregate_scans(rows, expected_slices=2)

    def test_reject_malformed_slice_records(self):
        for key, value in (("patient_id", ""), ("image_id", 10), ("relative_path", " "),
                           ("slice_index", "78"), ("slice_index", -1), ("label", 2),
                           ("probability", math.nan), ("probability", 1.1)):
            row = slice_record("10", 78, 0.5)
            row[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                aggregate_scans([row], expected_slices=1)
        with self.assertRaises(ValueError):
            aggregate_scans([{"image_id": "10"}], expected_slices=1)
        for count in (0, -1, True, 1.5):
            with self.subTest(expected_slices=count), self.assertRaises(ValueError):
                aggregate_scans([slice_record("10", 78, 0.5)], expected_slices=count)


if __name__ == "__main__":
    unittest.main()
