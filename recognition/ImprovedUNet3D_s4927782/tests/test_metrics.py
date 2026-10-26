"""Check overlap arithmetic, absent-class policy, and patient-balanced means."""

import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:
    raise unittest.SkipTest("NumPy and PyTorch are required; metric runtime checks were not executed.")

UTILS_PATH = Path(__file__).resolve().parents[1] / "utils.py"
SPEC = importlib.util.spec_from_file_location("project_metric_utils", UTILS_PATH)
utils = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = utils
SPEC.loader.exec_module(utils)


class CaseMetricTests(unittest.TestCase):
    def test_exact_dice_iou_and_counts_with_numpy_or_torch(self):
        # Background: I=1, T=1, P=2; class 1: I=2, T=3, P=2;
        # class 2: I=1, T=2, P=2. These are hand-calculated, unequal overlaps.
        target = np.array([0, 1, 1, 1, 2, 2], dtype=np.int64)
        prediction = np.array([0, 1, 1, 2, 2, 0], dtype=np.int64)
        for converter in (np.asarray, torch.as_tensor):
            with self.subTest(input_type=converter.__name__):
                report = utils.compute_case_metrics(converter(prediction), converter(target), 3)
                classes = report["per_class"]
                self.assertEqual([item["class_index"] for item in classes], [0, 1, 2])
                for item, expected in zip(classes, [(2 / 3, 1 / 2, 1, 2, 1, 2),
                                                     (4 / 5, 2 / 3, 3, 2, 2, 3),
                                                     (1 / 2, 1 / 3, 2, 2, 1, 3)]):
                    dice, iou, target_count, predicted_count, intersection, union = expected
                    self.assertAlmostEqual(item["dice"], dice)
                    self.assertAlmostEqual(item["iou"], iou)
                    self.assertEqual(item["target_voxels"], target_count)
                    self.assertEqual(item["predicted_voxels"], predicted_count)
                    self.assertEqual(item["intersection"], intersection)
                    self.assertEqual(item["union"], union)
                self.assertAlmostEqual(report["foreground_mean_dice"], 0.65)
                self.assertAlmostEqual(report["foreground_mean_iou"], 0.5)

    def test_both_absent_is_undefined_and_excluded_from_foreground_mean(self):
        target = np.array([0, 1, 1], dtype=np.int64)
        report = utils.compute_case_metrics(target.copy(), target, 3)
        absent = report["per_class"][2]
        self.assertIsNone(absent["dice"])
        self.assertIsNone(absent["iou"])
        self.assertEqual(absent["target_voxels"], 0)
        self.assertEqual(absent["predicted_voxels"], 0)
        self.assertAlmostEqual(report["foreground_mean_dice"], 1.0)
        self.assertAlmostEqual(report["foreground_mean_iou"], 1.0)

    def test_false_positive_for_absent_target_is_zero_not_undefined(self):
        target = np.zeros(4, dtype=np.int64)
        prediction = np.array([0, 0, 0, 2], dtype=np.int64)
        report = utils.compute_case_metrics(prediction, target, 3)
        self.assertEqual(report["per_class"][2]["dice"], 0.0)
        self.assertEqual(report["per_class"][2]["iou"], 0.0)
        self.assertEqual(report["per_class"][2]["union"], 1)
        self.assertIsNone(report["per_class"][1]["dice"])
        self.assertEqual(report["foreground_mean_dice"], 0.0)
        self.assertEqual(report["foreground_mean_iou"], 0.0)
        self.assertAlmostEqual(report["per_class"][0]["dice"], 6 / 7)

    def test_invalid_shapes_and_class_values_raise_value_error(self):
        valid = np.array([0, 1], dtype=np.int64)
        invalid_pairs = [
            (valid.reshape(1, 2), valid),
            (np.array([-1, 1]), valid),
            (np.array([0, 3]), valid),
            (valid, np.array([-1, 1])),
            (valid, np.array([0, 3])),
            (np.array([0.0, 0.5]), valid),
        ]
        for prediction, target in invalid_pairs:
            with self.subTest(prediction=prediction.tolist(), target=target.tolist()):
                with self.assertRaises(ValueError):
                    utils.compute_case_metrics(prediction, target, 3)


class PatientAggregationTests(unittest.TestCase):
    @staticmethod
    def case(patient, visit, prediction, target, num_classes=3):
        return {
            "patient": patient,
            "visit": visit,
            **utils.compute_case_metrics(
                np.asarray(prediction, dtype=np.int64),
                np.asarray(target, dtype=np.int64),
                num_classes,
            ),
        }

    def test_visits_are_averaged_within_patient_before_patient_mean(self):
        records = [
            self.case("synthetic_A", "visit_0", [0, 1], [0, 1]),
            self.case("synthetic_A", "visit_1", [1, 0], [0, 1]),
            self.case("synthetic_B", "visit_0", [1, 0], [0, 1]),
        ]
        report = utils.aggregate_by_patient(records)
        self.assertEqual(report["num_cases"], 3)
        self.assertEqual(report["num_patients"], 2)
        self.assertAlmostEqual(report["foreground_mean_dice"], 0.25)
        self.assertAlmostEqual(report["foreground_mean_iou"], 0.25)
        self.assertNotAlmostEqual(report["foreground_mean_dice"], 1 / 3)
        for class_index in (0, 1):
            self.assertAlmostEqual(report["per_class"][class_index]["dice"], 0.25)
            self.assertAlmostEqual(report["per_class"][class_index]["iou"], 0.25)
        self.assertIsNone(report["per_class"][2]["dice"])
        patients = {item["patient"]: item for item in report["patients"]}
        self.assertAlmostEqual(patients["synthetic_A"]["foreground_mean_dice"], 0.5)
        self.assertEqual(patients["synthetic_B"]["foreground_mean_dice"], 0.0)

    def test_absent_class_is_excluded_within_patient_and_between_patients(self):
        records = [
            self.case("synthetic_A", "visit_0", [0, 0], [0, 0]),
            self.case("synthetic_A", "visit_1", [2, 2], [2, 2]),
            self.case("synthetic_B", "visit_0", [2, 2], [0, 0]),
        ]
        report = utils.aggregate_by_patient(records)
        patients = {item["patient"]: item for item in report["patients"]}
        self.assertEqual(patients["synthetic_A"]["per_class"][2]["dice"], 1.0)
        self.assertEqual(patients["synthetic_B"]["per_class"][2]["dice"], 0.0)
        self.assertAlmostEqual(report["per_class"][2]["dice"], 0.5)
        self.assertIsNone(report["per_class"][1]["dice"])

    def test_foreground_mean_is_patient_balanced_with_different_present_classes(self):
        records = [
            self.case("synthetic_A", "visit_0", [0, 1], [0, 1]),
            self.case("synthetic_B", "visit_0", [0, 0], [1, 2]),
        ]
        report = utils.aggregate_by_patient(records)
        # Patient A foreground mean=1, patient B=0: equally weighted mean=0.5.
        # Averaging global per-class means instead would incorrectly give 0.25.
        self.assertAlmostEqual(report["foreground_mean_dice"], 0.5)
        self.assertAlmostEqual(report["foreground_mean_iou"], 0.5)
        self.assertAlmostEqual(report["per_class"][1]["dice"], 0.5)
        self.assertEqual(report["per_class"][2]["dice"], 0.0)

    def test_empty_records_have_no_fabricated_scores(self):
        report = utils.aggregate_by_patient([])
        self.assertEqual(report["num_cases"], 0)
        self.assertEqual(report["num_patients"], 0)
        self.assertEqual(report["patients"], [])
        self.assertIsNone(report["foreground_mean_dice"])
        self.assertIsNone(report["foreground_mean_iou"])


if __name__ == "__main__":
    unittest.main()
