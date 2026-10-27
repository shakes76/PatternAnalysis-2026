"""Hand-computable metric checks on constructed class IDs, not MRI experiments."""

import json
from pathlib import Path
import sys
import unittest

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from metrics import SegmentationMetrics


class SegmentationMetricChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def example():
        target = torch.tensor([[0, 0, 0], [1, 1, 2]], dtype=torch.int64)
        prediction = torch.tensor([[0, 1, 0], [1, 2, 1]], dtype=torch.int64)
        return prediction, target, torch.ones_like(target, dtype=torch.bool)

    def test_hand_computed_confusion_dice_iou_and_absent_class(self):
        metric = SegmentationMetrics(num_classes=4)
        metric.update(*self.example())
        result = metric.compute()

        self.assertEqual(result["confusion_matrix"], [
            [2, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]
        ])
        for actual, expected in zip(result["dice_per_class"][:3], [0.8, 0.4, 0.0]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(result["iou_per_class"][:3], [2 / 3, 0.25, 0.0]):
            self.assertAlmostEqual(actual, expected)
        self.assertIsNone(result["dice_per_class"][3])
        self.assertIsNone(result["iou_per_class"][3])
        self.assertAlmostEqual(result["mean_dice_excluding_class0"], 0.2)
        self.assertAlmostEqual(result["mean_iou_excluding_class0"], 0.125)
        self.assertAlmostEqual(result["pixel_accuracy"], 0.5)
        self.assertEqual(result["valid_pixels"], 6)
        self.assertEqual(metric.confusion_matrix.dtype, torch.int64)
        self.assertEqual(metric.confusion_matrix.device.type, "cpu")
        # Undefined scores must serialize as null, not a nonstandard NaN.
        json.dumps(result, allow_nan=False)

    def test_accumulation_matches_one_update_including_batch_shape(self):
        prediction, target, valid = self.example()
        once = SegmentationMetrics(3)
        once.update(prediction.reshape(2, 1, 3), target.reshape(2, 1, 3), valid.reshape(2, 1, 3))
        split = SegmentationMetrics(3)
        split.update(prediction[:1], target[:1], valid[:1])
        split.update(prediction[1:], target[1:], valid[1:])
        self.assertEqual(split.compute(), once.compute())
        # Repeated compute does not reset or otherwise alter the accumulated split.
        self.assertEqual(split.compute(), once.compute())

    def test_padding_and_ignore_index_exclude_arbitrary_predictions(self):
        target = torch.tensor([[[0, 1, -100], [2, -100, -1]]], dtype=torch.int64)
        prediction = torch.tensor([[[0, 1, 999], [2, -999, 888]]], dtype=torch.int64)
        valid = torch.tensor([[[True, True, True], [True, False, False]]])
        metric = SegmentationMetrics(3)
        metric.update(prediction, target, valid)
        result = metric.compute()
        self.assertEqual(result["confusion_matrix"], torch.eye(3, dtype=torch.int64).tolist())
        self.assertEqual(result["valid_pixels"], 3)
        self.assertEqual(result["dice_per_class"], [1.0, 1.0, 1.0])
        self.assertEqual(result["pixel_accuracy"], 1.0)

        changed = prediction.clone()
        changed[~valid | (target == -100)] = -5000
        other = SegmentationMetrics(3)
        other.update(changed, target, valid)
        self.assertEqual(other.compute(), result)

    def test_predicted_only_class_scores_zero_and_enters_macro(self):
        metric = SegmentationMetrics(3)
        target = torch.tensor([[0, 0]], dtype=torch.int64)
        prediction = torch.tensor([[0, 2]], dtype=torch.int64)
        metric.update(prediction, target, torch.ones_like(target, dtype=torch.bool))
        result = metric.compute()
        self.assertIsNone(result["dice_per_class"][1])
        self.assertEqual(result["dice_per_class"][2], 0.0)
        self.assertEqual(result["iou_per_class"][2], 0.0)
        self.assertEqual(result["mean_dice_excluding_class0"], 0.0)
        self.assertEqual(result["mean_iou_excluding_class0"], 0.0)

    def test_no_valid_pixels_has_undefined_scores_and_zero_counts(self):
        metric = SegmentationMetrics(3)
        prediction = torch.full((1, 2, 3), 999, dtype=torch.int64)
        target = torch.full((1, 2, 3), -100, dtype=torch.int64)
        metric.update(prediction, target, torch.zeros_like(target, dtype=torch.bool))
        result = metric.compute()
        self.assertEqual(result["valid_pixels"], 0)
        self.assertEqual(result["dice_per_class"], [None, None, None])
        self.assertEqual(result["iou_per_class"], [None, None, None])
        self.assertIsNone(result["mean_dice_excluding_class0"])
        self.assertIsNone(result["mean_iou_excluding_class0"])
        self.assertIsNone(result["pixel_accuracy"])
        self.assertEqual(result["confusion_matrix"], [[0, 0, 0]] * 3)

    def test_invalid_class_ids_at_valid_pixels_are_rejected_atomically(self):
        for bad_value in (-1, 3):
            for field in ("prediction", "target"):
                with self.subTest(bad_value=bad_value, field=field):
                    prediction = torch.zeros((1, 1), dtype=torch.int64)
                    target = torch.zeros_like(prediction)
                    if field == "prediction":
                        prediction[0, 0] = bad_value
                    else:
                        target[0, 0] = bad_value
                    metric = SegmentationMetrics(3)
                    with self.assertRaises(ValueError):
                        metric.update(prediction, target, torch.ones_like(target, dtype=torch.bool))
                    self.assertEqual(metric.compute()["valid_pixels"], 0)

    def test_invalid_shapes_and_dtypes_are_rejected(self):
        metric = SegmentationMetrics(3)
        prediction, target, valid = self.example()
        with self.assertRaises(ValueError):
            metric.update(prediction[0], target, valid)
        with self.assertRaises(ValueError):
            metric.update(prediction.reshape(-1), target.reshape(-1), valid.reshape(-1))
        with self.assertRaises(TypeError):
            metric.update(prediction.float(), target, valid)
        with self.assertRaises(TypeError):
            metric.update(prediction, target, valid.long())
        for bad_classes in (0, -1, 1.5, True):
            with self.subTest(num_classes=bad_classes):
                with self.assertRaises(ValueError):
                    SegmentationMetrics(bad_classes)

    def test_unsigned_target_does_not_wrap_negative_ignore_id(self):
        metric = SegmentationMetrics(3)
        prediction = torch.zeros((1, 1), dtype=torch.int64)
        target = torch.tensor([[156]], dtype=torch.uint8)
        with self.assertRaises(ValueError):
            metric.update(prediction, target, torch.ones_like(target, dtype=torch.bool))
        self.assertEqual(metric.compute()["valid_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
