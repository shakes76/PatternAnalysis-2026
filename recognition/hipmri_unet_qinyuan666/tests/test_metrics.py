"""Metric contract checks, including absent-class and gradient edge cases."""

import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import boundary_f1, confusion_matrix, scores_from_confusion, segmentation_loss


class SegmentationMetricsTests(unittest.TestCase):
    def test_perfect_prediction_and_absent_class(self):
        target = torch.tensor([[[0, 1], [1, 0]]])
        counts = confusion_matrix(target, target, 3)
        self.assertEqual(counts.dtype, torch.int64)
        self.assertEqual(counts.tolist(), [[2, 0, 0], [0, 2, 0], [0, 0, 0]])
        scores = scores_from_confusion(counts)
        self.assertEqual(scores["per_class"][1]["dice"], 1.0)
        self.assertEqual(scores["per_class"][1]["iou"], 1.0)
        self.assertIsNone(scores["per_class"][2]["dice"])
        self.assertIsNone(scores["per_class"][2]["iou"])
        self.assertEqual(scores["macro_foreground_dice"], 1.0)
        json.dumps(scores, allow_nan=False)

    def test_no_overlap_and_empty_foreground_are_distinct(self):
        target = torch.tensor([[[0, 1], [1, 0]]])
        scores = scores_from_confusion(confusion_matrix(1 - target, target, 2))
        self.assertEqual(scores["macro_foreground_dice"], 0.0)
        empty = torch.zeros_like(target)
        scores = scores_from_confusion(confusion_matrix(empty, empty, 2))
        self.assertIsNone(scores["macro_foreground_dice"])
        self.assertIsNone(scores["per_class"][1]["dice"])
        scores = scores_from_confusion(confusion_matrix(target, empty, 2))
        self.assertEqual(scores["per_class"][1]["dice"], 0.0)
        self.assertEqual(scores["per_class"][1]["support_pixels"], 0)

    def test_multiclass_orientation_and_logits(self):
        target = torch.tensor([[[0, 0, 1], [1, 2, 2]]])
        prediction = torch.tensor([[[0, 1, 1], [2, 2, 2]]])
        logits = torch.nn.functional.one_hot(prediction, 3).permute(0, 3, 1, 2).float() * 10
        counts = confusion_matrix(logits, target, 3)
        self.assertEqual(counts.tolist(), [[1, 1, 0], [0, 1, 1], [0, 0, 2]])
        self.assertTrue(torch.equal(counts, confusion_matrix(prediction[0], target[0], 3)))
        self.assertTrue(torch.equal(counts, confusion_matrix(logits[0], target[0], 3)))
        scores = scores_from_confusion(counts)
        self.assertAlmostEqual(scores["per_class"][1]["dice"], 0.5)
        self.assertAlmostEqual(scores["per_class"][1]["iou"], 1 / 3)
        self.assertAlmostEqual(scores["per_class"][2]["dice"], 0.8)
        self.assertAlmostEqual(scores["macro_foreground_dice"], 0.65)

    def test_loss_is_scalar_and_has_finite_nonzero_gradients(self):
        torch.manual_seed(11)
        logits = torch.randn(2, 3, 4, 5, requires_grad=True)
        targets = torch.arange(40).reshape(2, 4, 5) % 3
        loss = segmentation_loss(logits, targets)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_loss_perfect_vs_wrong_and_empty_foreground_penalty(self):
        targets = torch.tensor([[[0, 1], [1, 0]]])
        logits = torch.nn.functional.one_hot(targets, 2).permute(0, 3, 1, 2).float() * 30
        self.assertLess(float(segmentation_loss(logits, targets)), 1e-5)
        self.assertGreater(float(segmentation_loss(-logits, targets)), 20)
        empty = torch.zeros(1, 2, 2, dtype=torch.long)
        uncertain = torch.zeros(1, 2, 2, 2, requires_grad=True)
        loss = segmentation_loss(uncertain, empty)
        ce = torch.nn.functional.cross_entropy(uncertain, empty)
        self.assertGreater(float(loss - ce), 0.99)
        loss.backward()
        self.assertTrue(torch.isfinite(uncertain.grad).all())
        self.assertTrue((uncertain.grad[:, 1] > 0).all())

    def test_bad_shapes_labels_and_nonfinite_inputs_raise(self):
        target = torch.zeros(1, 2, 2, dtype=torch.long)
        logits = torch.zeros(1, 2, 2, 2)
        for bad_target in (target - 1, target + 2):
            with self.subTest(target=bad_target.tolist()):
                with self.assertRaises(ValueError):
                    segmentation_loss(logits, bad_target)
                with self.assertRaises(ValueError):
                    confusion_matrix(logits, bad_target, 2)
        with self.assertRaises(TypeError):
            segmentation_loss(logits, target.float())
        with self.assertRaises(ValueError):
            segmentation_loss(logits, target[:, :, :1])
        with self.assertRaises(ValueError):
            segmentation_loss(logits[:, :1], target)
        with self.assertRaises(ValueError):
            confusion_matrix(target[:, :, :1], target, 2)
        with self.assertRaises(ValueError):
            confusion_matrix(logits, target, 3)
        logits[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            segmentation_loss(logits, target)
        with self.assertRaises(ValueError):
            confusion_matrix(logits, target, 2)
        with self.assertRaises(ValueError):
            scores_from_confusion(torch.tensor([[1, 0], [0, -1]]))


class BoundaryMetricTests(unittest.TestCase):
    def test_perfect_boundary_and_empty_policy(self):
        mask = torch.ones(5, 5, dtype=torch.bool)
        empty = torch.zeros_like(mask)
        self.assertEqual(boundary_f1(mask, mask), 1.0)
        self.assertEqual(boundary_f1(mask, empty), 0.0)
        self.assertIsNone(boundary_f1(empty, empty))
        self.assertEqual(boundary_f1(torch.stack((mask, empty)),
                                     torch.stack((mask, empty))), 1.0)

    def test_tolerance_and_nonoverlap(self):
        left = torch.zeros(9, 9, dtype=torch.long)
        right = torch.zeros_like(left)
        left[4, 3], right[4, 5] = 1, 1
        self.assertEqual(boundary_f1(left, right, tolerance=0), 0.0)
        self.assertEqual(boundary_f1(left, right, tolerance=1), 0.0)
        self.assertEqual(boundary_f1(left, right, tolerance=2), 1.0)

    def test_invalid_binary_mask_and_tolerance(self):
        mask = torch.zeros(4, 4, dtype=torch.long)
        with self.assertRaises(ValueError):
            boundary_f1(mask + 2, mask)
        with self.assertRaises(ValueError):
            boundary_f1(mask, mask, tolerance=-1)


if __name__ == "__main__":
    unittest.main()
