"""Regression tests using hand-calculated segmentation metrics."""

import unittest

import torch

from metrics import confusion_matrix, dice_iou_from_confusion


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.target = torch.tensor([[0, 0, 0], [1, 1, 2]])
        self.prediction = torch.tensor([[0, 1, 1], [1, 2, 2]])

    def test_perfect_predictions(self):
        labels = torch.arange(6, dtype=torch.long)
        scores = dice_iou_from_confusion(confusion_matrix(labels, labels))
        for metric in ("dice", "iou"):
            torch.testing.assert_close(
                scores[metric], torch.ones(6, dtype=torch.float64)
            )

    def test_matrix_orientation_and_hand_calculation(self):
        matrix = confusion_matrix(self.prediction, self.target, num_classes=4)
        expected = torch.tensor([
            [1, 2, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ])
        self.assertTrue(torch.equal(matrix, expected))

        scores = dice_iou_from_confusion(matrix)
        expected_scores = {
            "dice": [1/2, 2/5, 2/3, float("nan")],
            "iou": [1/3, 1/4, 1/2, float("nan")],
        }
        for metric, values in expected_scores.items():
            torch.testing.assert_close(
                scores[metric],
                torch.tensor(values, dtype=torch.float64),
                equal_nan=True,
            )

    def test_missed_false_positive_and_absent_classes(self):
        matrix = confusion_matrix(
            torch.tensor([1, 1]), torch.tensor([0, 0]), num_classes=3
        )
        scores = dice_iou_from_confusion(matrix)
        for metric in ("dice", "iou"):
            torch.testing.assert_close(
                scores[metric],
                torch.tensor([0, 0, float("nan")], dtype=torch.float64),
                equal_nan=True,
            )

    def test_batch_accumulation(self):
        complete = confusion_matrix(self.prediction, self.target)
        accumulated = (
            confusion_matrix(self.prediction[:1], self.target[:1])
            + confusion_matrix(self.prediction[1:], self.target[1:])
        )
        self.assertTrue(torch.equal(accumulated, complete))

    def test_invalid_label_inputs(self):
        valid = torch.tensor([0], dtype=torch.long)
        cases = [
            ("float labels", valid.float(), valid),
            ("negative labels", torch.tensor([-1]), valid),
            ("out-of-range labels", torch.tensor([6]), valid),
            ("shape mismatch", torch.tensor([0, 1]), valid),
            ("empty inputs", valid[:0], valid[:0]),
        ]
        for name, prediction, target in cases:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    confusion_matrix(prediction, target)


if __name__ == "__main__":
    unittest.main()
