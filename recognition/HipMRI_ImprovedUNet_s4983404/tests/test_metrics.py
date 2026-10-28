"""Metric conventions on hand-computed confusion matrices and logits."""
from pathlib import Path
import sys
import unittest
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import confusion_counts, scores, segmentation_loss


def one_hot(labels: torch.Tensor, classes: int = 6) -> torch.Tensor:
    """(B, H, W) int labels -> (B, classes, H, W) float targets."""
    return F.one_hot(labels.long(), classes).permute(0, 3, 1, 2).float()


class ScoreTests(unittest.TestCase):
    def test_hand_computed_confusion(self) -> None:
        confusion = torch.zeros(6, 6)
        confusion[0, 0] = 8   # background correct
        confusion[0, 1] = 2   # background misread as class 1
        confusion[1, 1] = 3   # class 1 correct
        result = scores(confusion)
        # class 0: 2*8/(8+10) = 0.888..., 8/(18-8) = 0.8
        self.assertAlmostEqual(result['dice'][0], 16 / 18)
        self.assertAlmostEqual(result['iou'][0], 0.8)
        # class 1: 2*3/(5+3) = 0.75, 3/(8-3) = 0.6
        self.assertAlmostEqual(result['dice'][1], 0.75)
        self.assertAlmostEqual(result['iou'][1], 0.6)
        # classes absent from both truth and prediction stay undefined
        self.assertEqual(result['dice'][2:], [None] * 4)
        self.assertEqual(result['iou'][2:], [None] * 4)

    def test_one_sided_empty_class_scores_zero(self) -> None:
        confusion = torch.zeros(6, 6)
        confusion[0, 0] = 4
        confusion[0, 2] = 1   # class 2 predicted but never present
        confusion[3, 0] = 2   # class 3 present but never predicted
        result = scores(confusion)
        self.assertEqual(result['dice'][2], 0.0)
        self.assertEqual(result['iou'][2], 0.0)
        self.assertEqual(result['dice'][3], 0.0)
        self.assertEqual(result['iou'][3], 0.0)
        self.assertIsNone(result['dice'][4])


class ConfusionTests(unittest.TestCase):
    def test_perfect_prediction_is_diagonal(self) -> None:
        labels = torch.tensor([[[0, 1], [1, 2]]])
        target = one_hot(labels)
        confusion = confusion_counts(target * 20, target)
        self.assertEqual(tuple(confusion.shape), (6, 6))
        self.assertAlmostEqual(float(confusion.diag().sum()), 4.0)
        self.assertAlmostEqual(float(confusion.sum() - confusion.diag().sum()), 0.0)

    def test_padding_slices_are_ignored(self) -> None:
        labelled = one_hot(torch.tensor([[[0, 1]]]))
        padding = torch.zeros(1, 6, 1, 2)     # padding carries no active class at all
        target = torch.cat([labelled, padding], dim=0)
        logits = target * 20
        logits[1, 5] = 20                     # would be a false positive in the padding
        confusion = confusion_counts(logits, target)
        self.assertAlmostEqual(float(confusion.sum()), 2.0)
        self.assertAlmostEqual(float(confusion[5].sum()), 0.0)


class LossTests(unittest.TestCase):
    def test_confident_correct_prediction_stays_near_zero(self) -> None:
        labels = torch.tensor([[[0, 1], [1, 2]]])
        target = one_hot(labels)
        # Absent classes score just under dice 1 because of the 1e-6 smoothing, so a
        # perfect prediction keeps a floor of roughly 5e-3 instead of reaching zero.
        self.assertLess(float(segmentation_loss(target * 20, target)), 0.01)

    def test_confident_wrong_prediction_is_penalised(self) -> None:
        labels = torch.tensor([[[0, 1], [1, 1]]])
        target = one_hot(labels)
        wrong = F.one_hot(torch.full_like(labels, 2), 6).permute(0, 3, 1, 2).float()
        # Classes absent from the target still score dice ~1, so the loss drops by
        # about 0.6 from the missing overlap alone; anything above 0.3 is a real penalty.
        self.assertGreater(float(segmentation_loss(wrong * 20, target)), 0.3)

    def test_unannotated_slice_does_not_change_the_loss(self) -> None:
        labels = torch.tensor([[[0, 1, 1, 0]]])
        target = one_hot(labels)
        logits = target * 4 + 0.3
        blurred = logits.clone()
        blurred[0, 1, 0, 0] += 0.6                         # make the loss non-trivial
        alone = segmentation_loss(blurred, target)
        padded = segmentation_loss(
            torch.cat([blurred, torch.randn(1, 6, 1, 4) * 9], dim=0),
            torch.cat([target, torch.zeros(1, 6, 1, 4)], dim=0))
        self.assertAlmostEqual(float(alone), float(padded), places=5)


if __name__ == '__main__':
    unittest.main()
