"""Loss, label smoothing, Mixup and learning-rate schedule controls."""

import math
import unittest

import torch

from engine.scheduling import learning_rate
from utils.training_controls import ExecutionControls, make_criterion, mixup_batch


class TrainingControlTests(unittest.TestCase):
    def test_controls_round_trip_and_reject_unsupported_values(self):
        controls = ExecutionControls(context_slices=3, mixup_alpha=0.2, label_smoothing=0.1)
        self.assertEqual(ExecutionControls.from_dict(controls.to_dict()), controls)
        for field, value in (("context_slices", 2), ("loss", "cross_entropy"), ("precision", "amp_fp16"),
                             ("mixup_alpha", 1.5), ("label_smoothing", 1.0), ("drop_path", 1.0)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ExecutionControls(**{field: value})

    def test_weighted_and_smoothed_bce_match_hand_calculation(self):
        device = torch.device("cpu")
        logits, targets = torch.tensor([0.0]), torch.tensor([1.0])
        weighted = make_criterion(ExecutionControls(), pos_weight=2.0, device=device)
        self.assertAlmostEqual(weighted(logits, targets).item(), 2 * math.log(2), places=6)
        plain = make_criterion(ExecutionControls(loss="bce"), pos_weight=2.0, device=device)
        self.assertAlmostEqual(plain(logits, targets).item(), math.log(2), places=6)
        smoothed = make_criterion(ExecutionControls(loss="bce", label_smoothing=0.2), 1.0, device)
        logit = torch.tensor([2.0])
        expected = -(0.9 * math.log(torch.sigmoid(logit).item()) + 0.1 * math.log(1 - torch.sigmoid(logit).item()))
        self.assertAlmostEqual(smoothed(logit, targets).item(), expected, places=6)

    def test_mixup_mixes_within_the_batch_and_zero_alpha_is_identity(self):
        images = torch.arange(4.0).reshape(4, 1, 1, 1)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        same, labels_a, labels_b, lam = mixup_batch(images, labels, 0.0)
        self.assertIs(same, images)
        self.assertEqual(lam, 1.0)
        torch.manual_seed(0)
        mixed, labels_a, labels_b, lam = mixup_batch(images, labels, 0.2)
        self.assertTrue(0 <= lam <= 1)
        self.assertTrue(torch.equal(labels_a, labels))
        self.assertEqual(sorted(labels_b.tolist()), sorted(labels.tolist()))  # A permutation of the batch.

    def test_learning_rate_schedules(self):
        self.assertEqual(learning_rate(5, 10, 1e-3, "constant"), 1e-3)
        warm = [learning_rate(epoch, 10, 1e-3, "warmup_cosine", warmup=2) for epoch in range(1, 11)]
        self.assertAlmostEqual(warm[0], 5e-4)
        self.assertAlmostEqual(warm[1], 1e-3)
        self.assertAlmostEqual(warm[2], 1e-3)
        self.assertAlmostEqual(warm[-1], 1e-5)
        self.assertTrue(all(a >= b for a, b in zip(warm[1:], warm[2:])))


if __name__ == "__main__":
    unittest.main()
