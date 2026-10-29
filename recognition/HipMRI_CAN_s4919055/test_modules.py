"""Numerical baseline checks; passing is not segmentation-performance evidence."""

import unittest
import math

import torch
import torch.nn.functional as F

from modules import UNet2D, SoftDiceLoss, DiceCrossEntropyLoss


class BaselineTests(unittest.TestCase):
    def test_expected_shape_and_finite_gradients(self):
        torch.manual_seed(42)
        model = UNet2D()
        logits = model(torch.randn(2, 1, 32, 48))
        self.assertEqual(tuple(logits.shape), (2, 6, 32, 48))
        labels = torch.randint(0, 6, (2, 32, 48))
        target = F.one_hot(labels, 6).permute(0, 3, 1, 2).float()
        loss = SoftDiceLoss()(logits, target)
        loss.backward()
        self.assertTrue(0 < loss.item() < 1)
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)

    def test_dice_against_hand_calculation(self):
        # Uniform p=.5, equal class support: each Dice=.5 (up to smoothing).
        logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
        labels = torch.tensor([[[0, 1], [0, 1]]])
        target = F.one_hot(labels, 2).permute(0, 3, 1, 2).float()
        self.assertAlmostEqual(SoftDiceLoss()(logits, target).item(), .5, places=5)
        correct = (target * 2 - 1) * 20
        wrong = -correct
        self.assertLess(SoftDiceLoss()(correct, target).item(), 1e-5)
        self.assertGreater(SoftDiceLoss()(wrong, target).item(), .999)

    def test_empty_foreground_and_3d_targets_stay_finite(self):
        logits = torch.randn(1, 6, 4, 8, 8, requires_grad=True)
        target = F.one_hot(torch.zeros(1, 4, 8, 8, dtype=torch.long), 6).permute(0, 4, 1, 2, 3).float()
        loss = SoftDiceLoss(include_background=False)(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_shape_errors_fail_clearly(self):
        with self.assertRaises(ValueError): UNet2D()(torch.zeros(1, 1, 31, 32))
        with self.assertRaises(ValueError): SoftDiceLoss()(torch.zeros(1, 6, 2, 2), torch.zeros(1, 1, 2, 2))

    def test_dice_ce_matches_hand_calculation(self):
        logits = torch.zeros(1, 2, 2, 2)
        target = F.one_hot(torch.tensor([[[0, 1], [0, 1]]]), 2).permute(0, 3, 1, 2).float()
        self.assertAlmostEqual(DiceCrossEntropyLoss()(logits, target).item(), .5+math.log(2), places=5)
        self.assertLess(DiceCrossEntropyLoss()((target*2-1)*20, target).item(), 1e-5)

    def test_ce_supplies_gradient_for_confidently_missed_class(self):
        # One true-prostate pixel, but the model confidently assigns background.
        # Demonstrates a possible Dice saturation mode, not a diagnosis of real weights.
        logits = torch.tensor([20., -20., -20., -20., -20., -20.]).reshape(1, 6, 1, 1).requires_grad_()
        target = F.one_hot(torch.tensor([[[5]]]), 6).permute(0, 3, 1, 2).float()
        dice_gradient = torch.autograd.grad(SoftDiceLoss()(logits, target), logits)[0]
        combined_gradient = torch.autograd.grad(DiceCrossEntropyLoss()(logits, target), logits)[0]
        self.assertLess(abs(dice_gradient[0, 5, 0, 0].item()), 1e-7)
        self.assertLess(combined_gradient[0, 5, 0, 0].item(), -.5)
        self.assertTrue(torch.isfinite(combined_gradient).all())

    def test_dice_ce_3d_empty_foreground_is_finite(self):
        logits = torch.randn(1, 6, 2, 4, 4, requires_grad=True)
        target = F.one_hot(torch.zeros(1, 2, 4, 4, dtype=torch.long), 6).permute(0, 4, 1, 2, 3).float()
        loss = DiceCrossEntropyLoss()(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
