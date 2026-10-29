"""Analytical and gradient checks for the shared segmentation loss."""
import math
import unittest

import torch
from torch.nn import functional as F

from losses import dice_ce_loss
from modules import UNet2D


def encode(labels, classes):
    return F.one_hot(labels, num_classes=classes).permute(0, 3, 1, 2).float()


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.labels = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
        self.targets = encode(self.labels, 2)

    def test_uniform_prediction_matches_hand_calculation(self):
        smooth = 1e-5
        logits = torch.zeros(1, 2, 2, 2, dtype=torch.float64)
        actual = dice_ce_loss(logits, self.targets, smooth=smooth)
        # Each class has predicted mass 2, target mass 2 and intersection 1.
        expected = math.log(2) + 1 - (2 + smooth) / (4 + smooth)
        self.assertAlmostEqual(actual.item(), expected, places=12)

    def test_correct_predictions_have_lower_loss(self):
        correct = torch.where(self.targets.bool(), 20.0, -20.0)
        correct_loss = dice_ce_loss(correct, self.targets)
        wrong_loss = dice_ce_loss(-correct, self.targets)
        self.assertLess(correct_loss.item(), 1e-6)
        self.assertGreater(wrong_loss.item(), correct_loss.item())

    def test_large_wrong_logits_have_finite_gradients(self):
        logits = torch.where(self.targets.bool(), -1000.0, 1000.0).requires_grad_()
        loss = dice_ce_loss(logits, self.targets)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertGreater(torch.count_nonzero(logits.grad).item(), 0)

    def test_joint_channel_permutation_preserves_loss(self):
        logits = torch.tensor([[[[0.1, -0.2]], [[0.7, 0.3]], [[-0.1, 0.9]]]])
        targets = encode(torch.tensor([[[0, 2]]]), 3)
        order = [2, 0, 1]
        torch.testing.assert_close(
            dice_ce_loss(logits, targets),
            dice_ce_loss(logits[:, order], targets[:, order]),
        )

    def test_invalid_inputs_are_rejected(self):
        logits = torch.zeros(1, 2, 2, 2)
        cases = [
            ('shape', logits, self.targets[:, :, :1]),
            ('integer target', logits, self.targets.long()),
            ('integer logits', logits.long(), self.targets),
            ('soft target', logits, torch.full_like(self.targets, 0.5)),
            ('missing label', logits, torch.zeros_like(self.targets)),
            ('multiple labels', logits, torch.ones_like(self.targets)),
            ('non-finite logits', torch.full_like(logits, float('inf')), self.targets),
            ('non-finite target', logits, torch.full_like(self.targets, float('nan'))),
            ('empty', logits[:0], self.targets[:0]),
            ('one class', logits[:, :1], self.targets[:, :1]),
        ]
        for name, values, targets in cases:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    dice_ce_loss(values, targets)
        for smooth in (0, -1, float('nan'), float('inf')):
            with self.subTest(smooth=smooth):
                with self.assertRaises(ValueError):
                    dice_ce_loss(logits, self.targets, smooth=smooth)

    def test_unet_optimizer_step_updates_parameters(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3710)
            model = UNet2D(base_channels=4).cpu().train()
            images = torch.randn(1, 1, 31, 17)
            targets = encode(torch.randint(6, (1, 31, 17)), 6)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            optimizer.zero_grad(set_to_none=True)
            loss = dice_ce_loss(model(images), targets)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
            optimizer.step()
            self.assertTrue(any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, model.parameters())
            ))


if __name__ == '__main__':
    unittest.main()
