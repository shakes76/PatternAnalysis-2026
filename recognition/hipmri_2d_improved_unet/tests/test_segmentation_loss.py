"""Constructed one-hot and valid-pixel CE checks; no MRI or model training."""

from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from losses import masked_one_hot, segmentation_cross_entropy


class SegmentationLossChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def example():
        target = torch.tensor([[[0, 1, -100], [2, 1, -100]]], dtype=torch.int64)
        valid = target != -100
        logits = torch.tensor([
            [[[2.0, -1.0, 3.0], [-0.5, 0.0, -2.0]],
             [[0.0, 2.0, -3.0], [0.2, 1.0, 2.0]],
             [[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]]]
        ], dtype=torch.float32)
        return logits, target, valid

    def test_one_hot_is_literal_class_encoding_and_zero_on_padding(self):
        _, target, valid = self.example()
        encoded = masked_one_hot(target, valid, num_classes=3)
        expected = torch.tensor([
            [[[1, 0, 0], [0, 0, 0]],
             [[0, 1, 0], [0, 1, 0]],
             [[0, 0, 0], [1, 0, 0]]]
        ], dtype=torch.float32)
        self.assertEqual(encoded.dtype, torch.float32)
        self.assertEqual(tuple(encoded.shape), (1, 3, 2, 3))
        self.assertTrue(torch.equal(encoded, expected))
        self.assertTrue(torch.equal(encoded.sum(dim=1), valid.float()))

    def test_one_hot_ce_value_and_gradient_match_builtin_ignore_ce(self):
        initial, target, valid = self.example()
        custom_logits = initial.clone().requires_grad_()
        builtin_logits = initial.clone().requires_grad_()
        custom = segmentation_cross_entropy(custom_logits, target, valid)
        builtin = F.cross_entropy(builtin_logits, target, ignore_index=-100)
        torch.testing.assert_close(custom, builtin)
        custom.backward()
        builtin.backward()
        torch.testing.assert_close(custom_logits.grad, builtin_logits.grad)
        self.assertTrue(torch.isfinite(custom_logits.grad).all().item())

    def test_padding_logits_cannot_change_loss_or_valid_gradients(self):
        initial, target, valid = self.example()
        original_logits = initial.clone().requires_grad_()
        changed = initial.clone()
        changed[:, 0][~valid] = 1000.0
        changed[:, 1][~valid] = -1000.0
        changed[:, 2][~valid] = -500.0
        changed.requires_grad_()
        original_loss = segmentation_cross_entropy(original_logits, target, valid)
        changed_loss = segmentation_cross_entropy(changed, target, valid)
        torch.testing.assert_close(changed_loss, original_loss)
        original_loss.backward()
        changed_loss.backward()
        torch.testing.assert_close(changed.grad, original_logits.grad)
        invalid = (~valid).unsqueeze(1).expand_as(changed.grad)
        self.assertTrue(torch.all(changed.grad[invalid] == 0).item())

    def test_loss_mean_weights_valid_pixels_instead_of_samples_or_padding(self):
        target = torch.tensor([
            [[0, -100], [-100, -100]],
            [[1, 2], [0, -100]],
        ], dtype=torch.int64)
        valid = target != -100
        logits = torch.linspace(-2.0, 2.0, 2 * 3 * 2 * 2).reshape(2, 3, 2, 2)
        custom = segmentation_cross_entropy(logits, target, valid)
        expected = F.cross_entropy(logits, target, ignore_index=-100)
        torch.testing.assert_close(custom, expected)

    def test_valid_class_range_and_padding_target_contract_are_enforced(self):
        _, initial_target, valid = self.example()
        for value in (-1, 3):
            with self.subTest(valid_class=value):
                target = initial_target.clone()
                target[0, 0, 0] = value
                with self.assertRaises(ValueError):
                    masked_one_hot(target, valid, num_classes=3)
        target = initial_target.clone()
        target[~valid] = 0
        with self.assertRaises(ValueError):
            masked_one_hot(target, valid, num_classes=3)

    def test_no_valid_pixels_is_rejected_instead_of_returning_nan(self):
        target = torch.full((1, 2, 3), -100, dtype=torch.int64)
        valid = torch.zeros_like(target, dtype=torch.bool)
        logits = torch.zeros((1, 3, 2, 3), dtype=torch.float32)
        with self.assertRaises(ValueError):
            segmentation_cross_entropy(logits, target, valid)


if __name__ == "__main__":
    unittest.main()
