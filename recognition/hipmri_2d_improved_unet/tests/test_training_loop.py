"""Constructed training/validation branch checks; no MRI experiments."""

import math
from pathlib import Path
import sys
import unittest

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dataset import collate_padded, preprocess_pair
from modules import UNet
from train import run_epoch


class TrainingLoopChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def make_batches():
        def sample(height, width):
            image = torch.arange(height * width, dtype=torch.float32).reshape(
                1, height, width
            )
            target = torch.arange(height * width, dtype=torch.long).reshape(
                height, width
            ) % 6
            return preprocess_pair(image, target)

        # Two mixed-size samples, then a smaller final batch. Only original
        # pixels count: 17*19 + 16*32 + 16*16 = 1091.
        return [
            collate_padded([sample(17, 19), sample(16, 32)]),
            collate_padded([sample(16, 16)]),
        ]

    def test_training_updates_parameters_with_finite_gradients(self):
        torch.manual_seed(3710)
        model = UNet(in_channels=1, num_classes=6, base_channels=2)
        model.eval()
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        batches = self.make_batches()
        original_targets = [batch["target"].clone() for batch in batches]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        observed = []

        def record_forward(module, inputs, output):
            observed.append((module.training, torch.is_grad_enabled(), output.requires_grad))

        hook = model.register_forward_hook(record_forward)
        try:
            # run_epoch must explicitly enable the training branch, even if
            # its caller happened to be inside a no_grad context.
            with torch.no_grad():
                result = run_epoch(
                    model, batches, torch.device("cpu"), num_classes=6,
                    optimizer=optimizer,
                )
        finally:
            hook.remove()

        self.assertTrue(model.training)
        self.assertEqual(observed, [(True, True, True), (True, True, True)])
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["batches"], 2)
        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["valid_pixels"], 1091)
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), 1091)
        self.assertTrue(math.isfinite(result["cross_entropy"]))
        self.assertGreater(result["gradient_norm_range"][0], 0)
        self.assertTrue(all(math.isfinite(value) for value in result["gradient_norm_range"]))
        self.assertTrue(any(
            not torch.equal(before[name], value.detach())
            for name, value in model.named_parameters()
        ))
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))
        self.assertTrue(any(gradient.abs().sum().item() > 0 for gradient in gradients))
        self.assertTrue(optimizer.state)
        self.assertTrue(all(state["step"].item() == 2 for state in optimizer.state.values()))
        for batch, target in zip(batches, original_targets):
            self.assertTrue(torch.equal(batch["target"], target))

    def test_validation_disables_gradients_and_preserves_parameters(self):
        torch.manual_seed(3710)
        model = UNet(in_channels=1, num_classes=6, base_channels=2)
        model.train()
        for parameter in model.parameters():
            parameter.grad = None
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        batches = self.make_batches()
        original_targets = [batch["target"].clone() for batch in batches]
        observed = []

        def record_forward(module, inputs, output):
            observed.append((module.training, torch.is_grad_enabled(), output.requires_grad))

        hook = model.register_forward_hook(record_forward)
        try:
            with torch.enable_grad():
                result = run_epoch(
                    model, batches, torch.device("cpu"), num_classes=6,
                )
        finally:
            hook.remove()

        self.assertFalse(model.training)
        self.assertEqual(observed, [(False, False, False), (False, False, False)])
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(result["batches"], 2)
        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["valid_pixels"], 1091)
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), 1091)
        self.assertTrue(math.isfinite(result["cross_entropy"]))
        self.assertNotIn("gradient_norm_range", result)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        for batch, target in zip(batches, original_targets):
            self.assertTrue(torch.equal(batch["target"], target))


if __name__ == "__main__":
    unittest.main()
