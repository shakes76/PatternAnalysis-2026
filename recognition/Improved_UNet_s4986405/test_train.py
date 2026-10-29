"""Verify optimizer updates, validation isolation and epoch pixel accounting."""
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from train import run_epoch


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.head = nn.Conv2d(1, 6, kernel_size=1)
        self.calls = []
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.bias[2] = 5

    def forward(self, images):
        self.calls.append((self.training, torch.is_grad_enabled()))
        return self.head(self.norm(images))


class TrainingLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        images = torch.arange(36, dtype=torch.float32).reshape(3, 1, 3, 4) / 36
        self.labels = torch.arange(36).reshape(3, 3, 4) % 6
        targets = F.one_hot(self.labels, num_classes=6).permute(0, 3, 1, 2).float()
        self.loader = DataLoader(TensorDataset(images, targets), batch_size=2, shuffle=False)
        self.model = RecordingModel()

    def test_training_updates_and_batch_limit(self):
        before = [parameter.detach().clone() for parameter in self.model.parameters()]
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        result = run_epoch(self.model, self.loader, 'cpu', optimizer=optimizer, max_batches=1)
        self.assertEqual(result['batches'], 1)
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['pixels'], 24)
        self.assertEqual(result['confusion'].sum().item(), 24)
        self.assertEqual(self.model.calls, [(True, True)])
        self.assertTrue(any(
            not torch.equal(old, new.detach())
            for old, new in zip(before, self.model.parameters())
        ))
        self.assertTrue(torch.isfinite(torch.tensor(result['loss'])).item())

    def test_validation_preserves_state_and_pools_all_pixels(self):
        # Existing gradients must also survive validation unchanged.
        for parameter in self.model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        gradients = [parameter.grad.clone() for parameter in self.model.parameters()]
        result = run_epoch(self.model, self.loader, 'cpu')
        self.assertEqual(result['batches'], 2)
        self.assertEqual(result['samples'], 3)
        self.assertEqual(result['pixels'], 36)
        expected = torch.zeros(6, 6, dtype=torch.long)
        expected[:, 2] = 6  # Six true pixels per class; every prediction is class 2.
        self.assertTrue(torch.equal(result['confusion'], expected))
        self.assertEqual(self.model.calls, [(False, False), (False, False)])
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        for parameter, gradient in zip(self.model.parameters(), gradients):
            self.assertTrue(torch.equal(parameter.grad, gradient))

    def test_empty_loader_and_invalid_limit_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No samples'):
            run_epoch(self.model, [], 'cpu')
        for limit in (0, -1):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    run_epoch(self.model, self.loader, 'cpu', max_batches=limit)


if __name__ == '__main__':
    unittest.main()
