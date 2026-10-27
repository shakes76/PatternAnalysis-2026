"""Constructed CPU checks of the baseline; no MRI or model-quality experiment."""

import io
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from modules import UNet


class BaselineModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_output_preserves_multiple_spatial_sizes(self):
        torch.manual_seed(3710)
        model = UNet(base_channels=2).eval()
        for shape in ((2, 1, 32, 32), (2, 1, 32, 48), (1, 1, 16, 16)):
            with self.subTest(shape=shape), torch.inference_mode():
                logits = model(torch.randn(shape))
                self.assertEqual(tuple(logits.shape), (shape[0], 6, shape[2], shape[3]))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_positive_channel_and_class_configuration(self):
        model = UNet(in_channels=3, num_classes=2, base_channels=2)
        with torch.inference_mode():
            logits = model(torch.randn(1, 3, 32, 48))
        self.assertEqual(tuple(logits.shape), (1, 2, 32, 48))

    def test_forward_backward_and_optimizer_step(self):
        torch.manual_seed(3710)
        model = UNet(base_channels=2).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        image = torch.randn(2, 1, 32, 48)
        target = torch.randint(0, 6, (2, 32, 48))

        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(image), target)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.0)
        optimizer.step()
        self.assertTrue(any(
            not torch.equal(old, parameter.detach())
            for old, parameter in zip(before, model.parameters())
        ))

    def test_state_dict_round_trip_preserves_logits(self):
        torch.manual_seed(10)
        model = UNet(base_channels=2).eval()
        image = torch.randn(1, 1, 32, 48)
        with torch.inference_mode():
            expected = model(image)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = UNet(base_channels=2).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.inference_mode():
            actual = restored(image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_input_shapes_are_rejected(self):
        model = UNet(base_channels=2)
        for shape in ((1, 32, 32), (1, 2, 32, 32), (1, 1, 15, 32),
                      (1, 1, 31, 32), (1, 1, 32, 33), (0, 1, 32, 32)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(shape))
        with self.assertRaises(TypeError):
            model(torch.zeros((1, 1, 32, 32), dtype=torch.int64))

    def test_invalid_configuration_is_rejected(self):
        for keyword in ("in_channels", "num_classes", "base_channels"):
            for value in (0, -1, 2.5, True):
                with self.subTest(keyword=keyword, value=value), self.assertRaises(ValueError):
                    UNet(**{keyword: value})


if __name__ == "__main__":
    unittest.main()
