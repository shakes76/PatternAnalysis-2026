"""Synthetic CPU checks of Improved U-Net, without MRI or quality evaluation."""

import io
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from modules import ImprovedUNet, ResidualContext, UNet, build_model


class ImprovedModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(3710)

    def test_output_and_heads_preserve_spatial_contract(self):
        model = ImprovedUNet(base_channels=2).eval()
        for shape in ((2, 1, 32, 32), (1, 1, 32, 48),
                      (1, 1, 16, 32), (1, 1, 256, 128)):
            with self.subTest(shape=shape), torch.inference_mode():
                heads = model.forward_heads(torch.randn(shape))
                for name, divisor in (("quarter", 4), ("half", 2), ("full", 1)):
                    self.assertEqual(
                        tuple(heads[name].shape),
                        (shape[0], 6, shape[2] // divisor, shape[3] // divisor),
                    )
                    self.assertTrue(torch.isfinite(heads[name]).all().item())
                logits = model(torch.randn(shape))
                self.assertEqual(tuple(logits.shape), (shape[0], 6, shape[2], shape[3]))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_configuration_supports_other_input_and_class_counts(self):
        model = ImprovedUNet(in_channels=3, num_classes=2, base_channels=4).eval()
        with torch.inference_mode():
            self.assertEqual(tuple(model(torch.randn(1, 3, 32, 48)).shape), (1, 2, 32, 48))

    def test_zero_residual_branch_is_identity(self):
        block = ResidualContext(4).train()
        for module in block.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        values = torch.randn(2, 4, 8, 12)
        torch.testing.assert_close(block(values), values, rtol=0, atol=0)

    def test_normalization_activation_dropout_and_scale_configuration(self):
        model = ImprovedUNet(base_channels=2)
        contexts = [getattr(model, f"context{level}") for level in range(1, 6)]
        self.assertEqual([block.first.conv.in_channels for block in contexts], [2, 4, 8, 16, 32])
        for level in range(2, 6):
            conv = getattr(model, f"down{level}").conv
            self.assertEqual(conv.stride, (2, 2))
            self.assertEqual(conv.kernel_size, (3, 3))
        for level in range(1, 5):
            block = getattr(model, f"localize{level}")
            spatial = block.spatial.conv
            channel = block.channel.conv
            self.assertEqual(spatial.kernel_size, (3, 3))
            self.assertEqual(spatial.in_channels, spatial.out_channels)
            self.assertEqual(channel.kernel_size, (1, 1))
            self.assertEqual(channel.in_channels, spatial.out_channels)
            self.assertEqual(channel.in_channels, 2 * channel.out_channels)
        norms = [module for module in model.modules() if isinstance(module, nn.InstanceNorm2d)]
        activations = [module for module in model.modules() if isinstance(module, nn.LeakyReLU)]
        dropout = [module for module in model.modules() if isinstance(module, nn.Dropout2d)]
        self.assertTrue(norms and activations)
        self.assertEqual(len(dropout), 5)
        self.assertTrue(all(module.affine and not module.track_running_stats for module in norms))
        self.assertTrue(all(module.negative_slope == 0.01 for module in activations))
        self.assertTrue(all(module.p == 0.3 for module in dropout))
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in model.modules()))

    def test_forward_matches_progressive_nearest_logit_fusion(self):
        model = ImprovedUNet(base_channels=2).eval()
        image = torch.randn(1, 1, 32, 48)
        with torch.inference_mode():
            heads = model.forward_heads(image)
            half = F.interpolate(heads["quarter"], scale_factor=2, mode="nearest") + heads["half"]
            expected = F.interpolate(half, scale_factor=2, mode="nearest") + heads["full"]
            actual = model(image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_finite_backward_updates_model_and_all_segmentation_heads(self):
        model = ImprovedUNet(base_channels=2).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.stem.weight.detach().clone()
        image = torch.randn(2, 1, 32, 48)
        target = torch.randint(0, 6, (2, 32, 48))
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(image), target)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))
        for head in (model.head_quarter, model.head_half, model.head_full):
            self.assertGreater(head.weight.grad.abs().sum().item(), 0.0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.stem.weight.detach()))
        self.assertTrue(all(torch.isfinite(parameter).all().item() for parameter in model.parameters()))

    def test_eval_checkpoint_round_trip_preserves_logits(self):
        model = ImprovedUNet(base_channels=2).eval()
        image = torch.randn(1, 1, 32, 48)
        with torch.inference_mode():
            expected = model(image)
        checkpoint = {
            "model": "improved_unet",
            "model_config": {"in_channels": 1, "num_classes": 6, "base_channels": 2},
            "model_state": model.state_dict(),
        }
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        restored = build_model(loaded["model"], **loaded["model_config"]).eval()
        restored.load_state_dict(loaded["model_state"])
        with torch.inference_mode():
            actual = restored(image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_factory_preserves_baseline_and_rejects_unknown_names(self):
        baseline = build_model("baseline_unet", base_channels=2)
        improved = build_model("improved_unet", base_channels=2)
        self.assertIsInstance(baseline, UNet)
        self.assertIsInstance(improved, ImprovedUNet)
        self.assertEqual(set(baseline.state_dict()), set(UNet(base_channels=2).state_dict()))
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model("attention_unet")

    def test_incompatible_architecture_checkpoint_is_rejected(self):
        baseline = UNet(base_channels=2)
        improved = ImprovedUNet(base_channels=2)
        with self.assertRaises(RuntimeError):
            improved.load_state_dict(baseline.state_dict())
        with self.assertRaises(RuntimeError):
            baseline.load_state_dict(improved.state_dict())
        with self.assertRaises(RuntimeError):
            ImprovedUNet(base_channels=4).load_state_dict(improved.state_dict())

    def test_invalid_inputs_and_configuration_are_rejected(self):
        model = ImprovedUNet(base_channels=2)
        for shape in ((1, 32, 32), (1, 2, 32, 32), (1, 1, 15, 32),
                      (1, 1, 31, 32), (1, 1, 32, 33), (0, 1, 32, 32)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(shape))
        with self.assertRaisesRegex(ValueError, "bottleneck spatial"):
            model(torch.zeros(1, 1, 16, 16))
        with self.assertRaises(TypeError):
            model(torch.zeros((1, 1, 32, 32), dtype=torch.int64))
        for keyword in ("in_channels", "num_classes", "base_channels"):
            for value in (0, -1, 2.5, True):
                with self.subTest(keyword=keyword, value=value), self.assertRaises(ValueError):
                    ImprovedUNet(**{keyword: value})


if __name__ == "__main__":
    unittest.main()
