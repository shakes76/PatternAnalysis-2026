"""Shared model/pipeline integration on constructed tensors; no MRI reads."""

import math
from pathlib import Path
import sys
import unittest

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dataset import collate_padded, preprocess_pair
from losses import segmentation_cross_entropy
from modules import UNet, build_model
from train import run_epoch


class SharedModelPipelineChecks(unittest.TestCase):
    model_names = ("baseline_unet", "improved_unet")

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def make_batch():
        samples = []
        for height, width in ((17, 19), (32, 16)):
            image = torch.arange(height * width, dtype=torch.float32).reshape(
                1, height, width
            )
            target = torch.arange(height * width, dtype=torch.long).reshape(
                height, width
            ) % 6
            samples.append(preprocess_pair(image, target))
        # Both per-sample padding and extra batch padding are exercised.
        return collate_padded(samples)

    def test_both_models_train_through_the_shared_pipeline(self):
        for model_name in self.model_names:
            with self.subTest(model=model_name):
                torch.manual_seed(3710)
                model = build_model(model_name, base_channels=2).eval()
                batch = self.make_batch()
                self.assertEqual(tuple(batch["image"].shape), (2, 1, 32, 32))
                original_target = batch["target"].clone()
                original_valid = batch["valid_mask"].clone()
                before = {
                    name: value.detach().clone()
                    for name, value in model.named_parameters()
                }
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                observed = []

                def record_forward(module, inputs, output):
                    observed.append((
                        module.training, torch.is_grad_enabled(),
                        output.requires_grad, tuple(output.shape),
                    ))

                hook = model.register_forward_hook(record_forward)
                try:
                    with torch.no_grad():
                        result = run_epoch(
                            model, [batch], torch.device("cpu"), num_classes=6,
                            optimizer=optimizer,
                        )
                finally:
                    hook.remove()

                self.assertTrue(model.training)
                self.assertEqual(observed, [(True, True, True, (2, 6, 32, 32))])
                self.assertEqual(result["optimizer_steps"], 1)
                self.assertEqual(result["batches"], 1)
                self.assertEqual(result["samples"], 2)
                self.assertEqual(result["valid_pixels"], 835)
                self.assertTrue(math.isfinite(result["cross_entropy"]))
                self.assertGreater(result["gradient_norm_range"][0], 0)
                self.assertTrue(all(
                    math.isfinite(value) for value in result["gradient_norm_range"]
                ))
                gradients = [parameter.grad for parameter in model.parameters()]
                self.assertTrue(all(gradient is not None for gradient in gradients))
                self.assertTrue(all(
                    torch.isfinite(gradient).all().item() for gradient in gradients
                ))
                self.assertGreater(sum(
                    gradient.abs().sum().item() for gradient in gradients
                ), 0)
                self.assertTrue(any(
                    not torch.equal(before[name], value.detach())
                    for name, value in model.named_parameters()
                ))
                self.assertTrue(torch.equal(batch["target"], original_target))
                self.assertTrue(torch.equal(batch["valid_mask"], original_valid))

    def test_both_models_validate_without_gradients_or_state_changes(self):
        for model_name in self.model_names:
            with self.subTest(model=model_name):
                torch.manual_seed(3710)
                model = build_model(model_name, base_channels=2).eval()
                batch = self.make_batch()
                with torch.inference_mode():
                    expected_loss = segmentation_cross_entropy(
                        model(batch["image"]), batch["target"], batch["valid_mask"]
                    ).item()
                model.train()
                before = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
                observed = []

                def record_forward(module, inputs, output):
                    observed.append((
                        module.training, torch.is_grad_enabled(), output.requires_grad
                    ))

                hook = model.register_forward_hook(record_forward)
                try:
                    with torch.enable_grad():
                        result = run_epoch(
                            model, [batch], torch.device("cpu"), num_classes=6,
                        )
                finally:
                    hook.remove()

                self.assertFalse(model.training)
                self.assertEqual(observed, [(False, False, False)])
                self.assertEqual(result["optimizer_steps"], 0)
                self.assertEqual(result["valid_pixels"], 835)
                self.assertTrue(math.isfinite(result["cross_entropy"]))
                self.assertAlmostEqual(result["cross_entropy"], expected_loss, places=6)
                self.assertNotIn("gradient_norm_range", result)
                self.assertTrue(all(
                    parameter.grad is None for parameter in model.parameters()
                ))
                # state_dict includes parameters and every persistent buffer.
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(before[name], value), name)

    def test_existing_baseline_checkpoint_preserves_factory_logits(self):
        checkpoint_path = (
            PROJECT_DIR.parents[1] / "outputs" / "baseline_smoke_20261025" / "best.pt"
        )
        if not checkpoint_path.is_file():
            self.skipTest(f"Optional local baseline checkpoint absent: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["model"], "baseline_unet")
        factory_model = build_model("baseline_unet", **checkpoint["model_config"]).eval()
        direct_model = UNet(**checkpoint["model_config"]).eval()
        factory_model.load_state_dict(checkpoint["model_state"])
        direct_model.load_state_dict(checkpoint["model_state"])
        image = torch.linspace(-1, 1, 32 * 48).reshape(1, 1, 32, 48)
        with torch.inference_mode():
            factory_logits = factory_model(image)
            direct_logits = direct_model(image)
        torch.testing.assert_close(factory_logits, direct_logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
