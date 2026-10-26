"""Torch-dependent checks of shapes, supervision, and target validation."""

import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    raise unittest.SkipTest("PyTorch is required; model runtime checks were not executed.")

MODEL_PATH = Path(__file__).resolve().parents[1] / "modules.py"
SPEC = importlib.util.spec_from_file_location("project_models", MODEL_PATH)
models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(models)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_2d_output_and_odd_size_recovery(self):
        model = models.UNet2D(num_classes=4, base_channels=2)
        output = model(torch.randn(1, 1, 31, 35))
        self.assertEqual(tuple(output["logits"].shape), (1, 4, 31, 35))
        self.assertEqual(output["aux_logits"], [])
        loss = models.segmentation_loss(output, torch.randint(4, (1, 31, 35)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss).item()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_3d_outputs_recover_odd_sizes_and_keep_native_aux_sizes(self):
        model = models.ImprovedUNet3D(num_classes=4, base_channels=2)
        output = model(torch.randn(1, 1, 17, 19, 21))
        self.assertEqual(tuple(output["logits"].shape), (1, 4, 17, 19, 21))
        self.assertEqual(
            [tuple(head.shape) for head in output["aux_logits"]],
            [(1, 4, 8, 9, 10), (1, 4, 4, 4, 5)],
        )

    def test_3d_loss_reaches_every_auxiliary_head_with_finite_gradients(self):
        model = models.ImprovedUNet3D(num_classes=3, base_channels=2)
        output = model(torch.randn(1, 1, 16, 16, 16))
        for head in output["aux_logits"]:
            head.retain_grad()
        loss = models.segmentation_loss(output, torch.randint(3, (1, 16, 16, 16)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss).item()))
        for head in output["aux_logits"]:
            self.assertIsNotNone(head.grad)
            self.assertTrue(bool(torch.isfinite(head.grad).all().item()))
            self.assertGreater(float(head.grad.abs().sum().item()), 0)
        for auxiliary_head in model.aux_heads:
            self.assertIsNotNone(auxiliary_head.weight.grad)
            self.assertGreater(float(auxiliary_head.weight.grad.abs().sum().item()), 0)
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all().item()))

    def test_invalid_targets_raise_clear_errors(self):
        output = {"logits": torch.randn(1, 3, 8, 8), "aux_logits": []}
        invalid = [
            (torch.zeros(1, 8, 8), "integer class IDs"),
            (torch.full((1, 8, 8), 3, dtype=torch.long), "class IDs must be"),
            (torch.full((1, 8, 8), -1, dtype=torch.long), "class IDs must be"),
            (torch.zeros(2, 8, 8, dtype=torch.long), "match the logits batch"),
            (torch.zeros(1, 3, 8, 8, dtype=torch.long), "target must have shape"),
        ]
        for target, message in invalid:
            with self.subTest(message=message, shape=tuple(target.shape)):
                with self.assertRaisesRegex(ValueError, message):
                    models.segmentation_loss(output, target)

    def test_missing_auxiliary_weight_is_rejected(self):
        output = models.ImprovedUNet3D(num_classes=3, base_channels=2)(
            torch.randn(1, 1, 8, 8, 8)
        )
        with self.assertRaisesRegex(ValueError, "one positive, finite aux weight"):
            models.segmentation_loss(output, torch.zeros(1, 8, 8, 8, dtype=torch.long), aux_weights=(0.5,))

    def test_weighted_ce_matches_torch_reference_and_changes_class_emphasis(self):
        logits = torch.tensor([[[[2., 2.], [2., 2.]],
                                [[0., 0.], [0., 0.]],
                                [[-1., -1.], [-1., -1.]]]])
        target = torch.tensor([[[0, 0], [1, 2]]])
        weights = torch.tensor([0.2, 1.0, 3.0])
        output = {"logits": logits, "aux_logits": []}
        actual = models.segmentation_loss(output, target, dice_weight=0, class_weights=weights)
        expected = torch.nn.functional.cross_entropy(logits, target, weight=weights)
        unweighted = models.segmentation_loss(output, target, dice_weight=0)
        self.assertAlmostEqual(float(actual.item()), float(expected.item()), places=6)
        self.assertGreater(abs(float(actual.item()) - float(unweighted.item())), 0.1)

    def test_class_weights_apply_to_every_3d_head(self):
        logits = torch.randn(1, 3, 8, 8, 8)
        auxiliary = [torch.randn(1, 3, 4, 4, 4), torch.randn(1, 3, 2, 2, 2)]
        target = torch.randint(3, (1, 8, 8, 8))
        # Floating weights may differ in dtype; each head must use its own dtype.
        weights = torch.tensor([0.2, 1.0, 3.0], dtype=torch.float64)
        actual = models.segmentation_loss(
            {"logits": logits, "aux_logits": auxiliary},
            target, dice_weight=0, class_weights=weights,
        )
        expected = torch.nn.functional.cross_entropy(logits, target, weight=weights.float())
        for head, coefficient in zip(auxiliary, (0.5, 0.25)):
            resized_target = torch.nn.functional.interpolate(
                target.unsqueeze(1).float(), size=head.shape[2:], mode="nearest"
            ).squeeze(1).long()
            expected = expected + coefficient * torch.nn.functional.cross_entropy(
                head, resized_target, weight=weights.float()
            )
        expected = expected / 1.75
        self.assertAlmostEqual(float(actual.item()), float(expected.item()), places=6)

    def test_invalid_class_weights_raise_clear_errors(self):
        output = {"logits": torch.randn(1, 3, 8, 8), "aux_logits": []}
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        invalid = [
            ([1.0, 1.0, 1.0], "floating-point torch tensor"),
            (torch.ones(2), "one entry per class"),
            (torch.ones(1, 3), "must be 1D"),
            (torch.ones(3, dtype=torch.long), "floating-point torch tensor"),
            (torch.tensor([1.0, float("nan"), 1.0]), "finite values"),
            (torch.tensor([1.0, float("inf"), 1.0]), "finite values"),
            (torch.tensor([1.0, -1.0, 1.0]), "nonnegative and not all zero"),
            (torch.zeros(3), "nonnegative and not all zero"),
            (torch.ones(3, device="meta"), "same device"),
        ]
        for weights, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    models.segmentation_loss(output, target, class_weights=weights)


if __name__ == "__main__":
    unittest.main()
