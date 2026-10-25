"""CPU tests for model behavior; run with unittest discovery from project root."""

import io
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import ImprovedUNet2D, UNet2D, build_model


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)

    def test_spatial_shape_and_finite_outputs_in_both_modes(self):
        for name in ("unet", "improved_unet"):
            model = build_model(name, in_channels=3, num_classes=4, base_channels=4)
            for training in (True, False):
                model.train(training)
                for height, width in ((32, 32), (35, 49), (64, 48)):
                    with self.subTest(model=name, training=training, shape=(height, width)):
                        with torch.no_grad():
                            result = model(torch.randn(1, 3, height, width))
                        self.assertEqual(result.shape, (1, 4, height, width))
                        self.assertTrue(torch.isfinite(result).all())

    def test_loss_reaches_every_encoder_decoder_and_segmentation_head(self):
        for name in ("unet", "improved_unet"):
            with self.subTest(model=name):
                model = build_model(name, base_channels=4)
                logits = model(torch.randn(2, 1, 33, 37))
                target = torch.randint(0, 2, (2, 33, 37))
                F.cross_entropy(logits, target).backward()
                groups = list(model.encoders) + list(model.decoders)
                groups += list(model.segmentation_heads) if name == "improved_unet" else [model.head]
                for index, group in enumerate(groups):
                    gradients = [parameter.grad for parameter in group.parameters()]
                    self.assertTrue(all(grad is not None for grad in gradients), index)
                    self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients), index)
                    self.assertGreater(sum(grad.abs().sum().item() for grad in gradients), 0.0, index)

    def test_checkpoint_round_trip_preserves_predictions(self):
        x = torch.randn(1, 1, 35, 41)
        for name in ("unet", "improved_unet"):
            with self.subTest(model=name):
                model = build_model(name, base_channels=4).eval()
                with torch.no_grad():
                    expected = model(x)
                checkpoint = io.BytesIO()
                torch.save(model.state_dict(), checkpoint)
                checkpoint.seek(0)
                restored = build_model(name, base_channels=4).eval()
                restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                with torch.no_grad():
                    actual = restored(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_eval_is_deterministic_and_train_uses_dropout(self):
        model = ImprovedUNet2D(base_channels=4)
        x = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            model.eval()
            torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)
            model.train()
            self.assertFalse(torch.equal(model(x), model(x)))

    def test_invalid_inputs_fail_clearly(self):
        for constructor in (UNet2D, ImprovedUNet2D):
            model = constructor(base_channels=4)
            for shape in ((1, 32, 32), (1, 2, 32, 32), (1, 1, 31, 48), (0, 1, 32, 32)):
                with self.subTest(model=constructor.__name__, shape=shape):
                    with self.assertRaises(ValueError):
                        model(torch.zeros(shape))
            with self.assertRaises(TypeError):
                model(torch.zeros(1, 1, 32, 32, dtype=torch.long))
            for kwargs in ({"in_channels": 0}, {"num_classes": -1}, {"base_channels": 0}):
                with self.assertRaises(ValueError):
                    constructor(**kwargs)
        with self.assertRaises(ValueError):
            build_model("unknown")


if __name__ == "__main__":
    unittest.main()
