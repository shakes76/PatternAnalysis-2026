"""Validate ConvNeXt computation at the real image size and its training behavior."""

import unittest

import torch
from torch import nn

from models import (
    MODEL_CHOICES, MODEL_NAMES, ConvNeXtBlock, ConvNeXtTiny, LayerNorm2d, SmallCNN,
    StochasticDepth, count_parameters, create_model, model_minimum_size,
)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        torch.manual_seed(3710)

    def test_convnext_native_resolution_forward_backward_and_architecture(self):
        model = ConvNeXtTiny().eval()
        self.assertEqual(count_parameters(model), 27_817_825)
        self.assertEqual(tuple(len(stage) for stage in model.stages), (3, 3, 9, 3))
        blocks = [block for stage in model.stages for block in stage]
        self.assertEqual(len(blocks), 18)
        for index, block in enumerate(blocks):
            self.assertEqual(block.depthwise.kernel_size, (7, 7))
            self.assertEqual(block.depthwise.groups, block.depthwise.in_channels)
            self.assertEqual(block.expand.out_features, 4 * block.expand.in_features)
            self.assertAlmostEqual(block.drop_path.probability, index * 0.1 / 17)
            self.assertTrue(torch.all(block.layer_scale == 1e-6).item())
        self.assertFalse(any(isinstance(layer, nn.modules.batchnorm._BatchNorm)
                             for layer in model.modules()))

        stage_shapes = []
        hooks = [stage.register_forward_hook(
            lambda _module, _inputs, output: stage_shapes.append(tuple(output.shape))
        ) for stage in model.stages]
        self.addCleanup(lambda: [hook.remove() for hook in hooks])
        images = torch.empty(1, 1, 240, 256).uniform_(-1, 1).requires_grad_(True)
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (1,))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertEqual(stage_shapes, [(1, 96, 60, 64), (1, 192, 30, 32),
                                       (1, 384, 15, 16), (1, 768, 7, 8)])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(1))
        loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
        for gradient in (images.grad, model.stem[0].weight.grad,
                         model.stages[0][0].depthwise.weight.grad,
                         model.classifier.weight.grad):
            self.assertGreater(torch.count_nonzero(gradient).item(), 0)

        # Evaluation consumes no dropout randomness and is repeatable after backward.
        with torch.inference_mode():
            state = torch.random.get_rng_state()
            first = model(images.detach())
            second = model(images.detach())
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_convnext_valid_boundaries_and_malformed_inputs(self):
        model = ConvNeXtTiny().eval()
        with torch.inference_mode():
            for shape in ((1, 1, 32, 32), (2, 1, 65, 71)):
                with self.subTest(shape=shape):
                    logits = model(torch.zeros(shape))
                    self.assertEqual(tuple(logits.shape), (shape[0],))
                    self.assertTrue(torch.isfinite(logits).all().item())
        invalid = [torch.zeros(1, 32, 32), torch.zeros(1, 3, 32, 32),
                   torch.zeros(1, 1, 31, 64), torch.zeros(1, 1, 64, 31)]
        for images in invalid:
            with self.subTest(shape=tuple(images.shape)):
                with self.assertRaisesRegex(ValueError, "Expected"):
                    model(images)

    def test_three_context_channels_change_only_the_stem(self):
        single, context = ConvNeXtTiny(1), ConvNeXtTiny(3)
        self.assertEqual(count_parameters(context) - count_parameters(single), 2 * 96 * 4 * 4)
        with torch.inference_mode():
            self.assertEqual(tuple(context.eval()(torch.zeros(2, 3, 240, 256)).shape), (2,))
            self.assertEqual(tuple(SmallCNN(3).eval()(torch.zeros(2, 3, 64, 64)).shape), (2,))

    def test_factories_keep_baseline_initialization_and_use_independent_models(self):
        self.assertEqual(MODEL_NAMES, ("small_cnn_v1", "convnext_lite_v1", "convnext_tiny_v1",
                                       "convnext_tiny_in1k_v1"))
        self.assertEqual(MODEL_CHOICES, {"cnn": "small_cnn_v1", "convnext_lite": "convnext_lite_v1",
                                         "convnext_tiny": "convnext_tiny_v1",
                                         "convnext_tiny_in1k": "convnext_tiny_in1k_v1"})
        torch.manual_seed(3710)
        old_baseline = SmallCNN()
        torch.manual_seed(3710)
        factory_baseline = create_model("small_cnn_v1")
        self.assertEqual(count_parameters(factory_baseline), 97_521)
        self.assertTrue(all(torch.equal(parameter, factory_baseline.state_dict()[name])
                            for name, parameter in old_baseline.state_dict().items()))
        first = create_model("convnext_tiny_v1")
        second = create_model("convnext_tiny_v1")
        self.assertIsNot(first, second)
        self.assertNotEqual(first.stem[0].weight.data_ptr(), second.stem[0].weight.data_ptr())
        self.assertFalse(torch.equal(first.stem[0].weight, second.stem[0].weight))
        self.assertEqual(model_minimum_size("small_cnn_v1"), 16)
        self.assertEqual(model_minimum_size("convnext_tiny_v1"), 32)
        for factory in (create_model, model_minimum_size):
            with self.assertRaisesRegex(ValueError, "Unsupported model name"):
                factory("unsupported_v1")

    def test_stochastic_depth_drops_whole_examples_only_during_training(self):
        layer = StochasticDepth(0.5)
        residual = torch.ones(128, 3, 4, 5)
        output = layer(residual)
        flattened = output.flatten(1)
        self.assertTrue(torch.all(flattened == flattened[:, :1]).item())
        self.assertEqual(set(output.unique().tolist()), {0.0, 2.0})
        layer.eval()
        state = torch.random.get_rng_state()
        self.assertTrue(torch.equal(layer(residual), residual))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(StochasticDepth(0.0)(residual), residual))
        for probability in (-0.01, 1.0, float("nan")):
            with self.assertRaises(ValueError):
                StochasticDepth(probability)

    def test_layer_scale_controls_residual_and_layer_norm_is_per_location(self):
        block = ConvNeXtBlock(8, 0.0)
        features = torch.randn(2, 8, 5, 7)
        with torch.no_grad():
            block.layer_scale.zero_()
            self.assertTrue(torch.equal(block(features), features))
            block.layer_scale.fill_(1.0)
            self.assertFalse(torch.equal(block(features), features))
        norm = LayerNorm2d(8)
        normalized = norm(features)
        self.assertTrue(torch.allclose(normalized.mean(1), torch.zeros(2, 5, 7), atol=1e-6))
        self.assertTrue(torch.allclose(normalized.var(1, unbiased=False),
                                       torch.ones(2, 5, 7), atol=2e-5))
        changed = features.clone()
        changed[1].mul_(100)
        self.assertTrue(torch.equal(norm(changed)[0], normalized[0]))


if __name__ == "__main__":
    unittest.main()
