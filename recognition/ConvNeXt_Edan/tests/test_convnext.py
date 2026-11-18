"""Tests for the native PyTorch ConvNeXt-Tiny implementation."""

import unittest
import torch
import torch.nn as nn

from modules import (
    ConvNeXtTiny,
    ConvNeXtBlock,
    create_model,
    count_parameters,
)


class TestNativeConvNeXt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.model = create_model("convnext", pretrained=False)
        cls.model.eval()

    def test_native_architecture(self):
        self.assertIsInstance(self.model.model, ConvNeXtTiny)
        self.assertEqual(
            count_parameters(self.model), 27_821_666
        )

        blocks = [
            m for m in self.model.modules()
            if isinstance(m, ConvNeXtBlock)
        ]

        self.assertEqual(len(blocks), 18)

    def test_hierarchical_feature_shapes(self):
        backbone = self.model.model
        x = torch.randn(1, 3, 224, 224)

        with torch.inference_mode():
            x = backbone.stem(x)
            self.assertEqual(
                tuple(x.shape), (1, 96, 56, 56)
            )

            expected = [
                (1, 96, 56, 56),
                (1, 192, 28, 28),
                (1, 384, 14, 14),
                (1, 768, 7, 7),
            ]

            for stage, shape in zip(
                backbone.stages, expected
            ):
                x = stage(x)
                self.assertEqual(tuple(x.shape), shape)

    def test_depthwise_convolutions(self):
        for block in self.model.modules():
            if isinstance(block, ConvNeXtBlock):
                conv = block.conv_dw
                self.assertIsInstance(conv, nn.Conv2d)
                self.assertEqual(conv.kernel_size, (7, 7))
                self.assertEqual(conv.groups, conv.in_channels)

    def test_finite_classification_logits(self):
        with torch.inference_mode():
            logits = self.model(
                torch.randn(2, 3, 224, 224)
            )

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all())

    def test_backward_pass(self):
        self.model.zero_grad(set_to_none=True)

        inputs = torch.randn(2, 3, 64, 64)
        labels = torch.tensor([0, 1])

        logits = self.model(inputs)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))

        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(
                torch.isfinite(parameter.grad).all(), name
            )

        self.model.zero_grad(set_to_none=True)


if __name__ == "__main__":
    unittest.main()
