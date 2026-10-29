"""Check the 2D residual model, supervision paths and baseline compatibility."""
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from losses import dice_ce_loss
from modules import ImprovedUNet2D, ResidualContext2D, UNet2D


class ImprovedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(3710)

    def test_default_model_preserves_hipmri_dimensions(self):
        model = ImprovedUNet2D().eval()
        with torch.no_grad():
            output = model(torch.randn(1, 1, 256, 128))
        self.assertEqual(tuple(output.shape), (1, 6, 256, 128))
        self.assertTrue(torch.isfinite(output).all().item())
        print('ImprovedUNet2D default parameter count:', sum(p.numel() for p in model.parameters()))

    def test_odd_dimensions_and_configurable_channels(self):
        model = ImprovedUNet2D(in_channels=3, num_classes=4, base_channels=4, dropout=0).eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 65, 33))
        self.assertEqual(tuple(output.shape), (2, 4, 65, 33))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_residual_identity_and_input_gradient(self):
        block = ResidualContext2D(3, dropout=0)
        with torch.no_grad():
            for layer in block.modules():
                if isinstance(layer, nn.Conv2d):
                    layer.weight.zero_()
        values = torch.randn(2, 3, 17, 9, requires_grad=True)
        output = block(values)
        torch.testing.assert_close(output, values, rtol=0, atol=0)
        output.sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values), rtol=0, atol=0)

    def test_all_three_segmentation_heads_contribute_raw_logits(self):
        model = ImprovedUNet2D(base_channels=4).eval()
        with torch.no_grad():
            for index, head in enumerate(model.seg_heads):
                head.weight.zero_()
                head.bias.zero_()
                head.bias[index] = index + 1
            output = model(torch.randn(1, 1, 64, 32))
        expected = torch.tensor([1., 2., 3., 0., 0., 0.]).view(1, 6, 1, 1).expand_as(output)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_shared_loss_backward_and_optimizer_update(self):
        model = ImprovedUNet2D(base_channels=4, dropout=0.3).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        images = torch.randn(2, 1, 64, 32, requires_grad=True)
        labels = torch.randint(6, (2, 64, 32))
        targets = F.one_hot(labels, num_classes=6).permute(0, 3, 1, 2).float()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer.zero_grad(set_to_none=True)
        loss = dice_ce_loss(model(images), targets)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(images.grad).all().item())
        self.assertGreater(torch.count_nonzero(images.grad).item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        for head in model.seg_heads:
            self.assertGreater(torch.count_nonzero(head.bias.grad).item(), 0)
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, new.detach()) for old, new in zip(before, model.parameters())))

    def test_evaluation_is_repeatable_without_consuming_dropout_rng(self):
        model = ImprovedUNet2D(base_channels=4, dropout=0.3).eval()
        images = torch.randn(1, 1, 64, 32)
        state = torch.get_rng_state()
        with torch.no_grad():
            first = model(images)
            second = model(images)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_invalid_inputs_and_configuration_rejected(self):
        model = ImprovedUNet2D(base_channels=4)
        for shape in ((1, 32, 32), (1, 2, 32, 32), (1, 1, 16, 16)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    model(torch.zeros(shape))
        for kwargs in ({'in_channels': 0}, {'num_classes': 0}, {'base_channels': 0},
                       {'base_channels': 1.5}, {'dropout': -0.1}, {'dropout': 1.0}, {'dropout': float('nan')}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ImprovedUNet2D(**kwargs)

    def test_existing_baseline_still_loads_and_preserves_odd_shape(self):
        model = UNet2D(base_channels=4).eval()
        clone = UNet2D(base_channels=4).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        images = torch.randn(1, 1, 31, 17)
        with torch.no_grad():
            expected = model(images)
            recovered = clone(images)
        self.assertEqual(tuple(recovered.shape), (1, 6, 31, 17))
        torch.testing.assert_close(recovered, expected, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in UNet2D().parameters()), 1962422)


if __name__ == '__main__':
    unittest.main()
