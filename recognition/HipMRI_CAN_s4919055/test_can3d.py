"""Known initialization, whole-volume geometry and trainability checks."""

import unittest

import torch

from can3d import AdaptiveInstanceNorm3D, CAN3D, ConvBlock3D
from modules import segmentation_loss


class CAN3DTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_identity_context_kernel_and_expanded_channels(self):
        value = torch.randn(1,3,4,6,8)
        layer = ConvBlock3D(3,5,dilation=4,identity=True)
        result = layer.conv(value)
        torch.testing.assert_close(result[:,:3],value)
        self.assertEqual(int(torch.count_nonzero(result[:,3:])),0)

    def test_adaptive_normalization_branches_and_parameter_gradients(self):
        value = torch.randn(2,3,4,6,8,requires_grad=True)
        norm = AdaptiveInstanceNorm3D(3)
        torch.testing.assert_close(norm(value),value)
        with torch.no_grad():
            norm.a.zero_(); norm.b.fill_(1)
        result = norm(value)
        torch.testing.assert_close(result.mean((2,3,4)),torch.zeros(2,3),atol=1e-6,rtol=0)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertTrue(torch.isfinite(norm.a.grad) and torch.isfinite(norm.b.grad))

    def test_full_volume_geometry_loss_gradient_and_optimizer_step(self):
        model = CAN3D()
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        image = torch.randn(1,1,8,16,20)
        labels = torch.zeros(1,8,16,20,dtype=torch.long)
        labels[:,3:5,6:10,7:11]=5
        target = torch.nn.functional.one_hot(labels,6).permute(0,4,1,2,3).float()
        before = model.head.weight.detach().clone()
        logits = model(image)
        self.assertEqual(logits.shape,target.shape)
        loss = segmentation_loss('dice-ce')(logits,target)
        loss.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before,model.head.weight))
        self.assertEqual([block.conv.dilation[0] for block in model.context],[2,4,8,1])
        with self.assertRaisesRegex(ValueError,'even spatial'):
            model(torch.randn(1,1,7,16,20))


if __name__ == '__main__':
    unittest.main()
