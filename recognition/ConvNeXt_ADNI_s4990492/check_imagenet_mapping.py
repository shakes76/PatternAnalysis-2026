"""CPU-only random-weight mapping check; no download or training data."""
import torch
from torchvision.models import convnext_tiny
from modules import ConvNeXtTinyRGB
from imagenet_weight_mapping import load_backbone

if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.manual_seed(42)
    custom = ConvNeXtTinyRGB().eval()
    head = custom.classifier.weight.detach().clone()
    reference = convnext_tiny(weights=None, stochastic_depth_prob=0.0).eval()
    assert load_backbone(custom, reference.state_dict()) == 180
    assert torch.equal(custom.classifier.weight, head)
    reference.classifier[2] = torch.nn.Linear(768, 2)
    reference.classifier[2].load_state_dict(custom.classifier.state_dict())
    inputs = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(custom(inputs), reference(inputs), rtol=1e-4, atol=2e-6)
    del reference
    custom(torch.rand(2, 3, 64, 64)).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in custom.parameters())
    print('PASS: 180 tensors mapped; binary head preserved; reference forward parity; finite gradients.')
    print('Synthetic check only. Official weight loading still needs the ImageNet smoke run.')
