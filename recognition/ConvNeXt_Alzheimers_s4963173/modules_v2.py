"""Optional RGB-pretrained adapter for the student's PyTorch ConvNeXt.

The original modules.py is never changed. Scratch experiments instantiate its
ConvNeXtTiny directly. This adapter is used ONLY for the approval-gated
ImageNet-initialised experiment. torchvision is not used to implement this model.

Architecture: Liu et al., A ConvNet for the 2020s, CVPR 2022 (course ref. 7).
Development assistance: ChatGPT; student review/testing/disclosure required.
"""
from __future__ import annotations
import torch
from torch import nn
from modules import ConvNeXtTiny


class RGBTransferConvNeXt(ConvNeXtTiny):
    """Same custom Tiny blocks, 3-channel pretrained stem and two-class head.

    Input is the existing [B,1,224,224] tensor in [-1,1]. Recover [0,1], repeat
    greyscale into RGB and apply ImageNet mean/std. No RGB information is invented.
    Existing aspect-preserving resize/padding is retained: this is not the
    torchvision weights' default resize/centre-crop pipeline.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.downsample[0][0] = nn.Conv2d(3, self.dims[0], 4, stride=4)
        self._initialise(self.downsample[0][0])
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406]).view(1,3,1,1))
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225]).view(1,3,1,1))

    def adapt_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (1,224,224):
            raise ValueError('Expected [B,1,224,224] input.')
        return ((x + 1) * .5).expand(-1,3,-1,-1).sub(self.rgb_mean).div(self.rgb_std)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.adapt_input(images)
        for downsample, stage in zip(self.downsample, self.stages):
            x = stage(downsample(x))
        return self.head(self.final_norm(x.mean(dim=(-2,-1))))


def import_torchvision_backbone(model: RGBTransferConvNeXt,
                               source: dict[str, torch.Tensor]) -> None:
    """Strict mapping of torchvision ConvNeXt state tensors into custom blocks.

    The 1000-class ImageNet head is deliberately not copied. All other source
    parameters must be consumed with matching shapes; unexpected/missing keys
    fail rather than silently yielding a partially pretrained network.
    """
    target = model.state_dict()
    mapping: dict[str,str] = {
        'downsample.0.0.weight':'features.0.0.weight',
        'downsample.0.0.bias':'features.0.0.bias',
        'downsample.0.1.norm.weight':'features.0.1.weight',
        'downsample.0.1.norm.bias':'features.0.1.bias',
        'final_norm.weight':'classifier.0.weight',
        'final_norm.bias':'classifier.0.bias',
    }
    for i in range(1,4):
        for suffix in ('weight','bias'):
            mapping[f'downsample.{i}.0.norm.{suffix}'] = f'features.{2*i}.0.{suffix}'
            mapping[f'downsample.{i}.1.{suffix}'] = f'features.{2*i}.1.{suffix}'
    for i, depth in enumerate(model.depths):
        for j in range(depth):
            start=f'stages.{i}.{j}'
            other=f'features.{2*i+1}.{j}'
            mapping[f'{start}.gamma'] = f'{other}.layer_scale'
            for ours, theirs in [('depthwise','0'),('norm','2'),('expand','3'),('contract','5')]:
                for suffix in ('weight','bias'):
                    mapping[f'{start}.{ours}.{suffix}']=f'{other}.block.{theirs}.{suffix}'
    allowed={'classifier.2.weight','classifier.2.bias'}
    if set(source) != set(mapping.values()) | allowed:
        raise ValueError('Pretrained state has unexpected or missing parameters; no partial loading.')
    if set(target) != set(mapping) | {'head.weight','head.bias','rgb_mean','rgb_std'}:
        raise ValueError('Custom model no longer matches the strict pretrained mapping.')
    for dest, src in mapping.items():
        value=source[src].detach().cpu()
        if dest.endswith('.gamma'):
            value=value.reshape(-1)
        if value.shape != target[dest].shape or not torch.isfinite(value).all():
            raise ValueError('Pretrained backbone shape/finite check failed.')
        target[dest]=value.clone()
    model.load_state_dict(target, strict=True)
