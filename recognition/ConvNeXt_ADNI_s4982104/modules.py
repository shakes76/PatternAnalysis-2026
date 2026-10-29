"""CNN baseline and self-implemented ConvNeXt-Tiny components."""
from torch import nn


class SimpleCNN(nn.Module):
    """Produce two raw logits (NC=0, AD=1) from [B,1,H,W] MRI slices.

    Three convolution/ReLU/pooling blocks reduce spatial size; global average
    pooling avoids a large flattened classifier. No pretrained weights.
    """
    def __init__(self):
        super().__init__()
        blocks=[]
        for inputs, outputs in [(1,16),(16,32),(32,64)]:
            blocks.extend([nn.Conv2d(inputs,outputs,3,padding=1),
                           nn.ReLU(),nn.MaxPool2d(2)])
        self.features=nn.Sequential(*blocks,nn.AdaptiveAvgPool2d(1))
        self.classifier=nn.Linear(64,2)

    def forward(self, images):
        return self.classifier(self.features(images).flatten(1))


# ConvNeXt architecture: Liu et al., "A ConvNet for the 2020s" (CVPR 2022),
# https://arxiv.org/abs/2201.03545. Component layout and state names adapted from
# TorchVision (BSD-3-Clause): https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
# See THIRD_PARTY_NOTICES.md. No TorchVision model class is used below.
import torch


class LayerNorm2d(nn.LayerNorm):
    """Normalize channels at each spatial location, using channels-last LayerNorm."""
    def forward(self, images):
        channels_last = images.permute(0, 2, 3, 1)
        return super().forward(channels_last).permute(0, 3, 1, 2)


class Permute(nn.Module):
    """Change tensor layout without learnable parameters."""
    def __init__(self, dimensions):
        super().__init__()
        self.dimensions = dimensions

    def forward(self, values):
        return values.permute(self.dimensions)


class DropPath(nn.Module):
    """Drop a residual branch independently per image during training only."""
    def __init__(self, probability):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError('DropPath probability must be in [0,1)')
        self.probability = probability

    def forward(self, residual):
        if not self.training or self.probability == 0:
            return residual
        survival = 1 - self.probability
        mask = torch.empty((residual.shape[0],) + (1,) * (residual.ndim-1),
                           dtype=residual.dtype, device=residual.device)
        mask.bernoulli_(survival).div_(survival)
        return residual * mask


class ConvNeXtBlock(nn.Module):
    """7x7 depthwise convolution, channel MLP, layer scale and residual path."""
    def __init__(self, channels, drop_probability):
        super().__init__()
        # Names intentionally match the official checkpoint; no permissive loading.
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels),
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(channels, eps=1e-6),
            nn.Linear(channels, 4*channels), nn.GELU(),
            nn.Linear(4*channels, channels), Permute((0, 3, 1, 2)))
        self.layer_scale = nn.Parameter(torch.full((channels, 1, 1), 1e-6))
        self.stochastic_depth = DropPath(drop_probability)

    def forward(self, values):
        branch = self.block(values) * self.layer_scale
        return values + self.stochastic_depth(branch)


class ConvNeXtTiny(nn.Module):
    """Tiny stages [3,3,9,3], widths [96,192,384,768]; raw class logits.

    Default stochastic depth 0.1 is architectural regularization, not image
    augmentation. The 1000-class variant permits exact pretrained-state checking
    before replacing only the final linear layer for NC/AD.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        stages = ((96,3), (192,3), (384,9), (768,3))
        features = [nn.Sequential(nn.Conv2d(3,96,4,stride=4),
                                  LayerNorm2d(96,eps=1e-6))]
        block_number = 0
        for stage_number, (width, depth) in enumerate(stages):
            blocks = []
            for _ in range(depth):
                blocks.append(ConvNeXtBlock(width, 0.1*block_number/17))
                block_number += 1
            features.append(nn.Sequential(*blocks))
            if stage_number < 3:
                features.append(nn.Sequential(LayerNorm2d(width,eps=1e-6),
                    nn.Conv2d(width,stages[stage_number+1][0],2,stride=2)))
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(LayerNorm2d(768,eps=1e-6),
                                         nn.Flatten(1),nn.Linear(768,num_classes))
        for layer in self.modules():
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                nn.init.trunc_normal_(layer.weight,std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, images):
        return self.classifier(self.avgpool(self.features(images)))


def build_model(name='simple_cnn', num_classes=2):
    """Build architecture only. Pretrained loading is explicit in the checker."""
    if name == 'simple_cnn' and num_classes == 2:
        return SimpleCNN()
    if name == 'convnext_tiny':
        return ConvNeXtTiny(num_classes)
    raise ValueError(f'Unsupported model/classes: {name}/{num_classes}')
