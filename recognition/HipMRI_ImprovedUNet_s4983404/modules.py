"""Explainable 2D segmentation models, implemented from papers rather than copied code.

Baseline: Ronneberger et al., MICCAI 2015, https://arxiv.org/abs/1505.04597.
Improved: 2D adaptation of Isensee et al., BRATS 2017 contribution (2018),
https://arxiv.org/abs/1802.10508, section 2.1 and figure 1.
"""
import torch
from pathlib import Path
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())


class BasicUNet(nn.Module):
    """Four pool levels; padded convolutions preserve the full field of view."""
    def __init__(self, classes: int = 6, width: int = 16) -> None:
        super().__init__()
        channels = [width * 2**level for level in range(5)]
        self.encoders = nn.ModuleList([DoubleConv(1, channels[0])])
        self.encoders.extend(DoubleConv(left, right)
                             for left, right in zip(channels, channels[1:]))
        self.ups = nn.ModuleList(nn.ConvTranspose2d(channels[i+1], channels[i], 2, 2)
                                for i in range(3, -1, -1))
        self.decoders = nn.ModuleList(DoubleConv(2*channels[i], channels[i])
                                     for i in range(3, -1, -1))
        self.output = nn.Conv2d(width, classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skips = []
        features = image
        for index, encoder in enumerate(self.encoders):
            features = encoder(features if index == 0 else F.max_pool2d(features, 2))
            skips.append(features)
        for upsample, decoder, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
            features = upsample(features)
            features = F.interpolate(features, size=skip.shape[-2:], mode='nearest')
            features = decoder(torch.cat((features, skip), dim=1))
        return self.output(features)


class PreActivation(nn.Sequential):
    def __init__(self, inputs: int, outputs: int, kernel: int = 3) -> None:
        super().__init__(nn.InstanceNorm2d(inputs, affine=True),
                         nn.LeakyReLU(0.01),
                         nn.Conv2d(inputs, outputs, kernel, padding=kernel//2))


class Context(nn.Module):
    """Pre-activation residual context, with dropout between its convolutions."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(PreActivation(channels, channels), nn.Dropout2d(0.3),
                                  PreActivation(channels, channels))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.body(features)


class ImprovedUNet(nn.Module):
    """Isensee-inspired 2D adaptation: context residuals and summed scale heads.

    3D operators become 2D; one MRI channel and six exclusive labels replace
    four modalities and overlapping tumour regions. No claim of 3D continuity.
    """
    def __init__(self, classes: int = 6, width: int = 16) -> None:
        super().__init__()
        channels = [width * 2**level for level in range(5)]
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        self.contexts = nn.ModuleList(Context(channel) for channel in channels)
        self.downs = nn.ModuleList(nn.Conv2d(left, right, 3, stride=2, padding=1)
                                  for left, right in zip(channels, channels[1:]))
        self.ups = nn.ModuleList(PreActivation(channels[i+1], channels[i])
                                for i in range(3, -1, -1))
        self.localizers = nn.ModuleList(nn.Sequential(
            PreActivation(2*channels[i], 2*channels[i]),
            PreActivation(2*channels[i], channels[i], 1)) for i in range(3, -1, -1))
        self.heads = nn.ModuleList(nn.Conv2d(channels[i], classes, 1)
                                  for i in (2, 1, 0))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.stem(image)
        skips = []
        for index, context in enumerate(self.contexts):
            features = context(features if index == 0 else self.downs[index-1](features))
            skips.append(features)
        outputs = []
        for index, (up, localize, skip) in enumerate(
                zip(self.ups, self.localizers, reversed(skips[:-1]))):
            features = up(F.interpolate(features, size=skip.shape[-2:], mode='nearest'))
            features = localize(torch.cat((features, skip), dim=1))
            if index >= 1:
                outputs.append(self.heads[index-1](features))
        # Summed segmentation heads inject gradients at three decoder resolutions.
        return sum(F.interpolate(output, size=image.shape[-2:], mode='nearest')
                   for output in outputs)


def build_model(name: str, width: int = 16) -> nn.Module:
    if name not in ('basic', 'improved'):
        raise ValueError('model must be basic or improved')
    return {'basic': BasicUNet, 'improved': ImprovedUNet}[name](width=width)


def load_checkpoint(path: Path, device: str) -> dict:
    """Frozen weights record their own architecture; never guess it from flags."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint.get('config') or {}
    if 'model' not in config or 'width' not in config:
        raise ValueError('checkpoint has no model/width configuration')
    return checkpoint
