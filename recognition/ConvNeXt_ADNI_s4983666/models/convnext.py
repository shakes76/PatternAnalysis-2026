"""ConvNeXt for grayscale MRI slices, implemented from scratch in PyTorch.

Architecture: Liu et al., "A ConvNet for the 2020s", CVPR 2022,
https://arxiv.org/abs/2201.03545 . The authors' reference implementation
(https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py)
was consulted for the block layout. ``ConvNeXtTiny`` and ``ConvNeXtLite`` are
randomly initialized; ``ConvNeXtTinyImageNet`` has the same layers plus a fixed
input normalization so that the authors' ImageNet-1k weights can be loaded
into it (models/pretrained.py).
"""

import torch
from torch import nn


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels at every pixel of an NCHW tensor (channels-first)."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels, eps=1e-6)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channels_last = features.permute(0, 2, 3, 1)
        return super().forward(channels_last).permute(0, 3, 1, 2)


class StochasticDepth(nn.Module):
    """Drop a whole residual branch per image during training (DropPath).

    Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016. Surviving
    branches are divided by the survival probability so the expected output
    is unchanged; evaluation is deterministic.
    """

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("Stochastic-depth probability must be in [0, 1).")
        self.probability = float(probability)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return residual
        survival = 1.0 - self.probability
        mask_shape = (residual.shape[0],) + (1,) * (residual.ndim - 1)
        mask = residual.new_empty(mask_shape).bernoulli_(survival)
        return residual * (mask / survival)


class ConvNeXtBlock(nn.Module):
    """7x7 depthwise conv -> LayerNorm -> 4x MLP (GELU) -> layer scale -> residual."""

    def __init__(self, channels: int, drop_probability: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        # 1x1 convolutions written as Linear layers on channels-last tensors.
        self.expand = nn.Linear(channels, 4 * channels)
        self.activation = nn.GELU()
        self.project = nn.Linear(4 * channels, channels)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-6))
        self.drop_path = StochasticDepth(drop_probability)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(features).permute(0, 2, 3, 1)
        residual = self.project(self.activation(self.expand(self.norm(residual))))
        residual = (residual * self.layer_scale).permute(0, 3, 1, 2)
        return features + self.drop_path(residual)


class ConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny with a configurable input channel count and one AD logit.

    ``input_channels`` is 1 for a single slice or 3 for a slice stacked with
    its two neighbours. DropPath rises linearly from 0 to ``drop_path`` across
    the blocks, as in the paper.
    """

    depths = (3, 3, 9, 3)
    channels = (96, 192, 384, 768)

    def __init__(self, input_channels: int = 1, drop_path: float = 0.1) -> None:
        super().__init__()
        if input_channels not in (1, 3):
            raise ValueError("input_channels must be 1 or 3.")
        if not 0.0 <= drop_path < 1.0:
            raise ValueError("drop_path must be in [0, 1).")
        self.input_channels = input_channels
        # Stem: non-overlapping 4x4 "patchify" convolution followed by LayerNorm.
        self.stem = nn.Sequential(nn.Conv2d(input_channels, self.channels[0], 4, stride=4),
                                  LayerNorm2d(self.channels[0]))
        # Between stages: LayerNorm then a 2x2 stride-2 convolution.
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(LayerNorm2d(previous), nn.Conv2d(previous, current, 2, stride=2))
            for previous, current in zip(self.channels[:-1], self.channels[1:])
        ])
        total_blocks = sum(self.depths)
        self.stages = nn.ModuleList()
        block_index = 0
        for channels, depth in zip(self.channels, self.depths):
            blocks = []
            for _ in range(depth):
                blocks.append(ConvNeXtBlock(channels, drop_path * block_index / (total_blocks - 1)))
                block_index += 1
            self.stages.append(nn.Sequential(*blocks))
        # Head: global average pooling -> LayerNorm -> Linear.
        self.final_norm = nn.LayerNorm(self.channels[-1], eps=1e-6)
        self.classifier = nn.Linear(self.channels[-1], 1)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        """Random truncated-normal initialization (std 0.02) as in the paper."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map [batch, channels, height, width] images to [batch] AD logits."""
        if images.ndim != 4 or images.shape[1] != self.input_channels or min(images.shape[-2:]) < 32:
            raise ValueError(f"Expected [batch, {self.input_channels}, height, width] with sides >= 32.")
        features = self.stages[0](self.stem(images))
        for downsample, stage in zip(self.downsample_layers, self.stages[1:]):
            features = stage(downsample(features))
        pooled = self.final_norm(features.mean(dim=(2, 3)))
        return self.classifier(pooled).squeeze(1)


class ImageNetInput(nn.Module):
    """Map inputs in [-1, 1] to the ImageNet mean/std normalization of the pretrained weights.

    The data pipeline stays unchanged; the constants are buffers, so they are
    saved with the checkpoint and cannot be trained.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return ((images + 1) / 2 - self.mean) / self.std


class ConvNeXtTinyImageNet(ConvNeXtTiny):
    """ConvNeXt-Tiny for three-slice input that starts from ImageNet-1k weights.

    The three context slices take the place of the RGB channels. The weights
    are loaded by ``models.pretrained.load_imagenet_weights``; the one-logit
    classifier always starts from random initialization.
    """

    def __init__(self, input_channels: int = 3, drop_path: float = 0.1) -> None:
        if input_channels != 3:
            raise ValueError("ImageNet weights expect three input channels (use --context-slices 3).")
        super().__init__(input_channels, drop_path)
        self.input_normalization = ImageNetInput()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected [batch, 3, height, width] images.")
        return super().forward(self.input_normalization(images))


class ConvNeXtLite(ConvNeXtTiny):
    """Smaller ConvNeXt with the same blocks: depths (2, 2, 6, 2), widths (48 ... 384)."""

    depths = (2, 2, 6, 2)
    channels = (48, 96, 192, 384)
