"""Independent PyTorch implementations for the HipMRI feasibility prototype.

References:
    Ronneberger et al. (2015), U-Net: https://arxiv.org/abs/1505.04597
    Isensee et al. (2018), Brain Tumor Segmentation and Radiomics Survival
    Prediction, section 2.1: https://arxiv.org/abs/1802.10508

The baseline uses the usual U-Net encoder/decoder with concatenated skips.
Unlike the original U-Net, it uses padded convolutions and configurable width.
ImprovedUNet2D is a 2D adaptation, not an exact reproduction of the 3D BraTS
network: it uses spatial Dropout2d, configurable width/classes, size-matched
upsampling, and three segmentation heads. Paper training, augmentation and
ensembling are not implemented here. Both models return unnormalized logits
of the input spatial size in training and evaluation; apply softmax only when
probabilities are needed. No external implementation code is used.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_configuration(in_channels: int, num_classes: int, base_channels: int) -> None:
    for name, value in (("in_channels", in_channels), ("num_classes", num_classes),
                        ("base_channels", base_channels)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_input(x: Tensor, in_channels: int) -> None:
    if x.ndim != 4:
        raise ValueError(f"Expected input [B,C,H,W], got shape {tuple(x.shape)}")
    if x.shape[0] < 1 or x.shape[1] != in_channels:
        raise ValueError(f"Expected a nonempty batch with {in_channels} channels")
    if min(x.shape[-2:]) < 32:
        raise ValueError("Input height and width must both be at least 32")
    if not torch.is_floating_point(x):
        raise TypeError("Input images must be floating-point tensors")


class _DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class _UNetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.fuse = _DoubleConv(2 * out_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        # Pooling rounds down odd dimensions; restore the exact skip dimensions.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((skip, x), dim=1))


class UNet2D(nn.Module):
    """Five-level padded U-Net baseline; input H and W must each be >= 32."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 base_channels: int = 16) -> None:
        super().__init__()
        _validate_configuration(in_channels, num_classes, base_channels)
        self.in_channels = in_channels
        channels = [base_channels * (2 ** level) for level in range(5)]
        self.encoders = nn.ModuleList([
            _DoubleConv(in_channels if level == 0 else channels[level - 1], width)
            for level, width in enumerate(channels)
        ])
        self.pool = nn.MaxPool2d(2)
        self.decoders = nn.ModuleList([
            _UNetUp(channels[level + 1], channels[level])
            for level in range(3, -1, -1)
        ])
        self.head = nn.Conv2d(channels[0], num_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        _validate_input(x, self.in_channels)
        skips = []
        for level, encoder in enumerate(self.encoders):
            if level > 0:
                x = self.pool(x)
            x = encoder(x)
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = decoder(x, skip)
        return self.head(x)


class _ContextStage(nn.Module):
    """Resolution/channel projection followed by a pre-activation residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.context = nn.Sequential(
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.Dropout2d(p=0.3),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.project(x)
        return shortcut + self.context(shortcut)


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )


class _LocalizationStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = _ConvNormAct(in_channels, out_channels, 3)
        self.localize = nn.Sequential(
            _ConvNormAct(2 * out_channels, 2 * out_channels, 3),
            _ConvNormAct(2 * out_channels, out_channels, 1),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.up(x)
        return self.localize(torch.cat((skip, x), dim=1))


class ImprovedUNet2D(nn.Module):
    """Isensee-inspired 2D residual U-Net with summed deep supervision.

    Five context levels use stride-2 convolutions, instance normalization,
    LeakyReLU(0.01), and spatial dropout(0.3). Decoder features at quarter,
    half, and full resolution each have a 1x1 segmentation head. Their logits
    are nearest-upsampled to full resolution and summed, so the final loss
    supervises all three heads. No separate auxiliary loss is required.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 base_channels: int = 16) -> None:
        super().__init__()
        _validate_configuration(in_channels, num_classes, base_channels)
        self.in_channels = in_channels
        channels = [base_channels * (2 ** level) for level in range(5)]
        self.encoders = nn.ModuleList([
            _ContextStage(in_channels if level == 0 else channels[level - 1],
                          width, stride=1 if level == 0 else 2)
            for level, width in enumerate(channels)
        ])
        self.decoders = nn.ModuleList([
            _LocalizationStage(channels[level + 1], channels[level])
            for level in range(3, -1, -1)
        ])
        self.segmentation_heads = nn.ModuleList([
            nn.Conv2d(channels[level], num_classes, 1)
            for level in (2, 1, 0)
        ])

    def forward(self, x: Tensor) -> Tensor:
        _validate_input(x, self.in_channels)
        output_size = x.shape[-2:]
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        logits = []
        for index, (decoder, skip) in enumerate(zip(self.decoders, reversed(skips[:-1]))):
            x = decoder(x, skip)
            if index >= 1:
                score = self.segmentation_heads[index - 1](x)
                if score.shape[-2:] != output_size:
                    score = F.interpolate(score, size=output_size, mode="nearest")
                logits.append(score)
        return logits[0] + logits[1] + logits[2]


def build_model(name: str, in_channels: int = 1, num_classes: int = 2,
                base_channels: int = 16) -> nn.Module:
    """Build ``unet`` or ``improved_unet``; no network downloads are performed."""
    models = {"unet": UNet2D, "improved_unet": ImprovedUNet2D}
    if name not in models:
        raise ValueError(f"Unknown model {name!r}; expected one of {tuple(models)}")
    return models[name](in_channels, num_classes, base_channels)
