"""Model components. Only PyTorch is used in this file, as required by the brief."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _norm(channels: int) -> nn.GroupNorm:
    """Choose a group count that divides the channel count."""
    return nn.GroupNorm(math.gcd(8, channels), channels)


class DoubleConv2D(nn.Sequential):
    """Two convolution–normalization–activation steps for the 2D U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet2D(nn.Module):
    """Conventional slice-wise U-Net baseline with three encoder reductions."""

    def __init__(self, in_channels: int = 1, classes: int = 2, width: int = 16):
        super().__init__()
        widths = [width, width * 2, width * 4, width * 8]
        self.encoder = nn.ModuleList(
            DoubleConv2D(in_channels if i == 0 else widths[i - 1], channels)
            for i, channels in enumerate(widths)
        )
        self.pool = nn.MaxPool2d(2)
        self.decoder = nn.ModuleList(
            DoubleConv2D(widths[i + 1] + widths[i], widths[i])
            for i in (2, 1, 0)
        )
        self.head = nn.Conv2d(width, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_features = []
        for i, block in enumerate(self.encoder):
            x = block(x if i == 0 else self.pool(x))
            skip_features.append(x)
        x = skip_features[-1]
        for block, skip in zip(self.decoder, reversed(skip_features[:-1])):
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return self.head(x)


class ResidualBlock3D(nn.Module):
    """Two 3D convolutions and an identity/projection residual path."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = _norm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _norm(out_channels)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv3d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        return F.relu(self.norm2(self.conv2(x)) + residual, inplace=True)


class ImprovedUNet3D(nn.Module):
    """Residual 3D U-Net with auxiliary logits for deep supervision.

    The output dictionary always contains full-resolution ``logits`` and
    two lower-resolution ``aux`` predictions. The caller decides whether
    to include auxiliary heads in the loss.
    """

    def __init__(self, in_channels: int = 1, classes: int = 2, width: int = 12):
        super().__init__()
        widths = [width, width * 2, width * 4, width * 8]
        self.encoder = nn.ModuleList(
            ResidualBlock3D(in_channels if i == 0 else widths[i - 1], channels)
            for i, channels in enumerate(widths)
        )
        self.pool = nn.MaxPool3d(2)
        self.decoder = nn.ModuleList(
            ResidualBlock3D(widths[i + 1] + widths[i], widths[i])
            for i in (2, 1, 0)
        )
        self.head = nn.Conv3d(width, classes, 1)
        self.aux_heads = nn.ModuleList([
            nn.Conv3d(widths[2], classes, 1),
            nn.Conv3d(widths[1], classes, 1),
        ])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        skip_features = []
        for i, block in enumerate(self.encoder):
            x = block(x if i == 0 else self.pool(x))
            skip_features.append(x)
        x = skip_features[-1]
        auxiliary_logits = []
        for i, (block, skip) in enumerate(zip(self.decoder, reversed(skip_features[:-1]))):
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
            if i < 2:
                auxiliary_logits.append(self.aux_heads[i](x))
        return {"logits": self.head(x), "aux": auxiliary_logits}
