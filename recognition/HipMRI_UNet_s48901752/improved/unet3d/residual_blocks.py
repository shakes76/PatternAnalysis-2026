"""Residual 3D U-Net: the standard UNet3D with residual convolution blocks.

The only architectural change from ``modules.UNet3D`` is the convolution block.
Every ``DoubleConv3D`` (four encoder stages, the bottleneck and four decoder
stages) is replaced by a ``ResidualBlock3D`` with the same input and output
widths. Everything else is inherited unchanged from ``modules.py``: max-pooling,
2x2x2 transposed-convolution upsampling, long skip concatenation
[skip, upsampled], the 1x1x1 head, channel widths and input checks.

    resunet3d_256   encoder 32 -> 64 -> 128 -> 256, bottleneck 256
    resunet3d_512   encoder 32 -> 64 -> 128 -> 256, bottleneck 512

Input  : [B, 1, D, H, W], D, H and W divisible by 16 (64^3 and 96^3 patches).
Output : [B, 6, D, H, W] raw logits.

Sections
    1. Imports
    2. Residual convolution block
    3. Residual 3D U-Net
"""

# =============================================================================
# 1. Imports
# =============================================================================
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from modules import ENCODER_CHANNELS, DoubleConv3D, UNet3D

RESUNET3D_VARIANTS: dict[str, int] = {
    "resunet3d_256": 256,
    "resunet3d_512": 512,
}


# =============================================================================
# 2. Residual convolution block
# =============================================================================
class ResidualBlock3D(nn.Module):
    """output = ReLU(F(x) + shortcut(x)).

    F(x)         Conv 3x3x3 -> BatchNorm3d -> ReLU -> Conv 3x3x3 -> BatchNorm3d
                 (the standard double convolution without its final ReLU)
    shortcut(x)  identity when in_channels == out_channels, otherwise a
                 1x1x1 projection Conv3d -> BatchNorm3d to match the width
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        if in_channels == out_channels:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.residual(x) + self.shortcut(x))


# =============================================================================
# 3. Residual 3D U-Net
# =============================================================================
class ResUNet3D(UNet3D):
    """UNet3D with every DoubleConv3D replaced by a ResidualBlock3D of the same widths."""

    def __init__(
        self,
        bottleneck_channels: int,
        in_channels: int = 1,
        num_classes: int = 6,
        encoder_channels: Sequence[int] = ENCODER_CHANNELS,
    ) -> None:
        super().__init__(bottleneck_channels, in_channels, num_classes, encoder_channels)
        for stage in (*self.encoder.stages, self.bottleneck, *self.decoder.stages):
            stage.conv = residual_like(stage.conv)


def residual_like(block: DoubleConv3D) -> ResidualBlock3D:
    """A ResidualBlock3D with the same input and output widths as ``block``."""
    first, second = block.block[0], block.block[3]
    return ResidualBlock3D(first.in_channels, second.out_channels)


def build_resunet3d(variant: str, in_channels: int = 1, num_classes: int = 6) -> ResUNet3D:
    """Build a named variant from ``RESUNET3D_VARIANTS``."""
    if variant not in RESUNET3D_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(RESUNET3D_VARIANTS)}")
    return ResUNet3D(RESUNET3D_VARIANTS[variant], in_channels=in_channels, num_classes=num_classes)
