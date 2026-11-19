"""Standard 2D U-Net components for six-class HipMRI segmentation.

Two variants share this implementation and differ only in the number of
bottleneck channels (see ``UNET2D_VARIANTS``):

    Model A  unet2d_256   encoder 32 -> 64 -> 128 -> 256, bottleneck 256
    Model B  unet2d_512   encoder 32 -> 64 -> 128 -> 256, bottleneck 512

Input  : [B, 1, H, W] float tensor, H and W divisible by 16.
Output : [B, 6, H, W] raw logits (no softmax; the loss expects logits).

This file depends on PyTorch only (COMP3710 spec, Section 4.1).

Sections
    1. Imports
    2. Convolution blocks
    3. Encoder
    4. Bottleneck
    5. Decoder
    6. Skip connections
    7. Complete U-Net model
    3D U-Net (end of file): the same sections for UNet3D (unet3d_256, unet3d_512),
    reusing ENCODER_CHANNELS, concatenate_skip and count_parameters
"""

# =============================================================================
# 1. Imports
# =============================================================================
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

# Encoder widths shared by both variants.
ENCODER_CHANNELS: tuple[int, ...] = (32, 64, 128, 256)

# The only place the bottleneck width of each variant is defined.
UNET2D_VARIANTS: dict[str, int] = {
    "unet2d_256": 256,
    "unet2d_512": 512,
}


# =============================================================================
# 2. Convolution blocks
# =============================================================================
class DoubleConv(nn.Module):
    """(Conv 3x3 -> BatchNorm -> ReLU) x 2, with same padding.

    Convolution bias is disabled because BatchNorm adds its own shift.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =============================================================================
# 3. Encoder
# =============================================================================
class EncoderStage(nn.Module):
    """DoubleConv for feature extraction, then 2x2 max-pool for downsampling.

    Returns the pooled tensor (input to the next stage) and the pre-pool
    features (the skip connection for the matching decoder stage).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.pool(skip), skip


class Encoder(nn.Module):
    """Four encoder stages: in_channels -> 32 -> 64 -> 128 -> 256."""

    def __init__(self, in_channels: int, channels: Sequence[int]) -> None:
        super().__init__()
        widths = [in_channels, *channels]
        self.stages = nn.ModuleList(
            EncoderStage(widths[i], widths[i + 1]) for i in range(len(channels))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        for stage in self.stages:
            x, skip = stage(x)
            skips.append(skip)
        return x, skips


# =============================================================================
# 4. Bottleneck
# =============================================================================
class Bottleneck(nn.Module):
    """DoubleConv at the lowest resolution (H/16 x W/16): 256 -> B channels.

    B is 256 (Model A) or 512 (Model B). This is the intended architectural
    difference between the two variants.
    """

    def __init__(self, in_channels: int, bottleneck_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, bottleneck_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# =============================================================================
# 5. Decoder
# =============================================================================
class DecoderStage(nn.Module):
    """2x2 transposed-conv upsampling, skip concatenation, then DoubleConv.

    After concatenation the tensor has 2 * out_channels channels
    (skip features + upsampled features), reduced back to out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(2 * out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(x)
        return self.conv(concatenate_skip(skip, upsampled))


class Decoder(nn.Module):
    """Mirror of the encoder: B -> 256 -> 128 -> 64 -> 32.

    Only the first stage's input width depends on the bottleneck size.
    """

    def __init__(self, bottleneck_channels: int, channels: Sequence[int]) -> None:
        super().__init__()
        widths = list(reversed(channels))  # [256, 128, 64, 32]
        inputs = [bottleneck_channels, *widths[:-1]]
        self.stages = nn.ModuleList(
            DecoderStage(in_ch, out_ch) for in_ch, out_ch in zip(inputs, widths, strict=True)
        )

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        # Deepest skip is used first.
        for stage, skip in zip(self.stages, reversed(skips), strict=True):
            x = stage(x, skip)
        return x


# =============================================================================
# 6. Skip connections
# =============================================================================
def concatenate_skip(skip: torch.Tensor, upsampled: torch.Tensor) -> torch.Tensor:
    """Concatenate encoder features with upsampled decoder features.

    Channel order is [skip, upsampled]. Spatial sizes must match exactly; the
    model rejects inputs whose height or width is not divisible by 16 instead
    of silently resizing.
    """
    if skip.shape[0] != upsampled.shape[0] or skip.shape[2:] != upsampled.shape[2:]:
        raise ValueError(
            f"Skip/upsampled shape mismatch: {tuple(skip.shape)} vs {tuple(upsampled.shape)}"
        )
    return torch.cat((skip, upsampled), dim=1)


# =============================================================================
# 7. Complete U-Net model
# =============================================================================
class UNet2D(nn.Module):
    """Standard 2D U-Net: encoder, bottleneck, decoder, 1x1 output head."""

    def __init__(
        self,
        bottleneck_channels: int,
        in_channels: int = 1,
        num_classes: int = 6,
        encoder_channels: Sequence[int] = ENCODER_CHANNELS,
    ) -> None:
        super().__init__()
        self.bottleneck_channels = int(bottleneck_channels)
        self.num_classes = int(num_classes)
        self.downsample_factor = 2 ** len(encoder_channels)

        self.encoder = Encoder(in_channels, encoder_channels)
        self.bottleneck = Bottleneck(encoder_channels[-1], bottleneck_channels)
        self.decoder = Decoder(bottleneck_channels, encoder_channels)
        self.head = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B, C, H, W], got {tuple(x.shape)}")
        height, width = x.shape[2:]
        if height % self.downsample_factor or width % self.downsample_factor:
            raise ValueError(
                f"Height and width must be divisible by {self.downsample_factor}, "
                f"got {height}x{width}"
            )
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.head(x)


def build_model(variant: str, in_channels: int = 1, num_classes: int = 6) -> UNet2D:
    """Build a named variant from ``UNET2D_VARIANTS``."""
    if variant not in UNET2D_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(UNET2D_VARIANTS)}")
    return UNet2D(UNET2D_VARIANTS[variant], in_channels=in_channels, num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 3D U-Net (standard; added after the 2D baselines, which are unchanged above)
# =============================================================================
# Same design as UNet2D with every layer swapped for its 3D counterpart:
#
#     unet3d_256   encoder 32 -> 64 -> 128 -> 256, bottleneck 256
#     unet3d_512   encoder 32 -> 64 -> 128 -> 256, bottleneck 512
#
# Input  : [B, 1, D, H, W] float tensor, D, H and W divisible by 16
#          (true for the 64^3 and 96^3 training and sliding-window patches).
# Output : [B, 6, D, H, W] raw logits.
# ENCODER_CHANNELS, concatenate_skip and count_parameters are shared with 2D.

UNET3D_VARIANTS: dict[str, int] = {
    "unet3d_256": 256,
    "unet3d_512": 512,
}


# -----------------------------------------------------------------------------
# 3D.2 Convolution blocks
# -----------------------------------------------------------------------------
class DoubleConv3D(nn.Module):
    """(Conv 3x3x3 -> BatchNorm3d -> ReLU) x 2, with same padding and no conv bias."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# -----------------------------------------------------------------------------
# 3D.3 Encoder
# -----------------------------------------------------------------------------
class EncoderStage3D(nn.Module):
    """DoubleConv3D, then 2x2x2 max-pool. Returns (pooled, pre-pool skip)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.pool(skip), skip


class Encoder3D(nn.Module):
    """Four encoder stages: in_channels -> 32 -> 64 -> 128 -> 256."""

    def __init__(self, in_channels: int, channels: Sequence[int]) -> None:
        super().__init__()
        widths = [in_channels, *channels]
        self.stages = nn.ModuleList(
            EncoderStage3D(widths[i], widths[i + 1]) for i in range(len(channels))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        for stage in self.stages:
            x, skip = stage(x)
            skips.append(skip)
        return x, skips


# -----------------------------------------------------------------------------
# 3D.4 Bottleneck
# -----------------------------------------------------------------------------
class Bottleneck3D(nn.Module):
    """DoubleConv3D at the lowest resolution (D/16 x H/16 x W/16): 256 -> B channels.

    B is 256 (unet3d_256) or 512 (unet3d_512), the only intended difference.
    """

    def __init__(self, in_channels: int, bottleneck_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv3D(in_channels, bottleneck_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# -----------------------------------------------------------------------------
# 3D.5 Decoder (3D.6 skip connections reuse concatenate_skip from Section 6)
# -----------------------------------------------------------------------------
class DecoderStage3D(nn.Module):
    """2x2x2 transposed-conv upsampling, skip concatenation, then DoubleConv3D."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(2 * out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(x)
        return self.conv(concatenate_skip(skip, upsampled))


class Decoder3D(nn.Module):
    """Mirror of the encoder: B -> 256 -> 128 -> 64 -> 32."""

    def __init__(self, bottleneck_channels: int, channels: Sequence[int]) -> None:
        super().__init__()
        widths = list(reversed(channels))  # [256, 128, 64, 32]
        inputs = [bottleneck_channels, *widths[:-1]]
        self.stages = nn.ModuleList(
            DecoderStage3D(in_ch, out_ch) for in_ch, out_ch in zip(inputs, widths, strict=True)
        )

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for stage, skip in zip(self.stages, reversed(skips), strict=True):
            x = stage(x, skip)
        return x


# -----------------------------------------------------------------------------
# 3D.7 Complete 3D U-Net model
# -----------------------------------------------------------------------------
class UNet3D(nn.Module):
    """Standard 3D U-Net: encoder, bottleneck, decoder, 1x1x1 output head."""

    def __init__(
        self,
        bottleneck_channels: int,
        in_channels: int = 1,
        num_classes: int = 6,
        encoder_channels: Sequence[int] = ENCODER_CHANNELS,
    ) -> None:
        super().__init__()
        self.bottleneck_channels = int(bottleneck_channels)
        self.num_classes = int(num_classes)
        self.downsample_factor = 2 ** len(encoder_channels)

        self.encoder = Encoder3D(in_channels, encoder_channels)
        self.bottleneck = Bottleneck3D(encoder_channels[-1], bottleneck_channels)
        self.decoder = Decoder3D(bottleneck_channels, encoder_channels)
        self.head = nn.Conv3d(encoder_channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input [B, C, D, H, W], got {tuple(x.shape)}")
        if any(n % self.downsample_factor for n in x.shape[2:]):
            raise ValueError(
                f"Depth, height and width must be divisible by {self.downsample_factor}, "
                f"got {'x'.join(map(str, x.shape[2:]))}"
            )
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.head(x)


def build_model3d(variant: str, in_channels: int = 1, num_classes: int = 6) -> UNet3D:
    """Build a named variant from ``UNET3D_VARIANTS``."""
    if variant not in UNET3D_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(UNET3D_VARIANTS)}")
    return UNet3D(UNET3D_VARIANTS[variant], in_channels=in_channels, num_classes=num_classes)
