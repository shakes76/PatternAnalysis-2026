"""Building blocks for a 2D U-Net segmentation model."""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Apply two 3x3 convolutions with ReLU, preserving spatial dimensions."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downsample with 2x2 max pooling, then apply DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Resize to skip dimensions, concatenate channels, then apply DoubleConv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """Four-level baseline U-Net returning full-resolution class logits."""

    def __init__(self, in_channels=1, num_classes=6, base_channels=16):
        super().__init__()
        if min(in_channels, num_classes, base_channels) < 1:
            raise ValueError("Channel counts and num_classes must be positive")

        self.in_channels = in_channels
        self.num_classes = num_classes
        b = base_channels

        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16)

        self.up1 = Up(b * 16, b * 8, b * 8)
        self.up2 = Up(b * 8, b * 4, b * 4)
        self.up3 = Up(b * 4, b * 2, b * 2)
        self.up4 = Up(b * 2, b, b)
        self.out = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected input shape [B, C, H, W]")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        if min(x.shape[-2:]) < 16:
            raise ValueError("Input height and width must each be at least 16")

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)


# 2D adaptation inspired by Isensee et al. (2018), arXiv:1802.10508.
# Uses this module's existing torch, nn and F imports.

class PreActConv2D(nn.Sequential):
    """Instance normalization and LeakyReLU before a feature convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__(
            nn.InstanceNorm2d(in_channels, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2, bias=False),
        )


class ResidualContext2D(nn.Module):
    """Two pre-activation convolutions with channel dropout and identity skip."""

    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.body = nn.Sequential(
            PreActConv2D(channels, channels),
            nn.Dropout2d(p=dropout),
            PreActConv2D(channels, channels),
        )

    def forward(self, x):
        return x + self.body(x)


class ImprovedDown2D(nn.Sequential):
    """Learned stride-2 reduction followed by residual context aggregation."""

    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__(
            PreActConv2D(in_channels, out_channels, stride=2),
            ResidualContext2D(out_channels, dropout=dropout),
        )


class LocalizationUp2D(nn.Module):
    """Nearest resize, feature reduction, skip concatenation and localization."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = PreActConv2D(in_channels, out_channels)
        self.localize = nn.Sequential(
            PreActConv2D(out_channels + skip_channels, 2 * out_channels),
            PreActConv2D(2 * out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x, skip):
        # Exact skip size also supports odd input dimensions without cropping masks.
        x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
        x = self.reduce(x)
        return self.localize(torch.cat((skip, x), dim=1))


class ImprovedUNet2D(nn.Module):
    """Residual 2D U-Net with three summed, full-resolution segmentation heads.

    This is a documented 2D adaptation, not an exact 3D BraTS reproduction.
    Channel widths are b, 2b, 4b, 8b, 16b. Quarter-, half- and full-resolution
    decoder heads are resized and summed as raw logits before the shared loss.
    Input height/width must be at least 32 so bottleneck InstanceNorm retains
    multiple spatial values in both training and evaluation modes.
    """

    def __init__(self, in_channels=1, num_classes=6, base_channels=16, dropout=0.3):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (in_channels, num_classes, base_channels)):
            raise ValueError('Channel counts and num_classes must be positive integers')
        if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError('dropout must be in range [0, 1)')
        self.in_channels = in_channels
        self.num_classes = num_classes
        b = base_channels
        self.stem = nn.Conv2d(in_channels, b, kernel_size=3, padding=1, bias=False)
        self.context0 = ResidualContext2D(b, dropout=dropout)
        self.down1 = ImprovedDown2D(b, 2 * b, dropout=dropout)
        self.down2 = ImprovedDown2D(2 * b, 4 * b, dropout=dropout)
        self.down3 = ImprovedDown2D(4 * b, 8 * b, dropout=dropout)
        self.down4 = ImprovedDown2D(8 * b, 16 * b, dropout=dropout)
        self.up1 = LocalizationUp2D(16 * b, 8 * b, 8 * b)
        self.up2 = LocalizationUp2D(8 * b, 4 * b, 4 * b)
        self.up3 = LocalizationUp2D(4 * b, 2 * b, 2 * b)
        self.up4 = LocalizationUp2D(2 * b, b, b)
        self.seg_heads = nn.ModuleList([
            nn.Conv2d(4 * b, num_classes, kernel_size=1),
            nn.Conv2d(2 * b, num_classes, kernel_size=1),
            nn.Conv2d(b, num_classes, kernel_size=1),
        ])

    def forward_heads(self, x):
        """Return three full-resolution raw-logit maps for inspection/testing."""
        if x.ndim != 4:
            raise ValueError('Expected input shape [B, C, H, W]')
        if x.shape[1] != self.in_channels:
            raise ValueError(f'Expected {self.in_channels} input channels, got {x.shape[1]}')
        if min(x.shape[-2:]) < 32:
            raise ValueError('Input height and width must each be at least 32 for InstanceNorm')
        output_size = x.shape[-2:]
        x0 = self.context0(self.stem(x))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        decoded3 = self.up1(x4, x3)
        decoded2 = self.up2(decoded3, x2)
        decoded1 = self.up3(decoded2, x1)
        decoded0 = self.up4(decoded1, x0)
        features = (decoded2, decoded1, decoded0)
        logits = []
        for feature, head in zip(features, self.seg_heads):
            value = head(feature)
            if value.shape[-2:] != output_size:
                value = F.interpolate(value, size=output_size, mode='nearest')
            logits.append(value)
        return tuple(logits)

    def forward(self, x):
        quarter, half, full = self.forward_heads(x)
        return quarter + half + full
