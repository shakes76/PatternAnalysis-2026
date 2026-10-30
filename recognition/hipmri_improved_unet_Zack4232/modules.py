"""Implement standard and improved 2D U-Net models using PyTorch.

Reference:
Ronneberger et al. (2015), U-Net: Convolutional Networks for
Biomedical Image Segmentation.

UNet2D is the unchanged baseline. ImprovedUNet2D is a 2D adaptation
of Isensee et al. (2018), https://arxiv.org/abs/1802.10508.
"""

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Apply two padded convolutions with ReLU activations."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    """Upsample features and combine them with an encoder skip."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(
            out_channels + skip_channels, out_channels
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError("Decoder and skip spatial shapes do not match.")
        return self.conv(torch.cat((skip, x), dim=1))


class UNet2D(nn.Module):
    """Predict per-pixel logits for inputs divisible by 16."""

    def __init__(self, in_channels=1, num_classes=6, base_channels=32):
        super().__init__()
        if min(in_channels, num_classes, base_channels) < 1:
            raise ValueError("Channel counts must be positive.")

        self.in_channels = in_channels
        b = base_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)
        self.bottleneck = DoubleConv(b * 8, b * 16)

        self.dec4 = UpBlock(b * 16, b * 8, b * 8)
        self.dec3 = UpBlock(b * 8, b * 4, b * 4)
        self.dec2 = UpBlock(b * 4, b * 2, b * 2)
        self.dec1 = UpBlock(b * 2, b, b)
        self.output = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Expected input shape [B, in_channels, H, W].")
        height, width = x.shape[-2:]
        if min(height, width) < 16 or height % 16 or width % 16:
            raise ValueError("Input dimensions must be multiples of 16.")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))

        x = self.dec4(x, e4)
        x = self.dec3(x, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        return self.output(x)


class PreActConv2D(nn.Module):
    """Apply instance normalization, LeakyReLU, then convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.InstanceNorm2d(in_channels, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=kernel_size // 2, bias=False),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualContext2D(nn.Module):
    """Add a two-convolution residual branch with element-wise dropout."""

    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.conv1 = PreActConv2D(channels, channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = PreActConv2D(channels, channels)

    def forward(self, x):
        return x + self.conv2(self.dropout(self.conv1(x)))


class Localization2D(nn.Module):
    """Upsample, join the skip, and reduce channels with a 1x1 convolution."""

    def __init__(self, in_channels, skip_channels):
        super().__init__()
        self.up_conv = PreActConv2D(in_channels, skip_channels)
        self.mix = PreActConv2D(2 * skip_channels, 2 * skip_channels)
        self.reduce = PreActConv2D(2 * skip_channels, skip_channels, kernel_size=1)

    def forward(self, x, skip):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up_conv(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError("Decoder and skip spatial shapes do not match.")
        return self.reduce(self.mix(torch.cat((skip, x), dim=1)))


class ImprovedUNet2D(nn.Module):
    """A documented 2D adaptation of Isensee et al., arXiv:1802.10508.

    Uses residual context blocks, strided downsampling, instance normalization,
    nearest-neighbor upsampling, and summed multi-scale segmentation logits.
    This is not an exact reproduction of the paper's 3D training system.
    """

    def __init__(self, in_channels=1, num_classes=6, base_channels=32, dropout=0.3):
        super().__init__()
        if min(in_channels, num_classes, base_channels) < 1:
            raise ValueError("Channel counts must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1).")
        self.in_channels = in_channels
        channels = [base_channels * 2**level for level in range(5)]
        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1, bias=False)
        self.contexts = nn.ModuleList(
            ResidualContext2D(c, dropout) for c in channels
        )
        self.down = nn.ModuleList(
            PreActConv2D(channels[i], channels[i + 1], stride=2)
            for i in range(4)
        )
        self.decoder = nn.ModuleList(
            Localization2D(channels[i + 1], channels[i])
            for i in reversed(range(4))
        )
        # Heads at full, half, and quarter resolution. No softmax here.
        self.segmentation_heads = nn.ModuleList(
            nn.Conv2d(channels[i], num_classes, 1) for i in range(3)
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Expected input shape [B, in_channels, H, W].")
        height, width = x.shape[-2:]
        if min(height, width) < 16 or height % 16 or width % 16:
            raise ValueError("Input dimensions must be multiples of 16.")
        if (height // 16) * (width // 16) <= 1:
            raise ValueError("Instance normalization needs more than one bottleneck pixel.")
        x = self.contexts[0](self.stem(x))
        skips = [x]
        for down, context in zip(self.down, self.contexts[1:]):
            x = context(down(x))
            skips.append(x)
        logits = None
        for level, decoder in zip(reversed(range(4)), self.decoder):
            x = decoder(x, skips[level])
            if level <= 2:
                current = self.segmentation_heads[level](x)
                if logits is None:
                    logits = current
                else:
                    logits = current + torch.nn.functional.interpolate(
                        logits, size=current.shape[-2:], mode="nearest"
                    )
        return logits
