"""Independent baseline and Isensee et al. (2018)-inspired 2D U-Net models."""

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Two same-padding 3x3 convolutions, each followed by ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, values):
        return self.layers(values)


class UNet(nn.Module):
    """Four-downsampling U-Net returning [B,num_classes,H,W] raw logits.

    Same-padding convolutions preserve the padded sample dimensions. Input
    height and width must be multiples of 16, matching the preprocessing API.
    Channel width is a configurable resource choice; the default is 16.
    """

    required_multiple = 16

    def __init__(self, in_channels=1, num_classes=6, base_channels=16):
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("num_classes", num_classes),
            ("base_channels", base_channels),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        width = base_channels

        self.enc1 = DoubleConv(in_channels, width)
        self.enc2 = DoubleConv(width, width * 2)
        self.enc3 = DoubleConv(width * 2, width * 4)
        self.enc4 = DoubleConv(width * 4, width * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(width * 8, width * 16)

        self.up4 = nn.ConvTranspose2d(width * 16, width * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(width * 16, width * 8)
        self.up3 = nn.ConvTranspose2d(width * 8, width * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(width * 8, width * 4)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(width * 2, width)
        self.output = nn.Conv2d(width, num_classes, kernel_size=1)

    def forward(self, image):
        if image.ndim != 4:
            raise ValueError("U-Net input must have shape [B,C,H,W].")
        if image.shape[0] == 0 or image.shape[1] != self.in_channels:
            raise ValueError(f"Expected a nonempty batch with {self.in_channels} channels.")
        height, width = image.shape[-2:]
        if min(height, width) < self.required_multiple or (
            height % self.required_multiple or width % self.required_multiple
        ):
            raise ValueError("Input height and width must be >=16 and multiples of 16.")
        if not image.is_floating_point():
            raise TypeError("U-Net input must be floating-point image values.")

        skip1 = self.enc1(image)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        skip4 = self.enc4(self.pool(skip3))
        encoded = self.bottleneck(self.pool(skip4))

        decoded4 = self.dec4(torch.cat((self.up4(encoded), skip4), dim=1))
        decoded3 = self.dec3(torch.cat((self.up3(decoded4), skip3), dim=1))
        decoded2 = self.dec2(torch.cat((self.up2(decoded3), skip2), dim=1))
        decoded1 = self.dec1(torch.cat((self.up1(decoded2), skip1), dim=1))
        return self.output(decoded1)


class PreActConv(nn.Module):
    """Instance normalization and LeakyReLU before a feature convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.norm = nn.InstanceNorm2d(
            in_channels, affine=True, track_running_stats=False
        )
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )

    def forward(self, values):
        return self.conv(self.activation(self.norm(values)))


class ResidualContext(nn.Module):
    """Two preactivation convolutions with channel dropout and an identity path."""

    def __init__(self, channels):
        super().__init__()
        self.first = PreActConv(channels, channels)
        self.dropout = nn.Dropout2d(p=0.3)
        self.second = PreActConv(channels, channels)

    def forward(self, values):
        return values + self.second(self.dropout(self.first(values)))


class NearestUpConv(nn.Module):
    """Double each spatial dimension, then reduce the feature channel count."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.features = PreActConv(in_channels, out_channels)

    def forward(self, values):
        upsampled = torch.nn.functional.interpolate(
            values, scale_factor=2, mode="nearest"
        )
        return self.features(upsampled)


class LocalizationBlock(nn.Module):
    """Mix concatenated features spatially, then halve channels with a 1x1 kernel."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spatial = PreActConv(in_channels, in_channels)
        self.channel = PreActConv(in_channels, out_channels, kernel_size=1)

    def forward(self, values):
        return self.channel(self.spatial(values))


class ImprovedUNet(nn.Module):
    """2D adaptation of the residual U-Net in Isensee et al., arXiv:1802.10508.

    Five context scales feed a localization decoder. Its quarter-, half-, and
    full-resolution segmentation heads are progressively added into one raw
    logits tensor. The shared loss receives this tensor; there are no auxiliary
    losses. Channel-wise Dropout2d, affine instance normalization, and three
    heads are explicit choices for this 2D implementation, not a reproduction
    of the paper's complete 3D training recipe.
    """

    required_multiple = 16

    def __init__(self, in_channels=1, num_classes=6, base_channels=16):
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("num_classes", num_classes),
            ("base_channels", base_channels),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        width = base_channels

        self.stem = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.context1 = ResidualContext(width)
        self.down2 = PreActConv(width, width * 2, stride=2)
        self.context2 = ResidualContext(width * 2)
        self.down3 = PreActConv(width * 2, width * 4, stride=2)
        self.context3 = ResidualContext(width * 4)
        self.down4 = PreActConv(width * 4, width * 8, stride=2)
        self.context4 = ResidualContext(width * 8)
        self.down5 = PreActConv(width * 8, width * 16, stride=2)
        self.context5 = ResidualContext(width * 16)

        self.up4 = NearestUpConv(width * 16, width * 8)
        self.localize4 = LocalizationBlock(width * 16, width * 8)
        self.up3 = NearestUpConv(width * 8, width * 4)
        self.localize3 = LocalizationBlock(width * 8, width * 4)
        self.up2 = NearestUpConv(width * 4, width * 2)
        self.localize2 = LocalizationBlock(width * 4, width * 2)
        self.up1 = NearestUpConv(width * 2, width)
        self.localize1 = LocalizationBlock(width * 2, width)

        self.head_quarter = nn.Conv2d(width * 4, num_classes, kernel_size=1)
        self.head_half = nn.Conv2d(width * 2, num_classes, kernel_size=1)
        self.head_full = nn.Conv2d(width, num_classes, kernel_size=1)

    def _validate_input(self, image):
        if image.ndim != 4:
            raise ValueError("Improved U-Net input must have shape [B,C,H,W].")
        if image.shape[0] == 0 or image.shape[1] != self.in_channels:
            raise ValueError(f"Expected a nonempty batch with {self.in_channels} channels.")
        height, width = image.shape[-2:]
        if min(height, width) < self.required_multiple or (
            height % self.required_multiple or width % self.required_multiple
        ):
            raise ValueError("Input height and width must be >=16 and multiples of 16.")
        if (height // self.required_multiple) * (width // self.required_multiple) < 2:
            raise ValueError(
                "Improved U-Net needs at least two bottleneck spatial elements "
                "for InstanceNorm2d; use e.g. 16x32 or 32x32 instead of 16x16."
            )
        if not image.is_floating_point():
            raise TypeError("Improved U-Net input must be floating-point image values.")

    def forward_heads(self, image):
        """Return raw quarter/half/full logits before progressive fusion."""
        self._validate_input(image)
        skip1 = self.context1(self.stem(image))
        skip2 = self.context2(self.down2(skip1))
        skip3 = self.context3(self.down3(skip2))
        skip4 = self.context4(self.down4(skip3))
        encoded = self.context5(self.down5(skip4))

        decoded4 = self.localize4(torch.cat((self.up4(encoded), skip4), dim=1))
        decoded3 = self.localize3(torch.cat((self.up3(decoded4), skip3), dim=1))
        decoded2 = self.localize2(torch.cat((self.up2(decoded3), skip2), dim=1))
        decoded1 = self.localize1(torch.cat((self.up1(decoded2), skip1), dim=1))
        return {
            "quarter": self.head_quarter(decoded3),
            "half": self.head_half(decoded2),
            "full": self.head_full(decoded1),
        }

    def forward(self, image):
        heads = self.forward_heads(image)
        half = heads["half"] + torch.nn.functional.interpolate(
            heads["quarter"], size=heads["half"].shape[-2:], mode="nearest"
        )
        return heads["full"] + torch.nn.functional.interpolate(
            half, size=heads["full"].shape[-2:], mode="nearest"
        )


def build_model(name, **config):
    """Construct a named architecture without changing either model's defaults."""
    constructors = {"baseline_unet": UNet, "improved_unet": ImprovedUNet}
    if name not in constructors:
        raise ValueError(
            f"Unknown model {name!r}; expected baseline_unet or improved_unet."
        )
    return constructors[name](**config)
