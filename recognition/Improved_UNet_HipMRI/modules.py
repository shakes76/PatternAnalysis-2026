
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two convolutional layers used in the baseline U-Net."""

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


class Down(nn.Module):
    """Downsample features and apply two convolutional layers."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):
    """Upsample features and combine them with an encoder skip connection."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            in_channels // 2 + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        # Restore spatial resolution and combine encoder features.
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """
    Baseline U-Net for MRI image segmentation.

    The encoder extracts features at different resolutions.
    The decoder uses skip connections to recover spatial detail.
    """

    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024, 512, 512)
        self.up2 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up4 = Up(128, 64, 64)

        # Predict a class score for each pixel.
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Extract features and save them for skip connections.
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Upsample and combine features from the encoder.
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


class ResidualBlock(nn.Module):
    """
    Residual convolutional block used in the Improved U-Net.

    A shortcut connection adds the input features to the output
    of the convolutional layers. This helps information and
    gradients pass through the network.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        # Match channel dimensions when the input and output differ.
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Add the shortcut features before applying the activation.
        return self.relu(self.conv(x) + self.shortcut(x))


class ResidualDown(nn.Module):
    """Downsample features using max pooling and a residual block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualUp(nn.Module):
    """Upsample features and process skip connections with a residual block."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )

        self.conv = ResidualBlock(
            in_channels // 2 + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        # Combine upsampled features with encoder features.
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class ImprovedUNet(nn.Module):
    """
    Improved U-Net using residual convolutional blocks.

    The overall encoder-decoder structure is the same as the
    baseline U-Net, but residual blocks replace standard
    convolutional blocks to improve feature learning.
    """

    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()

        # Encoder with residual blocks
        self.inc = ResidualBlock(in_channels, 64)
        self.down1 = ResidualDown(64, 128)
        self.down2 = ResidualDown(128, 256)
        self.down3 = ResidualDown(256, 512)
        self.down4 = ResidualDown(512, 1024)

        # Decoder with residual blocks
        self.up1 = ResidualUp(1024, 512, 512)
        self.up2 = ResidualUp(512, 256, 256)
        self.up3 = ResidualUp(256, 128, 128)
        self.up4 = ResidualUp(128, 64, 64)

        # Final pixel classification
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

