import torch
import torch.nn as nn
import torch.nn.functional as F

# STANDARD U-NET

# Adapted from Ronneberger et al. (2015).
# Uses padding=1 to preserve spatial dimensions.
class DoubleConv(nn.Module):
    """Applies two 3x3 convolutions, each followed by ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        # Second convolution layer
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x

class Down(nn.Module):
    """Downsamples using max pooling followed by double convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsamples and combines encoder features using a skip connection."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # Upsample the decoder feature map
        x1 = self.up(x1)

        # Combine with encoder feature map
        x = torch.cat([x2, x1], dim=1)

        # Apply two convolutions
        return self.conv(x)

class OutConv(nn.Module):
    """Applies a 1x1 convolution to produce segmentation class scores."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Convert feature channels into the number of segmentation classes
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)


# IMPROVED 2D U-NET
# Adapted from Isensee et al. (2018).
class FirstContext(nn.Module):
    """First encoder context block of the Improved 2D U-Net."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1,
            bias=False
        )

        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.norm3 = nn.InstanceNorm2d(out_channels)

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1,
            bias=False
        )

        self.conv3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1,
            bias=False
        )

    def forward(self, x):
        out = self.conv1(x)

        # Preserve initial features for residual addition
        residual = out

        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)

        out = self.dropout(out)

        out = self.norm2(out)
        out = self.activation(out)
        out = self.conv3(out)

        # Combine learned features with the residual
        out = out + residual

        # Preserve features for the decoder skip connection
        skip = self.activation(out)

        out = self.norm3(out)
        out = self.activation(out)

        return out, skip


class DownContext(nn.Module):
    """Downsampling context block of the Improved 2D U-Net."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Downsample using a stride-2 convolution
        self.down = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=2, padding=1,
            bias=False
        )

        # Convolutions for residual feature learning
        self.conv1 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1,
            bias=False
        )

        # Separate normalisation stages
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.norm3 = nn.InstanceNorm2d(out_channels)

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.3)


    def forward(self, x):
        # Downsample and save residual features
        out = self.down(x)
        residual = out

        # First convolution stage
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.dropout(out)

        # Second convolution stage
        out = self.norm2(out)
        out = self.activation(out)
        out = self.conv2(out)

        # Residual addition
        out = out + residual

        # Normalise and activate the output
        out = self.norm3(out)
        out = self.activation(out)

        return out

class UpscaleBlock(nn.Module):
    """Upsampling block adapted from Isensee et al. (2018)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        out = self.norm1(x)
        out = self.activation(out)

        out = self.upsample(out)
        out = self.conv(out)

        out = self.norm2(out)
        out = self.activation(out)

        return out

class LocalisationBlock(nn.Module):
    """Decoder localisation block adapted from Isensee et al. (2018)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm = nn.InstanceNorm2d(in_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        features = self.conv1(x)
        features = self.norm(features)
        features = self.activation(features)

        # Reduce feature channels for the next decoder stage
        out = self.conv2(features)

        return out, features
