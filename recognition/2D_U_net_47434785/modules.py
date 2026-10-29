import torch
from torch import nn
from torch.nn import functional as F

"""Standard 2D U-Net baseline for COMP3710 HipMRI segmentation.

Architecture reference:
    Ronneberger, O., Fischer, P., & Brox, T. (2015).
    "U-Net: Convolutional Networks for Biomedical Image Segmentation."
    https://arxiv.org/abs/1505.04597

Implementation attribution:
    PyTorch code written with ChatGPT assistance based on Figure 1 and
    Section 2 of Ronneberger et al. (2015).

    Differs from reference architecture by adding batch normalisation 
    and Billinear interpolation instead of Transposed convolution.

"""
class DoubleConv(nn.Module):
    """Two 3x3 convolution -> batch normalisation -> ReLU operations.

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return extracted features, keeping the same height and width."""
        return self.layers(x)


class Encode(nn.Module):
    """One U-Net encoder stage: max pooling followed by double convolution.

    Args:
        in_channels: Feature channels before downsampling.
        out_channels: Feature channels after double convolution.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return features with half the input height and width."""
        return self.layers(x)


class Decode(nn.Module):
    """One U-Net decoder stage: upsample, concatenate skip features, convolve.

    Args:
        in_channels: Channels arriving from the previous decoder stage.
        skip_channels: Channels in the corresponding encoder skip tensor.
        out_channels: Channels produced by this decoder stage.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Billinear Upsample of x to skip_channels size, concatenate, and return refined features."""
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """Baseline 2D U-Net (Ronneberger et al., 2015), adapted for HipMRI.

    Uses five resolution levels, four pooling stages, four decoder stages,
    encoder-decoder skip connections and a final 1x1 output convolution.

    Args:
        in_channels: Number of input channels (1 for HipMRI).
        num_classes: Number of output segmentation classes (6 for HipMRI).
        base_channels: Number of feature channels at the first level (64).

    Returns from forward:
        Tensor of raw per-class logits with shape (B, num_classes, H, W).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6,
                 base_channels: int = 64):
        super().__init__()
        if in_channels < 1 or num_classes < 1 or base_channels < 1:
            raise ValueError("Channel counts must be positive integers.")

        c = base_channels
        # Encoder: features grow while spatial dimensions shrink.
        self.stem = DoubleConv(in_channels, c)
        self.down1 = Encode(c, 2 * c)
        self.down2 = Encode(2 * c, 4 * c)
        self.down3 = Encode(4 * c, 8 * c)
        self.down4 = Encode(8 * c, 16 * c)  # Bottleneck at 1/16 resolution.

        # Decoder: upsampling and concatenation with encoder skip features.
        self.up1 = Decode(16 * c, 8 * c, 8 * c)
        self.up2 = Decode(8 * c, 4 * c, 4 * c)
        self.up3 = Decode(4 * c, 2 * c, 2 * c)
        self.up4 = Decode(2 * c, c, c)

        # One output logit map per segmentation class.
        self.output_conv = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Segment a batch of MRI slices; return logits at input resolution."""
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.down4(s3)

        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, s0)
        return self.output_conv(x)


"""Improved U-NET Architecture (adapted from Isensee et al., 2018)
Improved baseline U-NET using pre-activation residual blocks, strided
convolutions, InstanceNorm, LeakyReLU, Dropout2d, decoder localisation
and multiscale deep supervision.
"""
class ResidualBlock2D(nn.Module):
    """Pre-activation residual block with instance normalisation and dropout.

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.
        dropout: Fraction of channels randomly dropped during training.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.dropout = nn.Dropout2d(p=dropout)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)

        # A 1x1 convolution matches channel counts when they differ.
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                                      kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return learned features plus the original input (residual connection)."""
        residual = self.shortcut(x)
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.dropout(x)
        x = self.conv2(self.act2(self.norm2(x)))
        return x + residual


class ImprovedEncode(nn.Module):
    """Downsample with a learned stride-2 convolution, then a residual block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, stride=2, padding=1,
                                    bias=False)
        self.context = ResidualBlock2D(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Halve height/width and extract features."""
        x = self.downsample(x)
        return self.context(x)


class ImprovedDecode(nn.Module):
    """Upsample and concatenate a skip connection, then localise features.

    The 2D localisation module adapts the 3x3 -> 1x1 convolution pattern
    of Isensee et al. (2018). The 1x1 convolution reduces channel count.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, 2 * out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(2 * out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Return localised decoder features at the skip connection's size."""
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class ImprovedUNet2D(nn.Module):
    """Isensee-inspired 2D U-Net with decoder localisation and deep supervision.

    The four decoder output heads produce logits at different resolutions.
    Their upsampled logits are summed; the existing training loss therefore
    propagates gradients to all heads without modifying train.py's loss.

    Forward output remains [B, num_classes, H, W], matching UNet2D.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6,
                 base_channels: int = 64):
        super().__init__()
        if in_channels < 1 or num_classes < 1 or base_channels < 1:
            raise ValueError("Channel counts must be positive integers.")

        c = base_channels
        # Original improved encoder remains unchanged.
        self.stem = ResidualBlock2D(in_channels, c)
        self.down1 = ImprovedEncode(c, 2 * c)
        self.down2 = ImprovedEncode(2 * c, 4 * c)
        self.down3 = ImprovedEncode(4 * c, 8 * c)
        self.down4 = ImprovedEncode(8 * c, 16 * c)

        # Localisation decoder: 3x3 convolution followed by 1x1 reduction.
        self.up1 = ImprovedDecode(16 * c, 8 * c, 8 * c)
        self.up2 = ImprovedDecode(8 * c, 4 * c, 4 * c)
        self.up3 = ImprovedDecode(4 * c, 2 * c, 2 * c)
        self.up4 = ImprovedDecode(2 * c, c, c)

        # Four segmentation heads for multi-scale deep supervision.
        self.seg_coarse = nn.Conv2d(8 * c, num_classes, kernel_size=1)
        self.seg_middle = nn.Conv2d(4 * c, num_classes, kernel_size=1)
        self.seg_fine = nn.Conv2d(2 * c, num_classes, kernel_size=1)
        self.output_conv = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return summed full-resolution logits for all six classes."""
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.down4(s3)

        x = self.up1(x, s3)
        coarse = self.seg_coarse(x)

        x = self.up2(x, s2)
        middle = self.seg_middle(x)

        x = self.up3(x, s1)
        fine = self.seg_fine(x)

        x = self.up4(x, s0)
        full = self.output_conv(x)

        # The same full-resolution loss supervises every intermediate head.
        target_size = full.shape[-2:]
        output = (
            full
            + F.interpolate(fine, size=target_size, mode="bilinear",
                            align_corners=False)
            + F.interpolate(middle, size=target_size, mode="bilinear",
                            align_corners=False)
            + F.interpolate(coarse, size=target_size, mode="bilinear",
                            align_corners=False)
        )
        return output
