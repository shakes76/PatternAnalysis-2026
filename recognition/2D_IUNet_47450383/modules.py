"""
modules.py - Model definitions for HipMRI 2D segmentation.

Contains two networks:
    UNet2D          Baseline 4-level U-Net (BatchNorm + ReLU, transposed-conv
                    upsampling).
    ImprovedUNet2D  Lighter U-Net variant using InstanceNorm + LeakyReLU,
                    residual context modules with dropout, strided-conv
                    downsampling, nearest-neighbour upsampling and deep
                    supervision (multi-scale segmentation outputs summed).

Both models take (B, n_channels, H, W) input and return raw, un-normalised
logits of shape (B, num_classes, H, W).

Course:  COMP3710 - 2.2 Segmentation (HipMRI)
Author:  <your name / student number>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_match(x, skip):
    """Zero-pad `x` so its spatial size matches `skip`.

    Needed when the input has odd dimensions, where upsampling a pooled
    feature map can come out a pixel smaller than its skip connection.
    """
    dh, dw = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
    if dh != 0 or dw != 0:
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
    return x


# =========================================================================== #
# Baseline U-Net
# =========================================================================== #
class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),  # bias redundant before BN
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Encoder step: 2x2 max-pool (halves H, W) then DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Decoder step: upsample, concatenate the skip connection, DoubleConv.

    `in_ch` is the channel count of the incoming (deeper) feature map. The
    transposed conv halves it, and concatenating the skip (in_ch // 2 channels)
    brings it back to `in_ch` for the DoubleConv.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = pad_to_match(self.up(x), skip)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet2D(nn.Module):
    """Standard 4-level U-Net.

    Args:
        n_channels: Number of input channels (1 for greyscale MRI).
        num_classes: Number of output classes.
        base: Number of filters in the first level (doubles each level).

    Outputs raw logits of shape (B, num_classes, H, W).
    """

    def __init__(self, n_channels, num_classes=6, base=64):
        super().__init__()
        b = base
        # Encoder
        self.inc = DoubleConv(n_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16)  # bottleneck
        # Decoder
        self.up1 = Up(b * 16, b * 8)
        self.up2 = Up(b * 8, b * 4)
        self.up3 = Up(b * 4, b * 2)
        self.up4 = Up(b * 2, b)
        # 1x1 conv maps features to per-class logits
        self.out = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder (keep each level's output for the skip connections)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)


# =========================================================================== #
# Improved U-Net
# =========================================================================== #
class ContextModule(nn.Module):
    """Pre-activation residual block with dropout:
    x + [IN -> LeakyReLU -> Conv3x3 -> Dropout -> IN -> LeakyReLU -> Conv3x3].
    """

    def __init__(self, ch, p=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(ch), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.Dropout2d(p),
            nn.InstanceNorm2d(ch), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)  # residual connection


def conv_in_act(in_ch, out_ch, k=3, stride=1):
    """Conv -> InstanceNorm -> LeakyReLU with 'same'-style padding.

    Use stride=2 to downsample, or k=1 for a channel-reducing 1x1 conv.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k // 2, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(0.01, inplace=True),
    )


class ImprovedDown(nn.Module):
    """Encoder step: stride-2 conv (halves H, W) then a ContextModule."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = conv_in_act(in_ch, out_ch, 3, stride=2)
        self.ctx = ContextModule(out_ch)

    def forward(self, x):
        return self.ctx(self.down(x))


class ImprovedUp(nn.Module):
    """Decoder step: upsample + conv, concatenate skip, then localisation conv(s).

    Args:
        in_ch: Channels of the incoming (deeper) feature map.
        out_ch: Channels after upsampling (equals the skip's channel count).
        final: If True, keep the concatenated width (out_ch * 2) instead of
            reducing to out_ch, so the final seg head sees the richer features.
    """

    def __init__(self, in_ch, out_ch, final=False):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv_in_act(in_ch, out_ch, 3),
        )
        if final:
            self.loc = conv_in_act(out_ch * 2, out_ch * 2, 3)
        else:
            self.loc = nn.Sequential(
                conv_in_act(out_ch * 2, out_ch * 2, 3),
                conv_in_act(out_ch * 2, out_ch, 1),  # 1x1 conv reduces channels
            )

    def forward(self, x, skip):
        x = pad_to_match(self.up(x), skip)
        return self.loc(torch.cat([skip, x], dim=1))


class ImprovedUNet2D(nn.Module):
    """U-Net variant with deep supervision-style multi-scale output.

    Segmentation maps from the last three decoder levels are combined: each
    coarser map is upsampled and added to the next finer one, so the final
    logits aggregate information from several scales.

    Args:
        n_channels: Number of input channels (1 for greyscale MRI).
        num_classes: Number of output classes.
        base: Number of filters in the first level (doubles each level).

    Outputs raw logits of shape (B, num_classes, H, W).
    """

    def __init__(self, n_channels, num_classes=6, base=16):
        super().__init__()
        b = base
        # Encoder
        self.inc = nn.Sequential(conv_in_act(n_channels, b, 3), ContextModule(b))
        self.down1 = ImprovedDown(b, b * 2)
        self.down2 = ImprovedDown(b * 2, b * 4)
        self.down3 = ImprovedDown(b * 4, b * 8)
        self.down4 = ImprovedDown(b * 8, b * 16)  # bottleneck

        # Decoder
        self.up1 = ImprovedUp(b * 16, b * 8)
        self.up2 = ImprovedUp(b * 8, b * 4)
        self.up3 = ImprovedUp(b * 4, b * 2)
        self.up4 = ImprovedUp(b * 2, b, final=True)  # outputs b * 2 channels

        # 1x1 segmentation heads for the last three decoder levels
        self.seg2 = nn.Conv2d(b * 4, num_classes, 1)  # from up2
        self.seg3 = nn.Conv2d(b * 2, num_classes, 1)  # from up3
        self.seg4 = nn.Conv2d(b * 2, num_classes, 1)  # from up4

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        d = self.up1(x5, x4)
        d2 = self.up2(d, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        # Multi-scale aggregation: upsample the coarser logits and add the next
        s = self.seg2(d2)
        s = F.interpolate(s, size=d3.shape[2:], mode="nearest") + self.seg3(d3)
        s = F.interpolate(s, size=d4.shape[2:], mode="nearest") + self.seg4(d4)
        return s