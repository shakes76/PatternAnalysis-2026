"""
modules.py - Network components for HipMRI prostate segmentation.

Contains:
    * ImprovedUNet3D - the proposed model, a 3D Improved U-Net following
      Isensee et al. (2018), "Brain tumor segmentation and radiomics survival
      prediction: Contribution to the BRATS 2017 challenge".
    * UNet2D         - the baseline, a standard 2D U-Net following
      Ronneberger et al. (2015), applied slice-by-slice to the same volumes.
    * DiceCELoss     - soft Dice + cross-entropy loss for class imbalance.

All tensors follow PyTorch's channels-first layout:
    3D: (batch, channels, depth, height, width)
    2D: (batch, channels, height, width)

Only PyTorch is used in this file (no NumPy), as required by the brief.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3D Improved U-Net building blocks
# ---------------------------------------------------------------------------

def _norm_act(channels: int) -> nn.Sequential:
    """Instance norm followed by LeakyReLU (slope 0.01), as in Isensee et al."""
    return nn.Sequential(
        nn.InstanceNorm3d(channels, affine=True),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
    )


class ContextModule(nn.Module):
    """
    Residual 'context module' used on the encoder path.

    Pre-activation residual block:
        x -> [norm, act, conv3] -> dropout -> [norm, act, conv3] -> + x

    The residual addition lets gradients flow through deep networks and lets
    each module learn a refinement of its input instead of a full mapping.
    """

    def __init__(self, channels: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            _norm_act(channels),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout3d(p=dropout),
            _norm_act(channels),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DownConv(nn.Module):
    """Stride-2 3x3x3 convolution that halves spatial size and changes channels."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            _norm_act(in_ch),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsamplingModule(nn.Module):
    """
    Upsampling module: nearest-neighbour upscale (x2) then a 3x3x3 conv that
    halves the number of feature maps. Isensee et al. use this instead of a
    transposed convolution to avoid checkerboard artefacts.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm_act(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LocalisationModule(nn.Module):
    """
    Localisation module: 3x3x3 conv (recombines the concatenated skip and
    upsampled features) followed by a 1x1x1 conv that halves the channels.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            _norm_act(in_ch),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            _norm_act(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImprovedUNet3D(nn.Module):
    """
    3D Improved U-Net (Isensee et al., 2018).

    Encoder: 5 levels with base * (1, 2, 4, 8, 16) feature maps. Each level is
             a (strided) conv followed by a residual context module.
    Decoder: upsampling module -> concatenate skip -> localisation module.
    Deep supervision: 1x1x1 segmentation layers on the outputs of the two
             lowest decoder levels are upscaled and summed into the final
             prediction, so coarse levels receive a direct training signal.

    Args:
        in_channels:  number of input image channels (1 for MRI).
        num_classes:  number of segmentation classes (incl. background).
        base:         feature maps at the top level (16 in the paper).
        dropout:      dropout probability inside context modules.

    Returns raw logits of shape (B, num_classes, D, H, W).
    Input D, H, W must be divisible by 16.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6,
                 base: int = 16, dropout: float = 0.3):
        super().__init__()
        f = [base * m for m in (1, 2, 4, 8, 16)]

        # ----- Encoder -----
        self.stem = nn.Conv3d(in_channels, f[0], kernel_size=3, padding=1, bias=False)
        self.ctx1 = ContextModule(f[0], dropout)
        self.down2, self.ctx2 = DownConv(f[0], f[1]), ContextModule(f[1], dropout)
        self.down3, self.ctx3 = DownConv(f[1], f[2]), ContextModule(f[2], dropout)
        self.down4, self.ctx4 = DownConv(f[2], f[3]), ContextModule(f[3], dropout)
        self.down5, self.ctx5 = DownConv(f[3], f[4]), ContextModule(f[4], dropout)

        # ----- Decoder -----
        self.up4 = UpsamplingModule(f[4], f[3])
        self.loc4 = LocalisationModule(f[3] * 2, f[3])
        self.up3 = UpsamplingModule(f[3], f[2])
        self.loc3 = LocalisationModule(f[2] * 2, f[2])
        self.up2 = UpsamplingModule(f[2], f[1])
        self.loc2 = LocalisationModule(f[1] * 2, f[1])
        self.up1 = UpsamplingModule(f[1], f[0])
        # Top level: 3x3x3 conv on the concatenation (2 * f0 channels)
        self.final_conv = nn.Sequential(
            nn.Conv3d(f[0] * 2, f[0] * 2, kernel_size=3, padding=1, bias=False),
            _norm_act(f[0] * 2),
        )

        # ----- Deep supervision segmentation layers -----
        self.seg3 = nn.Conv3d(f[2], num_classes, kernel_size=1)
        self.seg2 = nn.Conv3d(f[1], num_classes, kernel_size=1)
        self.seg1 = nn.Conv3d(f[0] * 2, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder (keep outputs for skip connections)
        s1 = self.ctx1(self.stem(x))
        s2 = self.ctx2(self.down2(s1))
        s3 = self.ctx3(self.down3(s2))
        s4 = self.ctx4(self.down4(s3))
        b = self.ctx5(self.down5(s4))

        # Decoder
        d4 = self.loc4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.loc3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.loc2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.final_conv(torch.cat([self.up1(d2), s1], dim=1))

        # Deep supervision: upscale coarse segmentations and sum
        seg = self.seg3(d3)
        seg = F.interpolate(seg, scale_factor=2, mode="trilinear", align_corners=False)
        seg = seg + self.seg2(d2)
        seg = F.interpolate(seg, scale_factor=2, mode="trilinear", align_corners=False)
        seg = seg + self.seg1(d1)
        return seg


# ---------------------------------------------------------------------------
# 2D U-Net baseline
# ---------------------------------------------------------------------------

class DoubleConv2D(nn.Module):
    """Two (3x3 conv -> BatchNorm -> ReLU) layers, the basic U-Net block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet2D(nn.Module):
    """
    Standard 2D U-Net (Ronneberger et al., 2015) used as the baseline.

    4 down-sampling steps (max-pool) and 4 up-sampling steps (transposed conv)
    with skip connections. It sees one axial slice at a time, so it has no
    information about neighbouring slices - this is exactly the limitation the
    3D model is meant to address.

    Returns raw logits of shape (B, num_classes, H, W).
    Input H, W must be divisible by 16.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6, base: int = 32):
        super().__init__()
        f = [base * m for m in (1, 2, 4, 8, 16)]
        self.enc1 = DoubleConv2D(in_channels, f[0])
        self.enc2 = DoubleConv2D(f[0], f[1])
        self.enc3 = DoubleConv2D(f[1], f[2])
        self.enc4 = DoubleConv2D(f[2], f[3])
        self.bottleneck = DoubleConv2D(f[3], f[4])
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv2D(f[4], f[3])
        self.up3 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv2D(f[3], f[2])
        self.up2 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv2D(f[2], f[1])
        self.up1 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv2D(f[1], f[0])
        self.head = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceCELoss(nn.Module):
    """
    Soft Dice loss + cross-entropy, works for both 2D and 3D logits.

    Cross-entropy gives stable per-voxel gradients; the Dice term is computed
    per class and averaged, so a tiny structure like the prostate (<2% of
    voxels) counts as much as the large body/background classes.

    Args:
        dice_weight: weight on the Dice term (CE weight is 1 - dice_weight).
        smooth:      small constant to avoid division by zero.

    forward(logits, target):
        logits: (B, C, *spatial) raw network outputs
        target: (B, *spatial) integer class labels
    """

    def __init__(self, dice_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        ce = F.cross_entropy(logits, target)

        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes).movedim(-1, 1).to(probs.dtype)
        dims = (0,) + tuple(range(2, logits.ndim))     # sum over batch + space
        intersection = (probs * one_hot).sum(dims)
        denom = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        dice_loss = 1 - dice.mean()

        return self.dice_weight * dice_loss + (1 - self.dice_weight) * ce


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (reported in the README resource table)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
