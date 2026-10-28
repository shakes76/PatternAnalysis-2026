"""
Model components for HipMRI 3D prostate segmentation.

Contains:
    - ImprovedUNet3D: 3D Improved U-Net (Isensee et al., 2017) with pre-activation residual context
      modules, localisation modules and deep supervision.
    - DiceLoss / CombinedLoss: losses for the heavily imbalanced prostate class (0.1-0.3% of voxels).

Only PyTorch is used here (no NumPy), as required by the assessment spec.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Isensee et al. use leaky ReLU with a negative slope of 1e-2 throughout
NEGATIVE_SLOPE = 1e-2


# ==========================================
# Improved UNet Building Blocks
# ==========================================
class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # Convolution followed by Instance Normalization (batches of 1-2 volumes are too small for BatchNorm) and leaky ReLU
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(NEGATIVE_SLOPE, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ContextModule(nn.Module):
    def __init__(self, channels, dropout=0.3):
        super().__init__()
        # Pre-activation residual unit: (norm -> act -> conv) twice, with dropout in between
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(NEGATIVE_SLOPE, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout3d(dropout),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(NEGATIVE_SLOPE, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        # Residual connection: the module only has to learn a correction to its input
        return x + self.block(x)


class EncoderLevel(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # Encoder step: a stride-2 3x3x3 convolution halves the spatial dimensions (instead of MaxPool),
        # then a residual context module extracts features at that resolution
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.context = ContextModule(out_channels)

    def forward(self, x):
        return self.context(self.conv(x))


class UpsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Decoder step: upscale by repeating voxels (avoids transposed convolution checkerboard artefacts),
        # then a 3x3x3 convolution halves the number of feature maps
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvNormAct(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class LocalisationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Recombines the concatenated skip and upsampled features: 3x3x3 conv, then 1x1x1 conv halving the feature maps
        self.block = nn.Sequential(
            ConvNormAct(in_channels, in_channels),
            ConvNormAct(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)


class ImprovedUNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=6, base_filters=16):
        super().__init__()
        f = base_filters
        # Encoder: 5 levels, 16 -> 32 -> 64 -> 128 -> 256 filters, each level after the first at half the resolution
        self.enc1 = EncoderLevel(n_channels, f, stride=1)
        self.enc2 = EncoderLevel(f, 2 * f, stride=2)
        self.enc3 = EncoderLevel(2 * f, 4 * f, stride=2)
        self.enc4 = EncoderLevel(4 * f, 8 * f, stride=2)
        self.enc5 = EncoderLevel(8 * f, 16 * f, stride=2)

        # Decoder: upsample, concatenate with the skip connection, then localise
        self.up4 = UpsamplingModule(16 * f, 8 * f)
        self.loc4 = LocalisationModule(16 * f, 8 * f)
        self.up3 = UpsamplingModule(8 * f, 4 * f)
        self.loc3 = LocalisationModule(8 * f, 4 * f)
        self.up2 = UpsamplingModule(4 * f, 2 * f)
        self.loc2 = LocalisationModule(4 * f, 2 * f)
        self.up1 = UpsamplingModule(2 * f, f)
        self.final = ConvNormAct(2 * f, 2 * f)

        # Deep supervision: segmentation layers at 1/4, 1/2 and full resolution, upscaled and summed
        self.seg3 = nn.Conv3d(4 * f, n_classes, kernel_size=1)
        self.seg2 = nn.Conv3d(2 * f, n_classes, kernel_size=1)
        self.seg1 = nn.Conv3d(2 * f, n_classes, kernel_size=1)
        self.upscale = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        # Passes data down the encoder, saving each level's output for the skip connections
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        # Passes data up the decoder, merging it with the saved encoder outputs at each step to recover spatial details
        d4 = self.loc4(torch.cat([x4, self.up4(x5)], dim=1))
        d3 = self.loc3(torch.cat([x3, self.up3(d4)], dim=1))
        d2 = self.loc2(torch.cat([x2, self.up2(d3)], dim=1))
        d1 = self.final(torch.cat([x1, self.up1(d2)], dim=1))

        # Deep supervision: coarse segmentations are added into the finer ones, so gradients reach the
        # lower decoder levels directly and the coarse levels learn to produce usable segmentations
        seg = self.upscale(self.seg3(d3)) + self.seg2(d2)
        return self.upscale(seg) + self.seg1(d1)


# ==========================================
# Losses
# ==========================================
class DiceLoss(nn.Module):
    """Soft multi-class Dice loss on softmax probabilities, averaged over classes.

    Every class counts equally regardless of size, so the tiny prostate class is not drowned out by
    the background the way it is with cross-entropy alone. Works for 2D (N, C, H, W) and 3D (N, C, D, H, W).
    """

    def __init__(self, n_classes, smooth=1.0):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        # Float32 so the sums over ~1M voxels cannot overflow under mixed precision
        probs = F.softmax(logits.float(), dim=1)
        # (N, *spatial) integer labels -> (N, C, *spatial) one-hot to match the probabilities
        one_hot = F.one_hot(targets, self.n_classes).movedim(-1, 1).float()

        dims = (0,) + tuple(range(2, logits.ndim))  # sum over batch and voxels, keep classes separate
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice.mean()


class CombinedLoss(nn.Module):
    """Cross-entropy (stable voxel-wise gradients) + Dice (class-balanced overlap), same as the 2D baseline."""

    def __init__(self, n_classes, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(n_classes)
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)
