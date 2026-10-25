"""
Network components and loss functions.

* ``ImprovedUNet3D`` - the 3D Improved U-Net of Isensee et al. (2018), the
  model under study.
* ``ImprovedUNet2D`` - the same architecture with 2D convolutions, applied
  slice by slice. It differs from the 3D model mainly in whether a kernel sees
  the neighbouring slices (it also has a third of the parameters and
  normalises each slice rather than each volume), so it isolates the effect
  of 3D context as far as an architecture can.
* ``UNet2D`` - a standard 2D U-Net (Ronneberger et al., 2015) applied slice by
  slice, the conventional baseline.
* ``DiceCELoss`` - soft multi-class Dice loss plus cross-entropy.

All components are pure PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Improved U-Net (Isensee et al., 2018), in 3D or 2D
# ---------------------------------------------------------------------------
# Every block takes ``dims`` (3 or 2) and picks the matching layer types, so
# the 2D and 3D models share one definition.

def conv(dims, *args, **kwargs):
    return (nn.Conv3d if dims == 3 else nn.Conv2d)(*args, **kwargs)


def norm_act(channels, dims=3):
    """Instance norm followed by leaky ReLU (slope 0.01), as in the paper."""
    norm = nn.InstanceNorm3d if dims == 3 else nn.InstanceNorm2d
    return nn.Sequential(norm(channels, affine=True), nn.LeakyReLU(0.01, inplace=True))


class ContextModule(nn.Module):
    """
    Pre-activation residual block: two 3x3(x3) convolutions with dropout
    in between. The block's input is added back by the caller.
    """

    def __init__(self, channels, dims=3, dropout=0.3):
        super().__init__()
        drop = nn.Dropout3d if dims == 3 else nn.Dropout2d
        self.block = nn.Sequential(
            norm_act(channels, dims),
            conv(dims, channels, channels, 3, padding=1),
            drop(dropout),
            norm_act(channels, dims),
            conv(dims, channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class EncoderLevel(nn.Module):
    """3x3(x3) convolution (stride 2 below the top level) + residual context."""

    def __init__(self, in_channels, out_channels, stride, dims=3):
        super().__init__()
        self.conv = conv(dims, in_channels, out_channels, 3, stride=stride, padding=1)
        self.context = ContextModule(out_channels, dims)

    def forward(self, x):
        x = self.conv(x)
        return x + self.context(x)


class UpsamplingModule(nn.Module):
    """Nearest-neighbour x2 upscale, then a 3x3(x3) conv that halves the features."""

    def __init__(self, in_channels, out_channels, dims=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv(dims, in_channels, out_channels, 3, padding=1),
            norm_act(out_channels, dims),
        )

    def forward(self, x):
        return self.block(x)


class LocalisationModule(nn.Module):
    """3x3(x3) conv on the concatenated features, then a 1x1(x1) conv halving them."""

    def __init__(self, in_channels, out_channels, dims=3):
        super().__init__()
        self.block = nn.Sequential(
            conv(dims, in_channels, in_channels, 3, padding=1),
            norm_act(in_channels, dims),
            conv(dims, in_channels, out_channels, 1),
            norm_act(out_channels, dims),
        )

    def forward(self, x):
        return self.block(x)


class ImprovedUNet(nn.Module):
    """
    Improved U-Net in ``dims`` = 3 or 2 dimensions.

    Encoder: five levels with ``base, 2*base, ..., 16*base`` features; each
    level is a (strided) convolution plus a residual context module.
    Decoder: upsampling modules, skip concatenation and localisation modules.
    Deep supervision: 1x1(x1) segmentation heads at the three finest decoder
    levels are upscaled and summed into the final logits.

    Input ``(N, in_channels, *spatial)`` with every spatial size divisible by
    16. Output logits ``(N, num_classes, *spatial)``.
    """

    def __init__(self, in_channels=1, num_classes=6, base=16, dims=3):
        super().__init__()
        f = [base * 2 ** i for i in range(5)]  # 16, 32, 64, 128, 256
        self.upscale = "trilinear" if dims == 3 else "bilinear"

        self.encoders = nn.ModuleList(
            [EncoderLevel(in_channels, f[0], stride=1, dims=dims)]
            + [EncoderLevel(f[i - 1], f[i], stride=2, dims=dims) for i in range(1, 5)]
        )
        self.encoder_out = norm_act(f[4], dims)

        # Decoder, coarse to fine.
        self.ups = nn.ModuleList([UpsamplingModule(f[i], f[i - 1], dims) for i in (4, 3, 2, 1)])
        self.locs = nn.ModuleList([LocalisationModule(2 * f[i], f[i], dims) for i in (3, 2, 1)])
        self.top = nn.Sequential(conv(dims, 2 * f[0], 2 * f[0], 3, padding=1),
                                 norm_act(2 * f[0], dims))

        # Segmentation heads at the 1/4, 1/2 and full-resolution decoder levels.
        self.seg_heads = nn.ModuleList([
            conv(dims, f[2], num_classes, 1),
            conv(dims, f[1], num_classes, 1),
            conv(dims, 2 * f[0], num_classes, 1),
        ])

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        x = self.encoder_out(skips.pop())

        seg = None
        for i, up in enumerate(self.ups):
            x = torch.cat([up(x), skips.pop()], dim=1)
            x = self.locs[i](x) if i < 3 else self.top(x)
            if i >= 1:  # deep supervision from the three finest levels
                head = self.seg_heads[i - 1](x)
                seg = head if seg is None else head + F.interpolate(
                    seg, scale_factor=2, mode=self.upscale, align_corners=False)
        return seg


class ImprovedUNet3D(ImprovedUNet):
    """3D Improved U-Net on whole volumes ``(N, C, D, H, W)``."""

    def __init__(self, in_channels=1, num_classes=6, base=16):
        super().__init__(in_channels, num_classes, base, dims=3)


class ImprovedUNet2D(ImprovedUNet):
    """The same network with 2D convolutions, on axial slices ``(N, C, H, W)``."""

    def __init__(self, in_channels=1, num_classes=6, base=16):
        super().__init__(in_channels, num_classes, base, dims=2)


# ---------------------------------------------------------------------------
# 2D U-Net baseline (Ronneberger et al., 2015)
# ---------------------------------------------------------------------------

class DoubleConv2D(nn.Sequential):
    """(3x3 conv -> batch norm -> ReLU) x 2."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet2D(nn.Module):
    """
    Standard 2D U-Net with four down/up-sampling steps.

    Input ``(N, in_channels, H, W)`` with H, W divisible by 16.
    Output logits ``(N, num_classes, H, W)``.
    """

    def __init__(self, in_channels=1, num_classes=6, base=32):
        super().__init__()
        f = [base * 2 ** i for i in range(5)]  # 32, 64, 128, 256, 512

        self.downs = nn.ModuleList([DoubleConv2D(in_channels, f[0])]
                                   + [DoubleConv2D(f[i - 1], f[i]) for i in range(1, 5)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(f[i], f[i - 1], 2, stride=2)
                                  for i in (4, 3, 2, 1)])
        self.decs = nn.ModuleList([DoubleConv2D(2 * f[i - 1], f[i - 1]) for i in (4, 3, 2, 1)])
        self.head = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.ups, self.decs):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceCELoss(nn.Module):
    """
    ``ce_weight * CrossEntropy + (1 - mean soft Dice over foreground classes)``.

    Soft Dice is computed over the whole batch for each class, which keeps the
    loss stable when a small class (e.g. prostate) is absent from a slice.
    Works for both 2D ``(N, C, H, W)`` and 3D ``(N, C, D, H, W)`` logits.
    """

    def __init__(self, ce_weight=1.0, include_background=False, smooth=1e-5):
        super().__init__()
        self.ce_weight = ce_weight
        self.first_class = 0 if include_background else 1
        self.smooth = smooth

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target)

        probs = logits.float().softmax(dim=1)
        one_hot = F.one_hot(target, logits.shape[1]).movedim(-1, 1).float()
        dims = (0,) + tuple(range(2, logits.ndim))
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)

        return self.ce_weight * ce + 1 - dice[self.first_class:].mean()


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
