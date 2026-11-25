"""
modules.py - 3D U-Net models and loss for HipMRI prostate MRI segmentation.

Author: Angus Slater (49615878)
Course: COMP3710 Pattern Recognition, The University of Queensland, 2026
Project: Improved 3D U-Net for prostate MRI segmentation (HipMRI)

Models:
- UNet2D (2D control): slice-wise 2D U-Net (Ronneberger et al., 2015, "U-Net:
  Convolutional Networks for Biomedical Image Segmentation", MICCAI). Same layout
  and widths as UNet3D with 2D layers; each axial slice is segmented independently,
  so it has no inter-slice context. Isolates the value of 3D context. ~1.9M params.
- UNet3D (baseline): Cicek et al., 2016, "3D U-Net: Learning Dense Volumetric
  Segmentation from Sparse Annotation" (MICCAI). Four 2x levels, max-pool down,
  transposed-conv up, DoubleConv blocks, 16-32-64-128-256 channels. ~5.6M params.
- ImprovedUNet3D (proposed): Isensee et al., 2018, "Brain Tumor Segmentation and
  Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge".
  Residual context modules with dropout, stride-2 conv downsampling, LeakyReLU,
  upsample+conv, localisation modules and deep supervision. ~9.5M params.

All three use InstanceNorm (batch size 2 makes BatchNorm statistics unreliable),
take and return whole volumes, and share the same loss and training setup, so
the architecture is the only difference between them.

Loss: DiceCELoss = cross-entropy + (1 - soft Dice averaged over classes);
soft Dice from Milletari et al., 2016, "V-Net" (3DV).

Input:  (B, 1, X, Y, Z), each spatial size divisible by 16 (we use 128x128x64).
Output: (B, 6, X, Y, Z) raw logits; softmax is applied in the loss and at inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two [3x3x3 conv -> InstanceNorm -> ReLU] layers: the basic U-Net block.

    Args:
        in_ch: input channels.
        out_ch: output channels.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            # bias=False: the norm layer subtracts the mean, which would cancel a bias anyway
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            # affine=True: learnable per-channel scale (gamma) and shift (beta) after normalising.
            # Instance (per-scan) norm: stats don't depend on the other scan in a batch of 2.
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """(B, in_ch, X, Y, Z) -> (B, out_ch, X, Y, Z); spatial size unchanged (padding=1)."""
        return self.block(x)


class UNet3D(nn.Module):
    """Baseline 3D U-Net (Cicek et al., 2016).

    Args:
        in_ch: input channels (1 for single-sequence MRI).
        n_classes: output classes (6: background + 5 structures).
        base: channels at the first level; doubled at each level down.
    """
    def __init__(self, in_ch=1, n_classes=6, base=16):
        super().__init__()
        c1, c2, c3, c4, c5 = (base * 2**i for i in range(5))  # 16, 32, 64, 128, 256

        # max-pool layer used in research paper (Çiçek et al. (2016) - no trainable parameters
        self.pool = nn.MaxPool3d(2)
        self.enc1 = DoubleConv(in_ch, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.bottleneck = DoubleConv(c4, c5)

        # Each up step: transposed conv doubles the size and halves the channels,
        # then DoubleConv merges it with the skip (so its input is 2x channels).
        self.up4, self.dec4 = nn.ConvTranspose3d(c5, c4, 2, stride=2), DoubleConv(c5, c4)
        self.up3, self.dec3 = nn.ConvTranspose3d(c4, c3, 2, stride=2), DoubleConv(c4, c3)
        self.up2, self.dec2 = nn.ConvTranspose3d(c3, c2, 2, stride=2), DoubleConv(c3, c2)
        self.up1, self.dec1 = nn.ConvTranspose3d(c2, c1, 2, stride=2), DoubleConv(c2, c1)

        self.out = nn.Conv3d(c1, n_classes, kernel_size=1)  # per-voxel class scores

    def forward(self, x):
        """(B, 1, X, Y, Z) -> (B, n_classes, X, Y, Z) logits."""
        e1 = self.enc1(x)                       # (B, 16, 128, 128, 64)
        e2 = self.enc2(self.pool(e1))           # (B, 32,  64,  64, 32)
        e3 = self.enc3(self.pool(e2))           # (B, 64,  32,  32, 16)
        e4 = self.enc4(self.pool(e3))           # (B, 128, 16,  16,  8)
        b = self.bottleneck(self.pool(e4))      # (B, 256,  8,   8,  4)

        # Skip connections: concatenate along the channel dimension.
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1) # (B, 6, 128, 128, 64)


def conv_norm_act(in_ch, out_ch, k=3, stride=1):
    """Conv3d -> InstanceNorm -> LeakyReLU, the Improved U-Net's basic layer.

    Args:
        in_ch, out_ch: input / output channels.
        k: kernel size (3, or 1 for a 1x1x1 channel-mixing conv).
        stride: 1 keeps the size; 2 halves it (learned downsampling).
    """
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, k, stride=stride, padding=k // 2, bias=False),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.LeakyReLU(0.01, inplace=True),  # small negative slope: no "dead" units
    )


class ContextModule(nn.Module):
    """Residual encoder block (Isensee et al., 2018):
    x + [IN -> LeakyReLU -> conv -> dropout -> IN -> LeakyReLU -> conv](x).

    The identity path lets gradients bypass the convs (He et al., 2016), so the
    block only has to learn a correction to its input. Shape unchanged.

    Args:
        ch: channels in and out.
        p_drop: dropout probability between the convs (regularisation).
    """
    def __init__(self, ch, p_drop=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(ch, affine=True), nn.LeakyReLU(0.01),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.Dropout3d(p_drop),  # drops whole feature maps, active only in model.train()
            nn.InstanceNorm3d(ch, affine=True), nn.LeakyReLU(0.01),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
        )

    def forward(self, x):
        """(B, ch, X, Y, Z) -> (B, ch, X, Y, Z)."""
        return x + self.block(x)  # residual connection


class UpModule(nn.Module):
    """Upsample 2x (trilinear), then conv to halve the channels.

    Used instead of a transposed conv to avoid checkerboard artefacts.

    Args:
        in_ch, out_ch: input / output channels.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = conv_norm_act(in_ch, out_ch)

    def forward(self, x):
        """(B, in_ch, X, Y, Z) -> (B, out_ch, 2X, 2Y, 2Z)."""
        return self.conv(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))


class LocalisationModule(nn.Module):
    """Decoder block: 3x3x3 conv merges skip + upsampled features, 1x1x1 conv halves channels.

    Args:
        in_ch: channels after concatenation (skip + upsampled).
        out_ch: output channels (in_ch / 2).
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(conv_norm_act(in_ch, in_ch), conv_norm_act(in_ch, out_ch, k=1))

    def forward(self, x):
        """(B, in_ch, X, Y, Z) -> (B, out_ch, X, Y, Z)."""
        return self.block(x)


class ImprovedUNet3D(nn.Module):
    """Improved 3D U-Net (Isensee et al., 2018, "Brain Tumor Segmentation and
    Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge").

    Differences from UNet3D: residual context modules, stride-2 conv downsampling,
    LeakyReLU, upsample+conv, localisation modules, and deep supervision.

    Args:
        in_ch: input channels (1 for single-sequence MRI).
        n_classes: output classes (6).
        base: channels at the first level; doubled at each level down.
    """

    def __init__(self, in_ch=1, n_classes=6, base=16):
        super().__init__()
        c1, c2, c3, c4, c5 = (base * 2**i for i in range(5))  # 16, 32, 64, 128, 256

        # Encoder: level 1 at full size, then stride-2 conv + context module per level.
        self.stem = nn.Conv3d(in_ch, c1, 3, padding=1, bias=False)
        self.ctx1 = ContextModule(c1)
        self.down2, self.ctx2 = conv_norm_act(c1, c2, stride=2), ContextModule(c2)
        self.down3, self.ctx3 = conv_norm_act(c2, c3, stride=2), ContextModule(c3)
        self.down4, self.ctx4 = conv_norm_act(c3, c4, stride=2), ContextModule(c4)
        self.down5, self.ctx5 = conv_norm_act(c4, c5, stride=2), ContextModule(c5)

        # Decoder: upsample (halve channels), concat skip (doubles back), localise (halve).
        self.up4, self.loc4 = UpModule(c5, c4), LocalisationModule(c5, c4)
        self.up3, self.loc3 = UpModule(c4, c3), LocalisationModule(c4, c3)
        self.up2, self.loc2 = UpModule(c3, c2), LocalisationModule(c3, c2)
        self.up1, self.final = UpModule(c2, c1), conv_norm_act(c2, c2)

        # Deep supervision: 1x1x1 segmentation heads at three decoder levels.
        self.seg3 = nn.Conv3d(c3, n_classes, 1)  # 1/4 resolution
        self.seg2 = nn.Conv3d(c2, n_classes, 1)  # 1/2 resolution
        self.seg1 = nn.Conv3d(c2, n_classes, 1)  # full resolution

    def forward(self, x):
        """(B, 1, X, Y, Z) -> (B, n_classes, X, Y, Z) logits."""
        e1 = self.ctx1(self.stem(x))            # (B, 16, 128, 128, 64)
        e2 = self.ctx2(self.down2(e1))          # (B, 32,  64,  64, 32)
        e3 = self.ctx3(self.down3(e2))          # (B, 64,  32,  32, 16)
        e4 = self.ctx4(self.down4(e3))          # (B, 128, 16,  16,  8)
        b = self.ctx5(self.down5(e4))           # (B, 256,  8,   8,  4)

        d4 = self.loc4(torch.cat([self.up4(b), e4], dim=1))   # (B, 128, 16, 16,  8)
        d3 = self.loc3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 64,  32, 32, 16)
        d2 = self.loc2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 32,  64, 64, 32)
        d1 = self.final(torch.cat([self.up1(d2), e1], dim=1)) # (B, 32, 128, 128, 64)

        # Deep supervision: coarse segmentations are upsampled and summed into finer ones,
        # so deep decoder levels get a short gradient path from the loss.
        up = lambda t: F.interpolate(t, scale_factor=2, mode="trilinear", align_corners=False)
        s = self.seg3(d3)
        s = up(s) + self.seg2(d2)
        s = up(s) + self.seg1(d1)
        return s               # (B, 6, 128, 128, 64)


class DoubleConv2D(nn.Module):
    """2D version of DoubleConv: two [3x3 conv -> InstanceNorm -> ReLU] layers."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """(N, in_ch, X, Y) -> (N, out_ch, X, Y)."""
        return self.block(x)


class UNet2D(nn.Module):
    """Slice-wise 2D U-Net (Ronneberger et al., 2015): the 2D-vs-3D control.

    Same layout and widths as UNet3D but with 2D layers. Each axial slice is
    segmented independently (no inter-slice context), so comparing it with the
    3D models isolates the value of 3D context at the prostate base and apex.
    Takes and returns whole volumes, so train.py / predict.py need no changes.

    Args:
        in_ch: input channels (1).
        n_classes: output classes (6).
        base: channels at the first level; doubled at each level down.
    """

    def __init__(self, in_ch=1, n_classes=6, base=16):
        super().__init__()
        c1, c2, c3, c4, c5 = (base * 2**i for i in range(5))  # 16, 32, 64, 128, 256
        self.pool = nn.MaxPool2d(2)
        self.enc1, self.enc2 = DoubleConv2D(in_ch, c1), DoubleConv2D(c1, c2)
        self.enc3, self.enc4 = DoubleConv2D(c2, c3), DoubleConv2D(c3, c4)
        self.bottleneck = DoubleConv2D(c4, c5)
        self.up4, self.dec4 = nn.ConvTranspose2d(c5, c4, 2, stride=2), DoubleConv2D(c5, c4)
        self.up3, self.dec3 = nn.ConvTranspose2d(c4, c3, 2, stride=2), DoubleConv2D(c4, c3)
        self.up2, self.dec2 = nn.ConvTranspose2d(c3, c2, 2, stride=2), DoubleConv2D(c3, c2)
        self.up1, self.dec1 = nn.ConvTranspose2d(c2, c1, 2, stride=2), DoubleConv2D(c2, c1)
        self.out = nn.Conv2d(c1, n_classes, 1)

    def forward_2d(self, x):
        """Standard 2D U-Net on a stack of slices: (N, 1, X, Y) -> (N, n_classes, X, Y)."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

    def forward(self, x):
        """(B, 1, X, Y, Z) volume -> (B, n_classes, X, Y, Z), each axial slice processed alone."""
        B, C, X, Y, Z = x.shape
        # Move the slice axis Z next to the batch, then merge them: every slice becomes
        # its own 2D sample, so the network never sees neighbouring slices.
        slices = x.permute(0, 4, 1, 2, 3).reshape(B*Z, C, X, Y)   # (B*Z, 1, X, Y)
        out = self.forward_2d(slices)                              # (B*Z, 6, X, Y)
        return out.reshape(B, Z, -1, X, Y).permute(0, 2, 3, 4, 1)  # (B, 6, X, Y, Z)


class DiceCELoss(nn.Module):
    """Cross-entropy + (1 - soft Dice averaged over classes).

    Dice weights every class equally regardless of size, so the 0.1% prostate
    matters as much as the background; CE gives stable gradients early on.
    Soft Dice loss from Milletari et al., 2016, "V-Net" (3DV).

    Args:
        n_classes: number of classes.
        eps: smoothing term so empty classes don't divide by zero.
    """
    def __init__(self, n_classes=6, eps=1e-5):
        super().__init__()
        self.n_classes, self.eps = n_classes, eps

    def forward(self, logits, target):
        """logits: (B, C, X, Y, Z) float; target: (B, X, Y, Z) int64 labels. Returns a scalar loss."""
        logits = logits.float()  # compute the loss in fp32 even under bf16 autocast
        ce = F.cross_entropy(logits, target)

        probs = torch.softmax(logits, dim=1)                                   # class probabilities
        onehot = F.one_hot(target, self.n_classes).permute(0, 4, 1, 2, 3).float()  # (B,X,Y,Z,C) -> (B,C,X,Y,Z)
        dims = (0, 2, 3, 4)                                                        # sum over batch + space
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)      # one soft Dice per class (6 classes)
        dice_loss = 1 - dice.mean()
        # Keep the two parts so train.py can log them separately (diagnostic only).
        self.last = {"ce": ce.item(), "dice": dice_loss.item()}
        return ce + dice_loss


if __name__ == "__main__":
    # Shape test on CPU for all three models: output shape, parameter count, loss runs.
    x = torch.randn(1, 1, 128, 128, 64)
    target = torch.randint(0, 6, (1, 128, 128, 64))
    for model in (UNet2D(), UNet3D(), ImprovedUNet3D()):
        y = model(x)
        print(f"{type(model).__name__}: output {tuple(y.shape)}, "
            f"{sum(p.numel() for p in model.parameters()):,} params, "
            f"loss {DiceCELoss()(y, target).item():.3f}")