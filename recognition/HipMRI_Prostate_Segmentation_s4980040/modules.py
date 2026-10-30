"""
modules.py - Model architectures for HipMRI prostate segmentation.

CURRENTLY CONTAINS
    UNet2D: the standard 2D U-Net (Ronneberger et al., MICCAI 2015) -> our BASELINE model.
    Later, the 3D U-Net and 3D Improved U-Net will be added to this same file.

=====================================================================
QUICK PYTORCH BASICS (read this first)
=====================================================================
Tensor shapes
    Images move through the network as 4D tensors shaped (B, C, H, W):
        B = batch size  -> how many slices are processed at once (e.g. 16)
        C = channels    -> how many "feature maps" per pixel
                           (1 for a greyscale MRI at the input; 64, 128... inside the network)
        H = height      -> 256 for our slices
        W = width       -> 128 for our slices
    Example: (16, 1, 256, 128) = 16 greyscale slices of size 256x128.

nn.Module
    Every layer and every model in PyTorch is a class that inherits from nn.Module.
    You always write two methods:
        __init__  -> CREATE the layers (this is where the learnable weights live)
        forward   -> say HOW data flows through those layers
    You never call forward() yourself: writing model(x) calls it for you.

    super().__init__() must be the first line of every __init__, it sets up
    nn.Module's internal bookkeeping (tracking weights, moving to GPU, etc.).

=====================================================================
HOW A U-NET WORKS (the "U" shape)
=====================================================================
                    input 256x128
    inc   ──────────────────────────────────────────►  up4  -> head -> output 256x128
      down1  ─────────────────────────────────────► up3
        down2  ────────────────────────────────► up2
          down3  ──────────────────────────► up1
                       down4 (bottleneck, 16x8)

    Left side (encoder, going DOWN):
        Each step halves the image size and doubles the channels.
        Think of it as zooming out: the network sees a wider area at once, so it learns
        WHAT things are (this blob is bladder, that's bone), but loses detail about exactly
        WHERE the edges are.
    Bottom (bottleneck):
        The most zoomed-out, most "summarised" view of the image.
    Right side (decoder, going UP):
        Each step doubles the size back up, until we're at full 256x128 again,
        so we can give every single pixel a label.
    Horizontal arrows (skip connections):
        Each decoder step gets a copy of the encoder output from the SAME size.
        This hands back the fine edge detail that was lost while zooming out,
        which is why U-Nets draw sharp boundaries.
    Output:
        For every pixel, 6 numbers ("logits"), one score per class:
        0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate.
        The class with the highest score is the prediction for that pixel.

DIFFERENCES FROM THE ORIGINAL PAPER (state these in the README)
    - padding=1 on every 3x3 conv, so the output is the same size as the input
      (the original paper used no padding and cropped the skip connections instead).
    - BatchNorm after each conv, for faster, more stable training.

Per the assignment rules, this file uses only PyTorch (no NumPy).
"""

import torch
import torch.nn as nn            # layers: Conv2d, BatchNorm2d, ReLU, MaxPool2d, ...
import torch.nn.functional as F  # stateless functions (no learnable weights), e.g. F.interpolate


# ===========================================================================
# Building block 1: DoubleConv
# ===========================================================================
class DoubleConv(nn.Module):
    """Two rounds of (3x3 conv -> BatchNorm -> ReLU). Used at EVERY level of the U-Net.

    What each layer does:
        Conv2d (3x3):  slides small 3x3 filters across the image. Each filter learns to
                       detect one pattern (an edge, a texture, a curve). out_ch filters
                       -> out_ch output channels (feature maps).
        BatchNorm2d:   rescales each channel to roughly mean 0, spread 1 across the batch.
                       Keeps numbers in a healthy range so training doesn't stall or explode.
        ReLU:          replaces negative values with 0. Without a non-linear step like this,
                       stacking layers would be no more powerful than a single layer.

    Doing it twice lets each level combine simple patterns into slightly richer ones.
    Shape: (B, in_ch, H, W) -> (B, out_ch, H, W)   (size stays the same thanks to padding=1)
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # nn.Sequential runs the layers in order: output of one feeds into the next
        self.block = nn.Sequential(
            # padding=1 adds a 1-pixel border so a 3x3 filter doesn't shrink the image.
            # bias=False because the BatchNorm right after adds its own bias anyway.
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),  # inplace=True overwrites the input to save memory
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ===========================================================================
# Building block 2: Down (one step down the left side of the U)
# ===========================================================================
class Down(nn.Module):
    """Halve the image size, then DoubleConv (which usually doubles the channels).

    MaxPool2d(2) looks at every 2x2 square of pixels and keeps only the biggest value,
    so height and width are both halved.
    Shape example: (B, 64, 256, 128) -> pool -> (B, 64, 128, 64) -> conv -> (B, 128, 128, 64)
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)          # no learnable weights, just picks maximums
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)   # shrink
        x = self.conv(x)   # learn features at this new, zoomed-out scale
        return x


# ===========================================================================
# Building block 3: Up (one step up the right side of the U)
# ===========================================================================
class Up(nn.Module):
    """Double the image size, attach the skip connection, then DoubleConv.

    Step by step, e.g. going from the bottleneck to the next level up (base=64):
        x    (B, 1024, 16,  8)  coming up from below
        up   (B,  512, 32, 16)  ConvTranspose2d doubles H and W, halves channels
        skip (B,  512, 32, 16)  saved earlier from the encoder at this same size
        cat  (B, 1024, 32, 16)  glue them together side by side along channels
        conv (B,  512, 32, 16)  DoubleConv mixes "what" (from below) with "where" (from skip)
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # ConvTranspose2d with stride=2 is a LEARNABLE way to upsample (roughly the reverse
        # of a strided conv): every input pixel is expanded into a 2x2 output patch.
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # After gluing on the skip we have out_ch (upsampled) + out_ch (skip) = 2 * out_ch
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)  # double the size

        # Safety net: with odd input sizes, upsampling can come out 1 pixel off.
        # Our 256x128 slices divide cleanly by 16, so this normally never runs.
        if x.shape[-2:] != skip.shape[-2:]:  # shape[-2:] = the last two dims, (H, W)
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        # dim=1 is the channel dimension in (B, C, H, W), so this stacks the two
        # sets of feature maps on top of each other (sizes must match in H and W)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ===========================================================================
# The full model
# ===========================================================================
class UNet2D(nn.Module):
    """Standard 2D U-Net: 4 steps down, 4 steps up. This is our BASELINE model.

    Args:
        in_channels: channels in the input image. 1, because MRI slices are greyscale.
        num_classes: channels in the output, one per label. 6 for our dataset.
        base:        number of channels at the top level; it doubles at every step down.
                     64 = same as the original paper (~31M learnable parameters).
                     32 = lighter and faster (~8M), handy if GPU memory is tight.

    Full shape journey for a 256x128 input with base=64:
        input       (B, 1,    256, 128)
        inc   -> x1 (B, 64,   256, 128)   full size
        down1 -> x2 (B, 128,  128,  64)   1/2 size
        down2 -> x3 (B, 256,   64,  32)   1/4 size
        down3 -> x4 (B, 512,   32,  16)   1/8 size
        down4 -> x5 (B, 1024,  16,   8)   1/16 size  <- bottleneck
        up1 (x5 + skip x4) -> (B, 512,  32,  16)
        up2 (   + skip x3) -> (B, 256,  64,  32)
        up3 (   + skip x2) -> (B, 128, 128,  64)
        up4 (   + skip x1) -> (B, 64,  256, 128)
        head               -> (B, 6,   256, 128)  <- 6 class scores for every pixel
    """

    def __init__(self, in_channels=1, num_classes=6, base=64):
        super().__init__()
        # Channels at each level: [64, 128, 256, 512, 1024] when base=64
        c = [base, base * 2, base * 4, base * 8, base * 16]

        # ---- Encoder (left side, going down) ----
        self.inc = DoubleConv(in_channels, c[0])  # first block: no pooling, stays full size
        self.down1 = Down(c[0], c[1])
        self.down2 = Down(c[1], c[2])
        self.down3 = Down(c[2], c[3])
        self.down4 = Down(c[3], c[4])             # bottleneck

        # ---- Decoder (right side, going up) ----
        self.up1 = Up(c[4], c[3])
        self.up2 = Up(c[3], c[2])
        self.up3 = Up(c[2], c[1])
        self.up4 = Up(c[1], c[0])

        # ---- Output layer ----
        # A 1x1 conv looks at one pixel at a time and turns its 64 feature values
        # into 6 class scores. It's basically a tiny classifier applied to every pixel.
        self.head = nn.Conv2d(c[0], num_classes, kernel_size=1)

    def forward(self, x):
        # ---- Going down: save every level's output, the decoder needs them as skips ----
        x1 = self.inc(x)     # full size
        x2 = self.down1(x1)  # 1/2
        x3 = self.down2(x2)  # 1/4
        x4 = self.down3(x3)  # 1/8
        x5 = self.down4(x4)  # 1/16 (bottleneck)

        # ---- Going up: each step upsamples and merges with the matching saved level ----
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # ---- Raw scores ("logits") out ----
        # No softmax here on purpose: PyTorch's CrossEntropyLoss applies it internally,
        # and doing it twice would break training.
        # To turn logits into predicted labels later:  preds = logits.argmax(dim=1)
        #   -> shape (B, 256, 128), each pixel holding a class number 0-5
        return self.head(x)

# ===========================================================================
# 3D Improved U-Net (Isensee et al., BrainLes/BRATS 2017) -> our MAIN (Hard) model
# ===========================================================================
# Same U shape as UNet2D, but:
#   - 3D convolutions (3x3x3) so it sees neighbouring slices too, not just one slice
#   - downsampling with stride-2 convs instead of max-pooling
#   - "context modules": residual blocks (input + learned correction), which train
#     more reliably in deep networks
#   - InstanceNorm + LeakyReLU instead of BatchNorm + ReLU (works better with tiny batches)
#   - deep supervision: segmentation maps from lower decoder levels are upsampled and
#     added to the final output, so the deeper layers get a direct learning signal
# residual=False and deep_supervision=False switch those two extras off, giving a
# plainer 3D U-Net (handy as a fallback and for an ablation study in the README).

class ContextModule3D(nn.Module):
    """Two 3x3x3 convs with dropout in between, in "pre-activation" order
    (normalise -> activate -> conv). Its output is ADDED to its input (residual)."""

    def __init__(self, ch, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),  # like ReLU but lets a little negative signal through
            nn.Conv3d(ch, ch, kernel_size=3, padding=1),
            nn.Dropout3d(dropout),             # randomly switch off whole channels: reduces overfitting
            nn.InstanceNorm3d(ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(ch, ch, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class UpsampleModule3D(nn.Module):
    """Double the size (nearest-neighbour), then a 3x3x3 conv that halves the channels."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode="nearest")  # size = the skip connection's (X, Y, Z)
        return self.conv(x)


class LocalizationModule3D(nn.Module):
    """After concatenating with the skip: a 3x3x3 conv, then a 1x1x1 conv that halves the channels."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ImprovedUNet3D(nn.Module):
    """3D Improved U-Net. Input (B, 1, X, Y, Z) -> output (B, num_classes, X, Y, Z).

    Channels per level with base=16: [16, 32, 64, 128, 256] (as in the paper).
    Shapes for a (128, 128, 64) volume:
        level 0: (16,  128, 128, 64)   full size
        level 1: (32,   64,  64, 32)   1/2
        level 2: (64,   32,  32, 16)   1/4
        level 3: (128,  16,  16,  8)   1/8
        level 4: (256,   8,   8,  4)   1/16 (bottleneck)
    """

    def __init__(self, in_channels=1, num_classes=6, base=16, residual=True,
                 deep_supervision=True, dropout=0.3):
        super().__init__()
        c = [base * 2 ** i for i in range(5)]  # [16, 32, 64, 128, 256]
        self.residual = residual
        self.deep_supervision = deep_supervision

        # ---- Encoder: first conv keeps full size, the rest halve it (stride 2) ----
        self.enc_convs = nn.ModuleList(
            [nn.Conv3d(in_channels, c[0], kernel_size=3, padding=1)]
            + [nn.Conv3d(c[i - 1], c[i], kernel_size=3, stride=2, padding=1) for i in range(1, 5)])
        self.context = nn.ModuleList([ContextModule3D(ch, dropout) for ch in c])

        # ---- Decoder: climb from the bottleneck back to level 3, 2, 1, 0 ----
        self.ups = nn.ModuleList([UpsampleModule3D(c[i + 1], c[i]) for i in (3, 2, 1, 0)])
        # after concatenating with the skip there are 2 * c[i] channels
        self.locs = nn.ModuleList([LocalizationModule3D(2 * c[i], c[i]) for i in (3, 2, 1)])
        self.final_conv = nn.Sequential(  # top level: one 3x3x3 conv on the 32 concatenated channels
            nn.Conv3d(2 * c[0], 2 * c[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(2 * c[0], affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # ---- Segmentation layers (1x1x1 convs -> class scores) at three levels ----
        self.seg_quarter = nn.Conv3d(c[2], num_classes, kernel_size=1)  # level 2 (1/4 size)
        self.seg_half = nn.Conv3d(c[1], num_classes, kernel_size=1)     # level 1 (1/2 size)
        self.seg_full = nn.Conv3d(2 * c[0], num_classes, kernel_size=1) # level 0 (full size)

    def forward(self, x):
        # ---- Going down: conv, then context module (+ residual), save as skip ----
        skips = []
        for conv, ctx in zip(self.enc_convs, self.context):
            x = conv(x)
            x = x + ctx(x) if self.residual else ctx(x)
            skips.append(x)  # skips[0] = full size ... skips[4] = bottleneck

        # ---- Going up ----
        x = skips[4]
        for step, i in enumerate((3, 2, 1, 0)):
            x = self.ups[step](x, size=skips[i].shape[2:])
            x = torch.cat([skips[i], x], dim=1)
            x = self.locs[step](x) if i > 0 else self.final_conv(x)
            if i == 2:
                seg_q = self.seg_quarter(x)
            elif i == 1:
                seg_h = self.seg_half(x)
        out = self.seg_full(x)

        # ---- Deep supervision: upsample the coarse maps and add them to the final one ----
        if self.deep_supervision:
            s = F.interpolate(seg_q, size=seg_h.shape[2:], mode="nearest") + seg_h
            out = F.interpolate(s, size=out.shape[2:], mode="nearest") + out
        return out  # raw logits, same as UNet2D

def count_parameters(model):
    """Count the learnable weights in a model (goes in the README's resource profiling table).

    model.parameters() yields every weight tensor; numel() = number of values in it;
    requires_grad = True means it's updated during training.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick self-test: `python modules.py` pushes a fake batch through the model to
# check the output shape. Runs on CPU, so it's fine on the Rangpur login node.
# This block does NOT run when train.py does `from modules import UNet2D`.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = UNet2D(in_channels=1, num_classes=6, base=64)
    dummy = torch.randn(2, 1, 256, 128)  # 2 fake "slices" filled with random noise
    with torch.no_grad():                # just checking shapes, so skip gradient tracking (faster, less memory)
        out = model(dummy)               # this calls model.forward(dummy)
    print("Input: ", tuple(dummy.shape))
    print("Output:", tuple(out.shape), "(expected (2, 6, 256, 128))")
    print(f"Trainable parameters: {count_parameters(model):,}")