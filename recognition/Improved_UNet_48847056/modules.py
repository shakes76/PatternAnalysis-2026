"""
modules.py - U-Net (2D or 3D) and Dice-based loss/metrics for HipMRI segmentation.

One UNet class serves both the 2D slice baseline (dims=2) and the 3D volumetric
model (dims=3), so the comparison differs only in spatial context, not in code.
Architecture: two 3x3 convs
per level, max-pool down, transposed-conv up, skip concatenation. BatchNorm is added
after each conv for faster, more stable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _layers(dims):
    """Helper function that returns the layers needed for either 2D or 3D U-Net architecture"""
    if dims == 2:
        return nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.ConvTranspose2d
    if dims == 3:
        return nn.Conv3d, nn.BatchNorm3d, nn.MaxPool3d, nn.ConvTranspose3d
    raise ValueError("dims must be 2 or 3")


class DoubleConv(nn.Module):
    """
    (conv 3x3 -> BN -> ReLU) x 2. Two convs per level widen the receptive field
    and add a non-linearity before each resolution change.
    padding = 1 to make output size same as input.
    """

    def __init__(self, in_ch, out_ch, dims=2):
        super().__init__()
        Conv, BN, _, _ = _layers(dims) # only conv and batchnorm needed for init
        self.block = nn.Sequential(
            Conv(in_ch, out_ch, 3, padding=1, bias=False), BN(out_ch), nn.ReLU(inplace=True),
            Conv(out_ch, out_ch, 3, padding=1, bias=False), BN(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # (N, in_ch, *spatial) ->(N, out_ch, *spatial) -- spatial size is unchanged
        return self.block(x)


class UNet(nn.Module):
    """Encoder-decoder with skip connections.

    in_ch: input channels (1 for MRI). num_classes: output logits per voxel.
    base: channels at the first level, doubled at each depth step.
    depth: number of down-sampling steps. Input spatial sizes must be divisible by
    2**depth (2D slices 256x128 with depth=4 is fine).
    """

    def __init__(self, in_ch=1, num_classes=6, base=32, depth=4, dims=2):
        super().__init__()
        _, _, Pool, UpConv = _layers(dims)
        # channel widths per level
        chs = [base * 2**i for i in range(depth + 1)]  # e.g. 32,64,128,256,512

        # Encoder (down) #
        self.inc = DoubleConv(in_ch, chs[0], dims) # for the first level with no pooling
        self.pool = Pool(2) # halves each spatial dim
        # then double conv
        self.down = nn.ModuleList(DoubleConv(chs[i], chs[i + 1], dims) for i in range(depth))

        # Decoder (up) #
        # transposed conv, doubles spatial size, halves channels
        self.up = nn.ModuleList(UpConv(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth)))
        # then double conv with the skip connection
        self.dec = nn.ModuleList(DoubleConv(chs[i] * 2, chs[i], dims) for i in reversed(range(depth)))

        # 1x1 conv to map top level efatures to one logit per class per pixel/voxel
        self.head = _layers(dims)[0](chs[0], num_classes, 1)

    def forward(self, x):
        # keep the output of each level to reuse in the decoder
        skips = [self.inc(x)]
        for down in self.down:
            skips.append(down(self.pool(skips[-1])))

        x = skips.pop()  # get the bottleneck

        # upsample and then fuse with encoder output at the same level
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([skips.pop(), x], dim=1))  # skip connection
        return self.head(x)  # raw logits, (N, num_classes, *spatial)


def dice_per_class(logits, target, num_classes, eps=1e-6):
    """Soft Dice for each class, averaged over the batch.

    dice = 2|P n T| / (|P| + |T|). "Soft" means P is the softmax probability rather
    than a hard 0/1 mask, so the score is differentiable and can be used in a loss.
 
    logits: (N, C, *spatial); target: (N, *spatial) int class indices.
    Returns a tensor of shape (C,).
 
    Note: Dice is computed per sample, then averaged. For a 2D slice where a class
    is absent, the score is ~0 unless the model predicts almost exactly nothing
    for that class, so these slices push its probability towards zero.
    """
    probs = logits.softmax(dim=1) # per-pixel class probs, sum to 1/C
    onehot = F.one_hot(target, num_classes).movedim(-1, 1).float()
    dims = tuple(range(2, probs.ndim))  # spatial dims 
    inter = (probs * onehot).sum(dims) # soft |P n T|
    denom = probs.sum(dims) + onehot.sum(dims) # |P| + |T|
    return ((2 * inter + eps) / (denom + eps)).mean(0) # add eps to avoid divide by zero


class DiceCELoss(nn.Module):
    """Cross-entropy + (1 - mean soft Dice).

    CE gives stable gradients early on; the Dice term counteracts the extreme class
    imbalance (prostate < 2% of voxels), which plain CE would let the background dominate.
    """

    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight, self.dice_weight = ce_weight, dice_weight

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target)
        dice = dice_per_class(logits, target, self.num_classes).mean()
        return self.ce_weight * ce + self.dice_weight * (1 - dice)


@torch.no_grad()
def hard_scores(logits, target, num_classes):
    """Hard (argmax) Dice and IoU per class, summed over the whole batch so tiny
    structures are not swamped by empty slices. Returns (dice[C], iou[C]); a class
    absent from both prediction and target gets NaN (exclude it with nanmean)."""
    pred = logits.argmax(1)
    dice, iou = [], []
    for c in range(num_classes):
        p, t = pred == c, target == c
        inter = (p & t).sum().float()
        ps, ts = p.sum().float(), t.sum().float()
        union = ps + ts - inter
        dice.append(2 * inter / (ps + ts) if ps + ts > 0 else torch.tensor(float("nan")))
        iou.append(inter / union if union > 0 else torch.tensor(float("nan")))
    return torch.stack(dice), torch.stack(iou)


@torch.no_grad()
def predict_volume(model, vol, dims):
    """Predict one 3D volume (H, W, D) tensor and return logits (C, H, W, D).

    dims=3: the whole volume goes through the 3D network in one pass.
    dims=2: each axial slice (last axis) goes through the 2D network as a batch,
    then the slice predictions are stacked back into a volume. This lets both
    models be scored on exactly the same test volumes.
    """
    if dims == 3:
        return model(vol[None, None])[0]
    slices = vol.permute(2, 0, 1)[:, None]  # (D, 1, H, W)
    return model(slices).permute(1, 2, 3, 0)  # (D, C, H, W) -> (C, H, W, D)


if __name__ == "__main__":
    # Shape smoke test for both modes
    for dims, shape in [(2, (2, 1, 256, 128)), (3, (1, 1, 64, 64, 32))]:
        m = UNet(dims=dims, base=16)
        out = m(torch.randn(shape))
        print(f"{dims}D out {tuple(out.shape)} params {sum(p.numel() for p in m.parameters()):,}")