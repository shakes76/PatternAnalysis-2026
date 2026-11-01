"""Residual 3D U-Net with GroupNorm and auxiliary decoder supervision."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Two spatially preserving convolutions with an optional projected residual.

    GroupNorm supports the experiment's batch size of one. The projection
    changes channels only; setting residual=False gives the baseline block.
    """

    def __init__(self, incoming, outgoing, residual=True):
        super().__init__()
        # Use the largest supported group count that divides the channel width.
        groups = next(g for g in (8, 4, 2, 1) if outgoing % g == 0)
        self.body = nn.Sequential(
            nn.Conv3d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outgoing),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outgoing),
        )
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv3d(incoming, outgoing, 1)
        self.residual = residual
        if not residual:
            self.skip = None

    def forward(self, x):
        y = self.body(x)
        return F.leaky_relu(y + self.skip(x) if self.residual else y, 0.01)


class ImprovedUNet3D(nn.Module):
    """Four-level encoder/decoder shared by the baseline and improved models.

    Inputs have shape (N, 1, D, H, W). Outputs are full-resolution class
    logits and optional half-resolution auxiliary logits. Module attribute
    names are part of the saved state_dict contract and must remain stable.
    """

    def __init__(self, num_classes=6, base_channels=16, architecture="improved"):
        super().__init__()
        if architecture not in ("baseline", "improved"):
            raise ValueError("Architecture must be baseline or improved")
        residual = architecture == "improved"
        widths = [base_channels * 2**i for i in range(4)]
        self.encoders = nn.ModuleList(
            [
                ResidualBlock(1 if i == 0 else widths[i - 1], width, residual)
                for i, width in enumerate(widths)
            ]
        )
        self.decoders = nn.ModuleList(
            [ResidualBlock(widths[i + 1] + widths[i], widths[i], residual) for i in (2, 1, 0)]
        )
        self.output = nn.Conv3d(widths[0], num_classes, 1)
        self.aux_output = nn.Conv3d(widths[1], num_classes, 1) if residual else None

    def forward(self, x):
        """Decode with encoder skips and return logits without applying softmax."""
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(F.max_pool3d(x, 2) if i else x)
            skips.append(x)
        auxiliary = None
        for i, decoder in enumerate(self.decoders):
            skip = skips[2 - i]
            # Match the stored skip shape, including odd-sized input dimensions.
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
            if i == 1 and self.aux_output is not None:
                auxiliary = self.aux_output(x)
        return self.output(x), auxiliary


def segmentation_loss(logits, target):
    """Cross entropy plus foreground soft Dice; compute reductions in float32."""
    logits = logits.float()
    probabilities = logits.softmax(1)
    encoded = F.one_hot(target, logits.shape[1]).movedim(-1, 1).float()
    # Pool over batch and spatial dimensions, retaining one score per class.
    axes = (0, 2, 3, 4)
    intersection = (probabilities * encoded).sum(axes)
    denominator = probabilities.sum(axes) + encoded.sum(axes)
    dice = (2 * intersection + 1e-5) / (denominator + 1e-5)
    # Background participates in cross entropy but not the foreground Dice mean.
    return F.cross_entropy(logits, target) + 1 - dice[1:].mean()


def deep_supervision_loss(outputs, target, auxiliary_weight=0.3):
    """Add auxiliary loss only for the improved architecture.

    Nearest-neighbour resizing retains integer class identities on the
    coarser target grid. The auxiliary head does not affect inference.
    """
    logits, auxiliary = outputs
    if auxiliary is None:
        return segmentation_loss(logits, target)
    small_target = F.interpolate(target[:, None].float(), size=auxiliary.shape[2:], mode="nearest")[
        :, 0
    ].long()
    return segmentation_loss(logits, target) + auxiliary_weight * segmentation_loss(
        auxiliary, small_target
    )
