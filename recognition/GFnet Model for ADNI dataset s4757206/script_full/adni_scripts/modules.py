"""CNN baseline and GFNet-inspired models for ADNI slice classification.

Inputs:
    Grayscale tensors with shape [batch, 1, height, width].

Outputs:
    Two logits per image: NC = 0, AD = 1.

Architecture reference for global filtering:
    https://github.com/raoyongming/GFNet/blob/master/gfnet.py

The GFNet-inspired implementation uses separate residual connections
around the global filter and MLP branches.
"""

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """Three-stage CNN with global average pooling."""

    def __init__(self):
        super().__init__()

        # Three convolution/pooling stages: 256 -> 128 -> 64 -> 32 pixels.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Global averaging avoids a large fully connected spatial classifier.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        """Map [B,1,H,W] images to [B,2] NC/AD logits."""
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class GlobalFilter(nn.Module):
    """Learnable frequency-domain filtering of channels-first features."""

    def __init__(self, height, width, channels):
        super().__init__()

        self.height = height
        self.width = width
        self.channels = channels

        # The final dimension stores real and imaginary components.
        self.complex_weight = nn.Parameter(
            torch.randn(
                channels,
                height,
                width // 2 + 1,
                2,
            ) * 0.02
        )

    def forward(self, x):
        # x: [batch, channels, height, width]
        x = torch.fft.rfft2(
            x,
            dim=(-2, -1),
            norm="ortho",
        )

        # rfft2 stores only W//2+1 frequency columns for real-valued inputs.
        # One complex filter per channel broadcasts over the batch dimension.
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight

        # Explicit spatial shape reconstructs the original feature grid.
        return torch.fft.irfft2(
            x,
            s=(self.height, self.width),
            dim=(-2, -1),
            norm="ortho",
        )


class GFBlock(nn.Module):
    """Global filter and MLP block with two residual connections."""

    def __init__(self, dim, grid_size, mlp_ratio=4, dropout=0.1):
        super().__init__()

        # Channel-last layout makes LayerNorm act on the feature dimension.
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilter(grid_size, grid_size, dim)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = dim * mlp_ratio

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [batch, height, width, channels]
        residual = x
        x = self.norm1(x)

        # GlobalFilter operates on channels-first tensors.
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.filter(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        # Separate identity paths surround the filter and the MLP.
        x = residual + x
        x = x + self.mlp(self.norm2(x))

        return x


class GFNet(nn.Module):
    """GFNet-inspired classifier with a fixed square input size.

    Defaults match the final experiment:
    image_size=256, patch_size=16, dim=128, depth=8, dropout=0.1.
    """

    def __init__(
        self,
        image_size=256,
        patch_size=16,
        dim=128,
        depth=8,
        dropout=0.1,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        grid_size = image_size // patch_size

        # Non-overlapping 16x16 patches produce a 16x16 feature grid.
        self.patch_embed = nn.Conv2d(
            1,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, grid_size, grid_size, dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            GFBlock(dim, grid_size, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 2)

    def forward(self, x):
        """Embed patches, mix spatial information, then average to two logits."""
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        x = x.mean(dim=(1, 2))
        return self.classifier(x)