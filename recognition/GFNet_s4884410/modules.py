"""
Model definitions for ADNI AD vs NC classification.

- BaselineCNN: a small 4-block convolutional baseline (388,834 parameters).
- GFNet: an isotropic Global Filter Network (GFNet-Ti size: 7,124,226 parameters).

Both take a batch of greyscale slices shaped (B, 1, 224, 224) and return two
logits per slice, (B, 2), ordered (NC, AD).

GFNet follows Rao et al., "Global Filter Networks for Image Classification",
NeurIPS 2021 (https://arxiv.org/abs/2107.00645), and was written with reference to
the official implementation (https://github.com/raoyongming/GFNet). DropPath was
written with reference to timm (https://github.com/huggingface/pytorch-image-models).
Everything is implemented in plain PyTorch, without timm or NumPy.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Baseline CNN
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2. Halves height and width.

    The convolution has no bias because BatchNorm adds its own shift.
    """

    def __init__(self, in_ch, out_ch):
        """in_ch / out_ch: number of input / output channels."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        """(B, in_ch, H, W) -> (B, out_ch, H/2, W/2)."""
        return self.block(x)


class BaselineCNN(nn.Module):
    """Small 4-block CNN baseline for AD vs NC (~0.39M parameters).

    Shapes for a 224 x 224 input with the default widths:
        (B, 1, 224, 224) -> 4 blocks -> (B, 256, 14, 14)
        -> global average pool -> (B, 256) -> dropout -> linear -> (B, 2)
    """

    def __init__(self, in_ch=1, num_classes=2, widths=(32, 64, 128, 256), dropout=0.3):
        """
        Args:
            in_ch: input channels (1 for greyscale).
            num_classes: number of output logits.
            widths: output channels of the four convolutional blocks.
            dropout: dropout rate before the final linear layer.
        """
        super().__init__()
        chans = (in_ch, *widths)
        self.features = nn.Sequential(
            *[ConvBlock(a, b) for a, b in zip(chans, chans[1:])]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(widths[-1], num_classes)
        )

    def forward(self, x):
        """(B, 1, 224, 224) -> logits (B, 2)."""
        return self.head(self.pool(self.features(x)))


# ---------------------------------------------------------------------------
# GFNet
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Split the image into non-overlapping patches and project each to a D-dim token.

    A convolution with kernel = stride = patch size is equivalent to cutting the
    image into patches and applying the same linear layer to each one.
    (B, C, 224, 224) -> (B, 196, D) for 16 x 16 patches.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=256):
        """img_size must be divisible by patch_size; embed_dim is the token width D."""
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by patch size"
        self.grid_size = img_size // patch_size  # tokens per side
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """(B, C, H, W) image -> (B, num_patches, D) tokens."""
        x = self.proj(x)  # (B, D, 14, 14)
        return x.flatten(2).transpose(1, 2)  # (B, 196, D)


class GlobalFilter(nn.Module):
    """Global filter layer: mixes information across all token positions.

    The tokens are laid out as an h x h grid, transformed with a 2D real FFT,
    multiplied element-wise by a learnable complex filter (one per channel), and
    transformed back:  X' = IFFT( K * FFT(X) ).
    This equals a circular convolution with a kernel as large as the whole grid,
    so every token can draw on every other token, at O(N log N) cost.

    rfft2 keeps only h x (h // 2 + 1) frequencies, because the spectrum of a real
    input is symmetric. The filter is stored as real and imaginary parts.
    """

    def __init__(self, dim, grid_size=14):
        """dim: channels per token; grid_size: tokens per side of the square grid."""
        super().__init__()
        self.h = grid_size
        self.w = grid_size // 2 + 1
        self.complex_weight = nn.Parameter(torch.randn(self.h, self.w, dim, 2) * 0.02)

    def forward(self, x):
        """(B, N, C) tokens -> (B, N, C), with N = grid_size ** 2."""
        B, N, C = x.shape
        x = x.reshape(B, self.h, self.h, C)
        # cuFFT supports half precision only for power-of-two sizes (14 is not),
        # so the FFT always runs in float32, even under mixed precision.
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x * torch.view_as_complex(self.complex_weight)
        x = torch.fft.irfft2(x, s=(self.h, self.h), dim=(1, 2), norm="ortho")
        return x.reshape(B, N, C)


class Mlp(nn.Module):
    """Token-wise MLP: Linear -> GELU -> Linear. Mixes information across channels."""

    def __init__(self, dim, hidden_dim, drop=0.0):
        """dim: token width; hidden_dim: hidden layer width; drop: dropout rate."""
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """(..., dim) -> (..., dim)."""
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class DropPath(nn.Module):
    """Stochastic depth: during training, skip a block's output for random samples.

    Each sample in the batch keeps the block's output with probability 1 - p,
    scaled by 1 / (1 - p) so the expected value is unchanged. Does nothing in
    eval mode or when p = 0.
    """

    def __init__(self, drop_prob=0.0):
        """drop_prob: probability p of skipping the block for each sample."""
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Apply the per-sample keep/drop mask in training mode; identity otherwise."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # one keep/drop decision per sample, broadcast over the other dimensions
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    """GFNet block: LayerNorm -> GlobalFilter -> LayerNorm -> MLP, with one residual.

    Follows the official implementation, which wraps a single residual connection
    around the whole block: x = x + DropPath(MLP(LN(Filter(LN(x))))).
    """

    def __init__(self, dim, grid_size, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        """
        Args:
            dim: token width.
            grid_size: tokens per side of the grid.
            mlp_ratio: MLP hidden width as a multiple of dim.
            drop: dropout rate inside the MLP.
            drop_path: stochastic-depth rate for this block.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.filter = GlobalFilter(dim, grid_size)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        """(B, N, dim) -> (B, N, dim)."""
        return x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))


class GFNet(nn.Module):
    """Isotropic GFNet: patch embedding, N global-filter blocks, average pool, linear head.

    Defaults match GFNet-Ti (width 256, depth 12) with 1 input channel and 2 classes.
    Shapes for a 224 x 224 input:
        (B, 1, 224, 224) -> patch embed -> (B, 196, 256) -> + position embedding
        -> 12 blocks -> LayerNorm -> mean over tokens -> (B, 256) -> head -> (B, 2)

    Drop-path rates rise linearly from 0 (first block) to drop_path_rate (last block).
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        num_classes=2,
        embed_dim=256,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
    ):
        """
        Args:
            img_size: input height and width (square).
            patch_size: patch height and width; img_size must be divisible by it.
            in_chans: input channels (1 for greyscale).
            num_classes: number of output logits.
            embed_dim: token width.
            depth: number of GFNet blocks.
            mlp_ratio: MLP hidden width as a multiple of embed_dim.
            drop_rate: dropout after the position embedding and inside the MLPs.
            drop_path_rate: stochastic-depth rate of the last block.
        """
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        grid = self.patch_embed.grid_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, grid, mlp_ratio, drop_rate, dpr[i])
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """Truncated-normal linear weights (std 0.02), zero biases, LayerNorm at identity."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """Pooled feature vector before the head, (B, D)."""
        x = self.pos_drop(self.patch_embed(x) + self.pos_embed)
        x = self.blocks(x)
        return self.norm(x).mean(dim=1)

    def forward(self, x):
        """(B, 1, 224, 224) -> logits (B, 2)."""
        return self.head(self.forward_features(x))


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
