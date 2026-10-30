"""Models for ADNI AD vs NC classification (PyTorch only, no NumPy).

BaselineCNN: small 4-block convolutional network used as the comparison
baseline for the confidence/calibration investigation.

GFNet: Global Filter Network, with an optional convolutional stem.

GFNet design (GlobalFilter layer, Global Filter blocks, overall architecture):
    Y. Rao, W. Zhao, Z. Zhu, J. Lu, J. Zhou, "Global Filter Networks for Image
    Classification", NeurIPS 2021 (arXiv:2107.00645). Each block replaces
    self-attention with an FFT, a learnable complex filter and an inverse FFT.
    The layer here is my own PyTorch implementation of the method described in
    that paper (rfft2 / irfft2 with a learnable filter of shape H x (W//2+1) x D),
    written with AI assistance as described in the README.

Convolutional stem (conv_stem=True), which replaces the paper's single 16 x 16
patch embedding with three small convolutions:
    T. Xiao, M. Singh, E. Mintun, T. Darrell, P. Dollar, R. Girshick,
    "Early Convolutions Help Transformers See Better", NeurIPS 2021
    (arXiv:2106.14881). The idea of using a small stack of stride-2 convolutions
    instead of a single large-patch embedding comes from this work. The specific
    layer sizes here are my own choice for 224 x 224 single-channel MRI slices.

DropPath (stochastic depth) follows G. Huang et al., "Deep Networks with
Stochastic Depth", ECCV 2016.
"""
import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """4 x (conv3x3 -> BatchNorm -> ReLU -> maxpool) -> global avg pool -> linear."""

    def __init__(self, num_classes=2, in_ch=1, widths=(16, 32, 64, 128)):
        super().__init__()
        layers, c = [], in_ch
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(c, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


class DropPath(nn.Module):
    """Stochastic depth: during training, skip a block's residual branch for random samples."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep)
        return x * mask / keep


class GFNetBlock(nn.Module):
    """norm -> frequency filter -> norm -> MLP, with one residual connection."""

    def __init__(self, dim, h, w, mlp_ratio=4, drop=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilter(dim, h, w)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))


class GFNet(nn.Module):
    """Small Global Filter Network: patch embedding, filter blocks, pooled linear head.

    conv_stem=True swaps the single linear patch projection for a small
    convolutional stem (two stride-2 convs, then a strided conv) with the same
    overall downsampling, so the model gets local features before the filters.
    """

    def __init__(self, num_classes=2, in_ch=1, img_size=224, patch=16,
                 dim=128, depth=4, mlp_ratio=4, drop=0.1, drop_path=0.0, conv_stem=False):
        super().__init__()
        g = img_size // patch  # token grid is g x g (14 x 14 for 224 / 16)
        if conv_stem:
            k = patch // 4
            self.patch_embed = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
                nn.Conv2d(64, dim, kernel_size=k, stride=k))
        else:
            self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, g * g, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(drop)
        rates = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.Sequential(*[GFNetBlock(dim, g, g, mlp_ratio, drop, rates[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (batch, tokens, dim)
        x = self.drop(x + self.pos)
        x = self.norm(self.blocks(x)).mean(dim=1)           # average over tokens
        return self.head(x)


class GlobalFilter(nn.Module):
    """Learnable filter applied in the frequency domain (the core of GFNet).

    Input is a grid of h x w patch tokens with `dim` channels. We take a 2D FFT
    over the grid, multiply each frequency by a learnable complex number, and
    transform back. Complex weights are stored as a real tensor with a last
    dimension of 2 (real, imaginary) so the optimiser handles them normally.
    """

    def __init__(self, dim, h=14, w=14):
        super().__init__()
        self.h, self.w = h, w
        # rfft2 keeps w // 2 + 1 frequencies along the width (real input is symmetric)
        self.weight = nn.Parameter(torch.randn(h, w // 2 + 1, dim, 2) * 0.02)

    def forward(self, x):  # x: (batch, h * w, dim)
        B, N, C = x.shape
        x = x.float().reshape(B, self.h, self.w, C)  # FFT needs float32
        X = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        X = X * torch.view_as_complex(self.weight)
        x = torch.fft.irfft2(X, s=(self.h, self.w), dim=(1, 2), norm="ortho")
        return x.reshape(B, N, C)
