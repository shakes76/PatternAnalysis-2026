"""Models for AD vs NC classification of ADNI brain MRI slices.

Contains two classifiers that take a 1 x 224 x 224 greyscale slice and return
two raw logits (NC, AD):

* ConvNetwork - the baseline CNN, adapted from the COMP3710 MNIST CNN practical.
* GFNet - a Global Filter Network (Rao et al., 2023, "Global Filter Networks
  for Image Classification"). It is my own implementation; the block layout
  follows the official code (github.com/raoyongming/GFNet).

Softmax is applied outside the models (see utils.collect_probs), so the
outputs can be used directly with nn.CrossEntropyLoss.
"""
import torch.nn as nn

# Adapted from the COMP3710 MNIST CNN practical (BasicBlock / ConvNetwork).


class BasicBlock(nn.Module):
    def __init__(self, in_dim, dim, kernel_size, stride=1, act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d, dropout_p=0.3):
        super().__init__()
        self.features = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, stride=stride, bias=True)
        self.norm1 = norm_layer(dim)
        self.act_layer = act_layer()
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        out = self.norm1(self.features(x))
        out = self.act_layer(out)
        return self.dropout(out)


class ConvNetwork(nn.Module):
    """Baseline CNN: stride-2 conv blocks, global average pool, linear head.

    channels=(64, 128) is the original 2-block model; the default has 4 blocks.
    """

    def __init__(self, channels=(64, 128, 256, 256), in_channels=1, num_classes=2, dropout_p=0.1):
        super().__init__()
        dims = [in_channels, *channels]
        self.blocks = nn.Sequential(*[
            BasicBlock(dims[i], dims[i + 1], 3, stride=2, dropout_p=dropout_p)
            for i in range(len(channels))
        ])
        self.ave_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.blocks(x)
        out = self.ave_pool(x).view(x.size(0), -1)
        return self.classifier(out)   # raw logits, no softmax


# ---------------------------------------------------------------------------
# GFNet (Rao et al., 2023, "Global Filter Networks for Image Classification").
# Own implementation; block layout follows the official code
# (github.com/raoyongming/GFNet).
# ---------------------------------------------------------------------------
import torch


class PatchEmbed(nn.Module):
    """Cut a slice into patches and project each to `embed_dim` numbers (a token)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=192):
        super().__init__()
        self.grid = img_size // patch_size                  # 14
        self.num_patches = self.grid * self.grid            # 196
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                                   # (B, 1, 224, 224)
        x = self.proj(x)                                    # (B, 192, 14, 14)
        return x.flatten(2).transpose(1, 2)                 # (B, 196, 192)


class GlobalFilter(nn.Module):
    """FFT each feature map, multiply by a learned complex filter, inverse FFT."""

    def __init__(self, dim, grid=14):
        super().__init__()
        self.grid = grid
        # rfft2 keeps grid//2 + 1 frequencies on the last axis; last dim 2 = (real, imag)
        self.weight = nn.Parameter(torch.randn(grid, grid // 2 + 1, dim, 2) * 0.02)

    def forward(self, x):                                   # (B, 196, C)
        B, N, C = x.shape
        x = x.float().reshape(B, self.grid, self.grid, C)   # tokens back to a 14x14 grid
        freq = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        freq = freq * torch.view_as_complex(self.weight)    # learned filter, per channel
        x = torch.fft.irfft2(freq, s=(self.grid, self.grid), dim=(1, 2), norm="ortho")
        return x.reshape(B, N, C)


class DropPath(nn.Module):
    """Stochastic depth: randomly skip a block's contribution during training."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep
        return x * mask / keep


class MLP(nn.Module):
    """Per-patch feed-forward network: Linear -> GELU -> Linear."""

    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class GFBlock(nn.Module):
    """x = x + MLP(LN(GlobalFilter(LN(x)))): one residual around the whole block."""

    def __init__(self, dim, grid=14, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.filter = GlobalFilter(dim, grid)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))


class GFNet(nn.Module):
    """Patch embedding + positional embedding -> N GFBlocks -> LN -> mean -> Linear."""

    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=2,
                 embed_dim=192, depth=6, mlp_ratio=4.0, drop=0.0, drop_path_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        grid = self.patch_embed.grid
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop)
        rates = torch.linspace(0, drop_path_rate, depth).tolist()   # deeper = more drop
        self.blocks = nn.ModuleList([
            GFBlock(embed_dim, grid, mlp_ratio, drop, rates[i]) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):                                   # (B, 1, 224, 224)
        x = self.pos_drop(self.patch_embed(x) + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)                        # average over the 196 patches
        return self.head(x)                                 # (B, 2) raw logits
