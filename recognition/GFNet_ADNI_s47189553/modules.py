"""
modules.py — models for ADNI AD vs NC classification.

"""

import torch
import torch.nn as nn
 
# --------------------------------------------------------------------------- #
# Model settings
# --------------------------------------------------------------------------- #
IMG_SIZE = 224                  # must match dataset.py img_size
IN_CHANS = 1                    # 1 = grayscale MRI, 3 = repeated channel for pretrained weights
NUM_CLASSES = 2                 # NC, AD
PATCH_SIZE = 16                 # 16x16 patches -> 14x14 tokens at 224
EMBED_DIM = 256                 # GFNet-Ti
DEPTH = 12                      # GFNet-Ti
MLP_RATIO = 4.0                 # FFN expansion
DROP_RATE = 0.0                 # dropout in FFN / head
DROP_PATH_RATE = 0.1            # stochastic depth (max value, linearly increased over blocks)
LAYER_SCALE_INIT = None         # e.g. 1e-6 to enable LayerScale (deeper models); None = off
INIT_STD = 0.02                 # trunc-normal std for Linear / pos-embed / filters
 
 
# --------------------------------------------------------------------------- #
# Building blocks — token tensors are channels-last: (B, H, W, C)
# --------------------------------------------------------------------------- #
class DropPath(nn.Module):
    """Stochastic depth: randomly drop the whole residual branch per sample."""
 
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep
 
 
class GlobalFilter(nn.Module):
    """Global filter layer (Algorithm 1):  x <- IFFT2( K ⊙ FFT2(x) ).
 
    K is a learnable complex filter of shape (H, W//2 + 1, C) — one weight per
    frequency per channel. rfft2 exploits conjugate symmetry of real inputs, so
    only half the frequencies along W are stored (Eq. 7).
    """
 
    def __init__(self, dim: int, h: int, w: int):
        super().__init__()
        self.h, self.w = h, w
        self.complex_weight = nn.Parameter(
            torch.randn(h, w // 2 + 1, dim, 2, dtype=torch.float32) * INIT_STD
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, H, W, C)
        B, H, W, C = x.shape
        if (H, W) != (self.h, self.w):
            raise ValueError(
                f"GlobalFilter built for {self.h}x{self.w} tokens, got {H}x{W}. "
                f"Use the img_size the model was built with."
            )
        dtype, device = x.dtype, x.device
        x = x.float()
        if device.type == "mps":
            x = x.cpu()      # Apple GPU lacks FFT / complex ops: run just this filter on CPU
        weight = torch.view_as_complex(self.complex_weight.to(x.device))
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        return x.to(device=device, dtype=dtype)
 
 
class Mlp(nn.Module):
    """FFN: Linear(D -> rD) -> GELU -> Linear(rD -> D)."""
 
    def __init__(self, dim: int, mlp_ratio: float, drop: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))
 
 
class Block(nn.Module):
    """GFNet block with a single residual (Sec. III-B):
 
        x <- x + DropPath( γ · FFN( LN( GlobalFilter( LN(x) ) ) ) )
 
    γ is LayerScale (only when layer_scale_init is not None).
    """
 
    def __init__(
        self,
        dim: int,
        h: int,
        w: int,
        mlp_ratio: float = MLP_RATIO,
        drop: float = DROP_RATE,
        drop_path: float = 0.0,
        layer_scale_init: float | None = LAYER_SCALE_INIT,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilter(dim, h, w)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop)
        self.drop_path = DropPath(drop_path)
        self.gamma = (
            nn.Parameter(layer_scale_init * torch.ones(dim))
            if layer_scale_init is not None
            else None
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(self.norm2(self.filter(self.norm1(x))))
        if self.gamma is not None:
            y = self.gamma * y
        return x + self.drop_path(y)
 
 
class PatchEmbed(nn.Module):
    """Non-overlapping patch embedding: Conv(k=p, s=p) -> LayerNorm.
    (B, C_in, H, W) -> (B, H/p, W/p, D)."""
 
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x).permute(0, 2, 3, 1))
 
 
def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=INIT_STD)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
 
 
def _drop_path_schedule(rate: float, n: int) -> list[float]:
    """Linearly increase drop-path from 0 (first block) to `rate` (last)."""
    return [rate * i / max(n - 1, 1) for i in range(n)]
 
 
# --------------------------------------------------------------------------- #
# GFNet
# --------------------------------------------------------------------------- #
class GFNet(nn.Module):
    """Isotropic GFNet:
 
        image (B, C, 224, 224)
          -> PatchEmbed 16x16          (B, 14, 14, D)
          -> + positional embedding
          -> DEPTH x Block              (global filter (14, 8) + FFN)
          -> LayerNorm -> global average pool -> Linear(D -> num_classes)
    """
 
    def __init__(
        self,
        img_size: int = IMG_SIZE,
        in_chans: int = IN_CHANS,
        num_classes: int = NUM_CLASSES,
        patch_size: int = PATCH_SIZE,
        embed_dim: int = EMBED_DIM,
        depth: int = DEPTH,
        mlp_ratio: float = MLP_RATIO,
        drop_rate: float = DROP_RATE,
        drop_path_rate: float = DROP_PATH_RATE,
        layer_scale_init: float | None = LAYER_SCALE_INIT,
    ):
        super().__init__()
        if img_size % patch_size:
            raise ValueError(f"img_size must be divisible by patch_size={patch_size}")
        self.img_size = img_size
        res = img_size // patch_size
 
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, res, res, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=INIT_STD)
        self.pos_drop = nn.Dropout(drop_rate)
 
        dpr = _drop_path_schedule(drop_path_rate, depth)
        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim, res, res, mlp_ratio=mlp_ratio, drop=drop_rate,
                    drop_path=dpr[i], layer_scale_init=layer_scale_init,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(_init_weights)
 
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Model built for {self.img_size}x{self.img_size} input, got "
                f"{x.shape[-2]}x{x.shape[-1]}. Match dataset img_size to the model."
            )
        x = self.pos_drop(self.patch_embed(x) + self.pos_embed)   # (B, 14, 14, D)
        x = self.blocks(x)
        return self.norm(x).mean(dim=(1, 2))                       # global average pool
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.head_drop(self.forward_features(x)))


# --------------------------------------------------------------------------- #
# SimpleCNN baseline (runs/simplecnn_*) — kept so predict.py can load it
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU, twice, then 2x2 max-pool (halves H and W)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """Four conv blocks (32 -> 64 -> 128 -> 256 channels) + global pool + classifier.

    Spatial size for a 224x224 input: 224 -> 112 -> 56 -> 28 -> 14 -> pooled to 1x1.
    About 1.2M parameters.
    """

    def __init__(self, in_chans: int = IN_CHANS, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_chans, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)          # [B, 256, h, w] -> [B, 256, 1, 1]
        self.classifier = nn.Sequential(
            nn.Flatten(),                            # -> [B, 256]
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),             # -> [B, num_classes]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
 
# --------------------------------------------------------------------------- #
# Quick shape check: python modules.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    model = GFNet().eval()
    x = torch.randn(2, IN_CHANS, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out = model(x)
    print(f"GFNet params={count_parameters(model) / 1e6:.2f}M  "
          f"input={tuple(x.shape)} -> output={tuple(out.shape)}")