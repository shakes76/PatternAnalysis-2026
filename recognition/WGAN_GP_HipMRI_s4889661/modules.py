"""
modules.py — Network components for the HipMRI GAN project

Contains
--------
* Generator      : noise vector z -> synthetic 1x128x64 MRI slice (shared by
                   the DCGAN baseline and the WGAN-GP model)
* Discriminator  : 1x128x64 slice -> real/fake score (logit). With
                   norm="batch" it is the DCGAN discriminator; with
                   norm="layer" it becomes the WGAN-GP critic.
* weights_init   : DCGAN weight initialisation

Architecture follows DCGAN (Radford, Metz & Chintala, ICLR 2016), adapted
from square 64x64 images to the 2:1 HipMRI aspect ratio (128x64).

Per the assessment spec, this file uses PyTorch only (no NumPy).
"""

import torch
import torch.nn as nn


# =============================================================================
# CONFIGURATION
# =============================================================================

Z_DIM = 100          # length of the input noise vector (DCGAN default)
IMG_CHANNELS = 1     # greyscale MRI
BASE_CH = 64         # channel width multiplier for G and D

# Smallest feature map is 4x2 (height x width). Five 2x upsamples give
# 4x2 -> 8x4 -> 16x8 -> 32x16 -> 64x32 -> 128x64, matching our data.
START_HW = (4, 2)


# =============================================================================
# GENERATOR
# =============================================================================

def _up_block(in_ch, out_ch):
    """Transposed conv that doubles H and W, then BatchNorm + ReLU."""
    return nn.Sequential(
        # kernel 4, stride 2, padding 1 -> output size = 2 x input size
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    """Maps noise z [B, Z_DIM] to an image [B, 1, 128, 64] in [-1, 1]."""

    def __init__(self, z_dim=Z_DIM, base=BASE_CH):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            # --- project noise to a 4x2 feature map ---------------------------
            # z is reshaped to [B, z_dim, 1, 1]; a (4,2) kernel with stride 1
            # expands the 1x1 input to 4x2
            nn.ConvTranspose2d(z_dim, base * 8, kernel_size=START_HW, stride=1,
                               padding=0, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),                      # [B, 512, 4, 2]

            # --- upsample, halving channels each time --------------------------
            _up_block(base * 8, base * 4),              # [B, 256, 8, 4]
            _up_block(base * 4, base * 2),              # [B, 128, 16, 8]
            _up_block(base * 2, base),                  # [B, 64, 32, 16]
            _up_block(base, base // 2),                 # [B, 32, 64, 32]

            # --- final upsample to image, no BatchNorm on the output ----------
            nn.ConvTranspose2d(base // 2, IMG_CHANNELS, kernel_size=4, stride=2,
                               padding=1, bias=False),  # [B, 1, 128, 64]
            nn.Tanh(),   # output range [-1, 1], matching dataset.py scaling
        )

    def forward(self, z):
        # Accept z as [B, z_dim] and add the two spatial dims the convs expect
        return self.net(z.view(z.size(0), self.z_dim, 1, 1))

# =============================================================================
# DISCRIMINATOR / CRITIC
# =============================================================================

def _norm_layer(norm, ch):
    """Normalisation layer choice.

    batch -> DCGAN discriminator (standard).
    layer -> WGAN-GP critic. GroupNorm with 1 group normalises each sample
             over (C, H, W), i.e. LayerNorm; it keeps samples independent,
             which the gradient penalty requires.
    """
    if norm == "batch":
        return nn.BatchNorm2d(ch)
    if norm == "layer":
        return nn.GroupNorm(1, ch)
    raise ValueError(f"Unknown norm: {norm}")


def _down_block(in_ch, out_ch, norm):
    """Strided conv that halves H and W, then normalisation + LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        _norm_layer(norm, out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Discriminator(nn.Module):
    """Maps an image [B, 1, 128, 64] to a score [B, 1].

    Outputs a raw logit (no sigmoid): for DCGAN the sigmoid lives inside
    BCEWithLogitsLoss; for WGAN-GP the critic score is unbounded by design.
    """

    def __init__(self, base=BASE_CH, norm="batch"):
        super().__init__()
        self.net = nn.Sequential(
            # --- first layer: no normalisation (DCGAN convention) -------------
            nn.Conv2d(IMG_CHANNELS, base // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),            # [B, 32, 64, 32]

            # --- downsample, doubling channels each time ----------------------
            _down_block(base // 2, base, norm),         # [B, 64, 32, 16]
            _down_block(base, base * 2, norm),          # [B, 128, 16, 8]
            _down_block(base * 2, base * 4, norm),      # [B, 256, 8, 4]
            _down_block(base * 4, base * 8, norm),      # [B, 512, 4, 2]

            # --- collapse the 4x2 map to a single score -----------------------
            nn.Conv2d(base * 8, 1, kernel_size=START_HW, stride=1, padding=0, bias=False),
        )                                               # [B, 1, 1, 1]

    def forward(self, x):
        return self.net(x).view(x.size(0), 1)           # [B, 1]

# =============================================================================
# WEIGHT INITIALISATION
# =============================================================================

def weights_init(m):
    """DCGAN init: conv weights ~ N(0, 0.02); norm scale ~ N(1, 0.02), shift 0.

    Apply with model.apply(weights_init).
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

# =============================================================================
# HELPERS
# =============================================================================

def count_params(model):
    """Number of trainable parameters (reported in the resource profiling table)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================================
# SELF-TEST: SHAPES, OUTPUT RANGE, PARAMETER COUNTS
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    B = 8

    G = Generator()
    D_dcgan = Discriminator(norm="batch")
    D_wgan = Discriminator(norm="layer")
    for m in (G, D_dcgan, D_wgan):
        m.apply(weights_init)

    # --- Generator: noise -> image of the right shape and range ---------------
    z = torch.randn(B, Z_DIM)
    fake = G(z)
    print("G output:", tuple(fake.shape), f"range [{fake.min():.2f}, {fake.max():.2f}]")
    assert fake.shape == (B, 1, 128, 64), "generator output shape wrong"
    assert fake.min() >= -1 and fake.max() <= 1, "tanh range violated"

    # --- Discriminators: image -> one score per sample -------------------------
    for name, D in (("DCGAN D", D_dcgan), ("WGAN critic", D_wgan)):
        score = D(fake)
        print(f"{name} output:", tuple(score.shape))
        assert score.shape == (B, 1), f"{name} output shape wrong"

    # --- Gradients flow end-to-end (G -> D -> scalar) --------------------------
    D_dcgan(G(z)).mean().backward()
    assert all(p.grad is not None for p in G.parameters()), "no gradient reached G"

    # --- Parameter counts (for the README profiling table) ---------------------
    print(f"Params  G: {count_params(G):,}  D(batch): {count_params(D_dcgan):,}  "
          f"D(layer): {count_params(D_wgan):,}")
    print("All modules.py checks passed.")