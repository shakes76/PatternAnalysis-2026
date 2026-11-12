"""Improved model: residual 3D U-Net with a 256-channel bottleneck (variant resunet3d_256).

Thin entry point. The architecture is defined in ``improved/unet3d/residual_blocks.py``:
``ResUNet3D`` is the standard ``modules.UNet3D`` with every DoubleConv3D replaced
by a ResidualBlock3D (identity shortcut in the bottleneck (256 -> 256);
1x1x1 projection shortcuts everywhere else, where the width changes).

The only difference from ``unet3d_512.py`` is the variant name below.

Run ``python -m improved.unet3d.unet3d_256`` from the project directory to print a
shape check and parameter count (CPU, one 64^3 patch).
"""

from __future__ import annotations

import torch

from improved.unet3d.residual_blocks import RESUNET3D_VARIANTS, ResUNet3D, build_resunet3d
from modules import count_parameters

VARIANT = "resunet3d_256"


def build_model() -> ResUNet3D:
    return build_resunet3d(VARIANT)


if __name__ == "__main__":
    model = build_model().eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 1, 64, 64, 64))
    print(f"{VARIANT}: bottleneck={RESUNET3D_VARIANTS[VARIANT]} channels")
    print(f"input [1, 1, 64, 64, 64] -> output {list(logits.shape)}")
    print(f"trainable parameters: {count_parameters(model):,}")
