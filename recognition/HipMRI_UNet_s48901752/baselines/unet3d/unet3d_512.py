"""Standard 3D U-Net with a 512-channel bottleneck.

Thin entry point. The architecture is defined once in ``modules.py`` (3D U-Net part):

    3D.2  Convolution blocks   DoubleConv3D
    3D.3  Encoder              EncoderStage3D, Encoder3D   (32 -> 64 -> 128 -> 256)
    3D.4  Bottleneck           Bottleneck3D                (256 -> 512 for this model)
    3D.5  Decoder              DecoderStage3D, Decoder3D   (512 -> 256 -> 128 -> 64 -> 32)
    3D.6  Skip connections     concatenate_skip (shared with 2D)
    3D.7  Complete model       UNet3D

The only difference from ``unet3d_256.py`` is the variant name below.

Run ``python -m baselines.unet3d.unet3d_512`` from the project directory to print a
shape check and parameter count (CPU, one 64^3 patch).
"""

from __future__ import annotations

import torch

from modules import UNET3D_VARIANTS, UNet3D, build_model3d, count_parameters

VARIANT = "unet3d_512"


def build_model() -> UNet3D:
    return build_model3d(VARIANT)


if __name__ == "__main__":
    model = build_model().eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 1, 64, 64, 64))
    print(f"{VARIANT}: bottleneck={UNET3D_VARIANTS[VARIANT]} channels")
    print(f"input [1, 1, 64, 64, 64] -> output {list(logits.shape)}")
    print(f"trainable parameters: {count_parameters(model):,}")
