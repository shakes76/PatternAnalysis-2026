"""Model B: standard 2D U-Net with a 512-channel bottleneck.

Thin entry point. The architecture is defined once in ``modules.py``:

    Section 2  Convolution blocks   DoubleConv
    Section 3  Encoder              EncoderStage, Encoder   (32 -> 64 -> 128 -> 256)
    Section 4  Bottleneck           Bottleneck              (256 -> 512 for this model)
    Section 5  Decoder              DecoderStage, Decoder   (512 -> 256 -> 128 -> 64 -> 32)
    Section 6  Skip connections     concatenate_skip
    Section 7  Complete model       UNet2D

The only difference from ``unet2d_256.py`` is the variant name below.

Run ``python -m baselines.unet2d.unet2d_512`` from the project directory to print a
shape check and parameter count.
"""

from __future__ import annotations

import torch

from modules import UNET2D_VARIANTS, UNet2D, count_parameters
from modules import build_model as _build_variant

VARIANT = "unet2d_512"


def build_model() -> UNet2D:
    return _build_variant(VARIANT)


if __name__ == "__main__":
    model = build_model().eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 1, 256, 256))
    print(f"{VARIANT}: bottleneck={UNET2D_VARIANTS[VARIANT]} channels")
    print(f"input [1, 1, 256, 256] -> output {list(logits.shape)}")
    print(f"trainable parameters: {count_parameters(model):,}")
