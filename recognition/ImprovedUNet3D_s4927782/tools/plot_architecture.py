"""Draw the implemented base_channels=16, six-class models from modules.py.

This is a structural illustration, not a measured performance comparison.
Regenerate from the project directory with python tools/plot_architecture.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#183046"
MUTED = "#516777"
BLUE = "#2876A6"
TEAL = "#0E7A77"
ORANGE = "#B26A1C"
ENCODER_X, DECODER_X, HEAD_X = 2.6, 8.2, 12.1
LEVEL_Y = (5.5, 4.1, 2.7, 1.3)
BLOCK_WIDTH, BLOCK_HEIGHT = 2.25, 0.76


def box(axis, x, y, text, *, color=BLUE, fill="#EAF3F9", width=BLOCK_WIDTH,
        height=BLOCK_HEIGHT, dashed=False, fontsize=13):
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.09", linewidth=1.5,
        facecolor=fill, edgecolor=color, linestyle="--" if dashed else "-",
        zorder=3,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=fontsize,
              color=INK, linespacing=1.35, zorder=4)


def arrow(axis, start, end, *, color=INK, dashed=False, connectionstyle="arc3"):
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.45,
        color=color, linestyle="--" if dashed else "-",
        connectionstyle=connectionstyle, zorder=2,
    ))


def panel(axis, dimensions):
    is_3d = dimensions == 3
    color, fill = (TEAL, "#E7F4F1") if is_3d else (BLUE, "#EAF3F9")
    axis.set(xlim=(0, 14.5), ylim=(-0.25, 7.25))
    axis.axis("off")
    title = "B   3D Improved U-Net" if is_3d else "A   Standard 2D U-Net baseline"
    subtitle = "Residual units + deep supervision | one whole cached volume" if is_3d else "Independent axial slices | predictions reconstructed into the same volume grid"
    axis.text(0.15, 7.02, title, fontsize=21, weight="bold", color=INK)
    axis.text(0.15, 6.65, subtitle, fontsize=13, color=MUTED)
    spatial = (("64 × 128 × 128", "32 × 64 × 64", "16 × 32 × 32", "8 × 16 × 16")
               if is_3d else ("128 × 128", "64 × 64", "32 × 32", "16 × 16"))
    axis.text(ENCODER_X, 6.14, "Input: 1 × " + spatial[0], ha="center", fontsize=13, color=INK)
    arrow(axis, (ENCODER_X, 6.02), (ENCODER_X, LEVEL_Y[0] + BLOCK_HEIGHT / 2 + 0.02))
    for level, (channels, y) in enumerate(zip((16, 32, 64, 128), LEVEL_Y)):
        box(axis, ENCODER_X, y, f"{channels} channels\n{spatial[level]}", color=color, fill=fill)
        if level < 3:
            arrow(axis, (ENCODER_X, y - BLOCK_HEIGHT / 2 - 0.03),
                  (ENCODER_X, LEVEL_Y[level + 1] + BLOCK_HEIGHT / 2 + 0.03), color=color)
            axis.text(ENCODER_X - 0.18, y - 0.71, "pool ×2", ha="right", va="center", fontsize=11, color=MUTED)
    for level in (2, 1, 0):
        y = LEVEL_Y[level]
        box(axis, DECODER_X, y, f"{(16, 32, 64)[level]} channels\n{spatial[level]}", color=color, fill=fill)
        arrow(axis, (ENCODER_X + BLOCK_WIDTH / 2 + 0.04, y),
              (DECODER_X - BLOCK_WIDTH / 2 - 0.04, y), color=MUTED, dashed=True)
        axis.text((ENCODER_X + DECODER_X) / 2, y + 0.14, "skip concatenation", ha="center", fontsize=11.5, color=MUTED)
    # First decoder step uses a routed arrow below the skip-connection line.
    arrow(axis, (ENCODER_X + BLOCK_WIDTH / 2 + 0.04, LEVEL_Y[3]),
          (DECODER_X, LEVEL_Y[2] - BLOCK_HEIGHT / 2 - 0.03), color=color,
          connectionstyle="angle,angleA=0,angleB=-90,rad=8")
    up_label = "trilinear resize ×2\n+ 3×3×3 Conv3d" if is_3d else "2×2 ConvTranspose2d\nstride 2"
    axis.text(5.55, 1.53, up_label, ha="center", va="bottom", fontsize=11.5, color=MUTED)
    for lower, upper in ((2, 1), (1, 0)):
        arrow(axis, (DECODER_X, LEVEL_Y[lower] + BLOCK_HEIGHT / 2 + 0.03),
              (DECODER_X, LEVEL_Y[upper] - BLOCK_HEIGHT / 2 - 0.03), color=color)
        axis.text(DECODER_X + 0.18, (LEVEL_Y[lower] + LEVEL_Y[upper]) / 2,
                  "resize + conv" if is_3d else "transposed conv", fontsize=11,
                  va="center", color=MUTED)
    main_label = ("1×1×1 Conv → 6 classes\nmain · full resolution · w = 1"
                  if is_3d else "1×1 Conv → 6 classes\nsingle full-resolution head")
    box(axis, HEAD_X, LEVEL_Y[0], main_label, color=color, fill=fill, width=3.05, height=0.86, fontsize=12.5)
    arrow(axis, (DECODER_X + BLOCK_WIDTH / 2 + 0.04, LEVEL_Y[0]),
          (HEAD_X - 3.05 / 2 - 0.04, LEVEL_Y[0]), color=color)
    axis.text(HEAD_X, 6.14, "MAIN: TRAINING + INFERENCE", ha="center", fontsize=11.5, weight="bold", color=color)
    if is_3d:
        for level, text in ((1, "½ resolution · w = 0.5"), (2, "¼ resolution · w = 0.25")):
            box(axis, HEAD_X, LEVEL_Y[level], "1×1×1 Conv → 6 classes\n" + text,
                color=ORANGE, fill="#FFF4E5", width=3.05, height=0.86, dashed=True, fontsize=12.5)
            arrow(axis, (DECODER_X + BLOCK_WIDTH / 2 + 0.04, LEVEL_Y[level]),
                  (HEAD_X - 3.05 / 2 - 0.04, LEVEL_Y[level]), color=ORANGE, dashed=True)
        axis.text(HEAD_X, 2.02, "AUXILIARY: TRAINING LOSS ONLY", ha="center", fontsize=11, weight="bold", color=ORANGE)
        axis.text(9.8, 1.10, r"$L_{3D}=(L_{main}+0.5L_{half}+0.25L_{quarter})/1.75$",
                  ha="center", fontsize=14, color=INK)
        axis.text(0.15, 0.45, "Residual unit: Conv3d → GroupNorm → LeakyReLU → Conv3d → GroupNorm;", fontsize=12, color=MUTED)
        axis.text(0.15, 0.12, "add identity / 1×1×1 shortcut → LeakyReLU. Downsampling: 2×2×2 MaxPool3d.", fontsize=12, color=MUTED)
    else:
        axis.text(HEAD_X, 4.55, "argmax over 6 class logits\n→ one class ID per pixel", ha="center", fontsize=12.5, color=INK)
        axis.text(HEAD_X, 3.55, "Reassemble axial predictions\nfor whole-volume evaluation", ha="center", fontsize=12.5, color=MUTED)
        axis.text(9.8, 1.10, r"$L_{2D}=L_{main}$", ha="center", fontsize=15, color=INK)
        axis.text(0.15, 0.45, "Feature block: 3×3 Conv2d → ReLU → 3×3 Conv2d → ReLU.", fontsize=12.5, color=MUTED)
        axis.text(0.15, 0.12, "Downsampling: 2×2 MaxPool2d. Upsampling: transposed convolution.", fontsize=12.5, color=MUTED)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 1, figsize=(14.5, 14.2))
    figure.patch.set_facecolor("white")
    panel(axes[0], 2)
    panel(axes[1], 3)
    figure.text(0.055, 0.025,
                "Shared patient splits and 64 × 128 × 128 resampled RAS grid. Feature-map sizes omit batch; 3D sizes are D × H × W.\n"
                "Each head loss: TRAIN-weighted categorical CE + soft multiclass Dice. Auxiliary targets use nearest-neighbour resizing; inference uses main argmax only.",
                fontsize=11.8, color=MUTED, linespacing=1.6)
    figure.subplots_adjust(left=0.035, right=0.985, top=0.985, bottom=0.078, hspace=0.11)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        output = args.output_dir / f"model_architecture.{suffix}"
        figure.savefig(output, dpi=args.dpi, facecolor="white")
        print(output)
    plt.close(figure)


if __name__ == "__main__":
    main()
