"""
This script was produced with the help of AI.

Visualise the learned global filters of a trained GFNet, in the style of Rao et al. (2021).

Each GFNet block learns one complex filter per channel over the 14 x 14 token grid
(stored as its 14 x 8 half-spectrum). For every layer this plots:
  - left:  frequency domain - magnitude of the full 14 x 14 spectrum, shifted so the
           lowest frequency (the average) is in the centre. A bright centre and dark
           edges means a low-pass filter (smooth, large-scale patterns); the reverse
           means high-pass (edges, fine detail).
  - right: spatial domain - the equivalent convolution kernel over the token grid
           (inverse FFT of the filter), centred on the origin. Blue and red are
           opposite signs; a sharp central spot means the filter acts locally, a
           spread-out pattern means it mixes information across the whole image.

Reads the filter weights directly from the checkpoint, so it needs no project code.
Each panel is scaled independently, so colours compare shapes, not absolute strength.

Usage (from the project folder):
    python scripts/plot_filters.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

# Settings
CHECKPOINT = Path("runs/gfnet_20261027_144653/best.pt")
OUT = Path("images/gfnet_filters.png")
N_FREQ = 24  # channels shown per layer in the frequency domain
N_SPATIAL = 6  # channels shown per layer in the spatial domain

SEQUENTIAL = (
    LinearSegmentedColormap.from_list(  # magnitude: light (low) -> dark blue (high)
        "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    )
)
DIVERGING = (
    LinearSegmentedColormap.from_list(  # signed kernel: blue (-), grey (0), red (+)
        "div_blue_red", ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#8f1f1e"]
    )
)


def load_filters(path):
    """List of complex filters, one per block, each shaped (h, h // 2 + 1, channels)."""
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    keys = [k for k in state if k.endswith("filter.complex_weight")]
    keys.sort(key=lambda k: int(k.split(".")[1]))  # blocks.<i>.filter.complex_weight
    return [torch.view_as_complex(state[k].float()) for k in keys]


def to_views(weight):
    """Full shifted frequency magnitude and centred spatial kernel, each (channels, h, h)."""
    h = weight.shape[0]
    kernel = torch.fft.irfft2(
        weight, s=(h, h), dim=(0, 1), norm="ortho"
    )  # real, (h, h, C)
    spectrum = torch.fft.fft2(kernel, dim=(0, 1), norm="ortho")  # full (h, h, C)
    freq = torch.fft.fftshift(spectrum.abs(), dim=(0, 1))
    space = torch.fft.fftshift(kernel, dim=(0, 1))
    return freq.permute(2, 0, 1).numpy(), space.permute(2, 0, 1).numpy()


def show(ax, img, cmap, signed):
    if signed:
        lim = np.abs(img).max() or 1.0
        ax.imshow(img, cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
    else:
        lo, hi = img.min(), img.max()
        if hi - lo < 1e-6 * max(
            hi, 1e-12
        ):  # flat filter: avoid magnifying rounding noise
            lo, hi = 0.0, 2 * hi or 1.0
        ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    filters = load_filters(CHECKPOINT)
    depth, channels = len(filters), filters[0].shape[-1]
    n_freq, n_space = min(N_FREQ, channels), min(N_SPATIAL, channels)
    print(
        f"{depth} layers, {channels} channels, filter {tuple(filters[0].shape[:2])} (half-spectrum)"
    )

    cell = 0.42
    height = cell * depth + 0.7
    fig = plt.figure(
        figsize=(cell * (n_freq + n_space) + 2.2, height), facecolor="#fcfcfb"
    )
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[n_freq, n_space],
        wspace=0.06,
        left=0.07,
        right=0.99,
        top=1 - 0.55 / height,
        bottom=0.02,
    )
    left = grid[0].subgridspec(depth, n_freq, wspace=0.08, hspace=0.08)
    right = grid[1].subgridspec(depth, n_space, wspace=0.08, hspace=0.08)

    for layer, weight in enumerate(filters):
        freq, space = to_views(weight)
        for c in range(n_freq):
            ax = fig.add_subplot(left[layer, c])
            show(ax, freq[c], SEQUENTIAL, signed=False)
            if c == 0:
                ax.set_ylabel(
                    f"layer {layer + 1}",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=9,
                    color="#52514e",
                )
        for c in range(n_space):
            show(fig.add_subplot(right[layer, c]), space[c], DIVERGING, signed=True)

    fig.text(
        0.07 + 0.92 * n_freq / (n_freq + n_space) / 2,
        0.995,
        f"Frequency domain: |filter|, first {n_freq} channels (centre = low frequency)",
        ha="center",
        va="top",
        fontsize=10,
        color="#0b0b0b",
    )
    fig.text(
        0.99 - 0.92 * n_space / (n_freq + n_space) / 2,
        0.995,
        f"Spatial domain: kernel,\nfirst {n_space} channels (blue -, red +)",
        ha="center",
        va="top",
        fontsize=10,
        color="#0b0b0b",
    )
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130, facecolor=fig.get_facecolor())
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
