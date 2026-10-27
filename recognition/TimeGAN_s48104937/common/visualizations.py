from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.constants import DEPTH
from common.constants import LEVELS


SOURCES = ("Real", "Synthetic")
LEVEL_LABELS = [f"bid {l}" for l in reversed(LEVELS)] + [f"ask {l}" for l in LEVELS]
SHADES = np.linspace(1.0, 0.35, DEPTH)   # level 1 darkest, deeper levels lighter


# --- Training Plots ---
def plot_losses(embedding_losses: dict, supervisor_losses: dict, joint_losses: dict, path: Path) -> None:
    """One panel per training phase, saved for the README."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, hist, title in [(axes[0], embedding_losses, "Phase 1: embedding (recon MSE)"),
                            (axes[1], supervisor_losses, "Phase 2: supervised (next-step MSE)"),
                            (axes[2], joint_losses, "Phase 3: joint training")]:
        for name, values in hist.items():
            ax.plot(values, label=name)
        ax.set(title=title, xlabel="Epoch")
        ax.legend()
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
