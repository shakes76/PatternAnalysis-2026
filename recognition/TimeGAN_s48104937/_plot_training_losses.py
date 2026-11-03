"""Plot TimeGAN training losses from a JSON history saved by any of the training scripts.

One panel per phase found in the file, so it works on the autoencoder-only
(phase 1), supervisor (phases 1-2) and full (phases 1-3) histories. Phase 3 puts
every loss on one log axis, since the adversarial losses (~1) and the auxiliary
terms (~0.05-0.5) differ in scale.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

LOSSES_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/recent_primary_losses.json")
OUT_PATH = LOSSES_PATH.with_suffix(".png")   # figure saved beside the JSON, with the same name
PANEL_SIZE = (4, 3.6)                        # inches per panel; 3 panels = 12 x 3.6, as plot_book_pair_v2

# Last epoch to show per phase; phases not listed are shown in full
MAX_EPOCHS = {"supervised": 15}

# Phases 1 and 2: key -> (title, y label)
TRAIN_VAL = {
    "embedding": ("Phase 1: Autoencoder", "Reconstruction MSE"),
    "supervised": ("Phase 2: Supervised", "Next-step MSE"),
}

# Phase-3 series: key -> display name
JOINT = {
    "d_loss": "Discriminator",
    "g_adv": "Generator",
    "g_spread_w1": "Spread W1",
    "g_mid_w1": "Mid W1",
    "g_moment": "Moment",
}


def style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))   # whole, round epochs
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)


def epochs(values: list) -> range:
    return range(1, len(values) + 1)


def has_data(history: dict, phase: str) -> bool:
    """True if the phase is in the file and at least one of its series has an epoch."""
    return any(len(v) > 0 for v in history.get(phase, {}).values())


def main() -> None:
    with open(LOSSES_PATH) as f:
        history = json.load(f)

    # Cut phases down to their MAX_EPOCHS before plotting
    for phase, last in MAX_EPOCHS.items():
        for name, values in history.get(phase, {}).items():
            history[phase][name] = values[:last]

    phases = [p for p in (*TRAIN_VAL, "joint") if has_data(history, p)]
    if not phases:
        raise SystemExit(f"No loss history found in {LOSSES_PATH}")

    fig, axes = plt.subplots(1, len(phases), figsize=(PANEL_SIZE[0] * len(phases), PANEL_SIZE[1]),
                             layout="constrained", squeeze=False)

    for ax, phase in zip(axes[0], phases):
        if phase in TRAIN_VAL:
            # Phases 1 and 2: train and validation
            for split, colour in (("train", "C0"), ("val", "C1")):
                values = history[phase][split]
                ax.plot(epochs(values), values, color=colour, lw=1.5, label=split.capitalize())
            style(ax, *TRAIN_VAL[phase])
            ax.legend(fontsize=8, frameon=False)
        else:
            # Phase 3: every loss on one log axis
            joint = history["joint"]
            for name, label in JOINT.items():
                if joint.get(name):
                    ax.plot(epochs(joint[name]), joint[name], lw=1.5, label=label)
            style(ax, "Phase 3: Joint", "Loss")
            # Legend inside the panel so it adds no width; extra headroom above the losses makes space for it
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * 4)
            ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left", handlelength=1.6, columnspacing=1.0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH} ({', '.join(phases)})")


if __name__ == "__main__":
    main()