"""The README's summary figures, drawn from the committed records in docs/ (local).

    python tools/figure_report.py --output docs/figures

architecture.png      the SD1.5 pipeline and what E2 and E3 train
progression-ssim.png  validation SSIM from SD to E3: what each step changed
loss-curves.png       training and validation loss of E2, E3 and B3 by epoch
ssim-box.png          per-sample test SSIM of every model, with the 0.6 target
kid-coverage.png      test KID and coverage of every model, with the real floor
mia-timestep.png      membership AUC at each noise step
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"
# Fixed categorical order (one hue per model family), from the reference palette.
COLOUR = {
    "e2": "#2a78d6",
    "e3": "#eb6834",
    "b3": "#1baf7a",
    "b1": "#eda100",
    "b2": "#e87ba4",
    "e0": "#008300",
    "sd": "#6250d6",
    "e1": "#8a8780",
}
COLOUR.update({"r1e2": COLOUR["e2"], "r1e3": COLOUR["e3"]})
NAME = {
    "sd": "SD, no training",
    "e0": "First LoRA (E0)",
    "e1": "Full fine-tune + recipe (E1)",
    "e2": "Full fine-tune, cleaned (E2)",
    "r1e2": "E2, 2nd seed",
    "e3": "LoRA adapter, recipe (E3)",
    "r1e3": "E3, 2nd seed",
    "b3": "From scratch (B3)",
    "b1": "VAE (B1)",
    "b2": "GAN (B2)",
}
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dde3"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)


def save(figure, path):
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print("wrote", path)


def box(axis, x, y, w, h, text, fill="#f6f8fa", edge=MUTED, bold=False, size=9):
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.2,
        )
    )
    axis.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        fontweight="bold" if bold else "normal",
        wrap=True,
    )


def arrow(axis, start, end, colour=MUTED, style="-|>", dashed=False):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            color=colour,
            linewidth=1.3,
            linestyle="--" if dashed else "-",
        )
    )


def architecture(output):
    """Pipeline with the UNet drawn as a U of stages; trainable parts marked."""
    figure, axis = plt.subplots(figsize=(11.5, 5.6))
    axis.set(xlim=(0, 23.0), ylim=(0, 10.6))
    axis.axis("off")
    frozen, trained = "#eef1f4", "#dbe9fb"
    axis.text(
        0.2,
        10.2,
        "Training: learn to predict the noise added to a latent.  Generation: start from noise, "
        "remove it in 30 DDIM steps with guidance 7.5, decode.",
        fontsize=9,
        color=INK,
    )
    # Image path.
    box(axis, 0.2, 6.9, 2.4, 1.2, "MRI slice\n256 x 256")
    box(axis, 3.2, 6.9, 2.6, 1.2, "VAE encoder\n(frozen, 84 M)", fill=frozen)
    box(axis, 6.4, 6.9, 2.6, 1.2, "latent z0\n4 x 32 x 32")
    box(axis, 6.4, 4.5, 2.6, 1.4, "add noise for\nstep t: z_t", size=8.5)
    arrow(axis, (2.6, 7.5), (3.2, 7.5))
    arrow(axis, (5.8, 7.5), (6.4, 7.5))
    arrow(axis, (7.7, 6.9), (7.7, 5.9))
    # The UNet: four down stages, a middle block, four up stages joined by skips.
    channels = (320, 640, 1280, 1280)
    x0, w, h, top, dy = 9.8, 0.95, 0.95, 7.1, 1.2

    def stage(x, y, label, attention):
        box(
            axis,
            x,
            y,
            w,
            h,
            label + ("\nattn" if attention else ""),
            fill=trained,
            edge=COLOUR["e2"],
            size=7.5,
        )
        if attention:  # where the E3 adapters sit
            axis.add_patch(
                FancyBboxPatch(
                    (x + w - 0.26, y + h - 0.24),
                    0.16,
                    0.14,
                    boxstyle="square,pad=0",
                    facecolor=COLOUR["e3"],
                    edgecolor="none",
                )
            )

    for i, c in enumerate(channels):
        y = top - i * dy
        stage(x0 + i * 1.0, y, str(c), i < 3)
        stage(x0 + 8.0 - i * 1.0, y, str(c), i < 3)
        axis.plot(
            [x0 + i * 1.0 + w, x0 + 8.0 - i * 1.0],
            [y + h / 2] * 2,
            color=GRID,
            linewidth=1,
            linestyle=":",
        )
    stage(x0 + 3.9, top - 4 * dy, "mid", True)
    axis.text(
        x0 + 4.45,
        9.35,
        "Denoising UNet (860 M parameters)",
        ha="center",
        fontsize=10,
        color=INK,
        fontweight="bold",
    )
    axis.text(
        x0 + 4.45,
        8.85,
        "down stages, middle block, up stages; dotted lines are skip connections",
        ha="center",
        fontsize=7.5,
        color=MUTED,
    )
    arrow(axis, (9.0, 5.2), (x0, top + h / 2))
    box(axis, 19.4, 6.9, 2.8, 1.2, "predicted\nnoise", fill="#ffffff")
    arrow(axis, (x0 + 8.0 + w, top + h / 2), (19.4, 7.5))
    box(axis, 19.4, 4.7, 2.8, 1.3, "MSE loss against\nthe true noise", size=8.5)
    arrow(axis, (20.8, 6.9), (20.8, 6.0))
    # Step embedding.
    box(
        axis,
        3.0,
        3.0,
        3.0,
        1.1,
        "step t\n(sinusoidal embedding)",
        size=7.5,
        fill="#ffffff",
    )
    arrow(axis, (6.0, 3.55), (x0 + 3.0, top - 3 * dy + 0.3), dashed=True)
    axis.text(6.3, 3.0, "enters every residual block", fontsize=7.5, color=MUTED)
    # Prompt path.
    box(
        axis,
        0.2,
        0.7,
        3.2,
        1.2,
        '"a brain MRI slice,\nAlzheimer disease"',
        fill="#ffffff",
        size=8.5,
    )
    box(axis, 4.0, 0.7, 3.0, 1.2, "CLIP text encoder\n(frozen, 123 M)", fill=frozen)
    box(axis, 7.6, 0.7, 2.7, 1.2, "77 x 768\ntoken embeddings")
    arrow(axis, (3.4, 1.3), (4.0, 1.3))
    arrow(axis, (7.0, 1.3), (7.6, 1.3))
    arrow(axis, (10.3, 1.3), (x0 + 4.35, top - 4 * dy), colour=COLOUR["e3"])
    axis.text(
        10.5,
        0.45,
        "cross-attention: attn blocks read the prompt",
        fontsize=7.5,
        color=MUTED,
    )
    # Key: what each model trains.
    axis.add_patch(
        FancyBboxPatch(
            (15.9, 0.4),
            6.75,
            2.6,
            boxstyle="round,pad=0.05",
            facecolor="white",
            edgecolor=GRID,
        )
    )
    axis.add_patch(
        FancyBboxPatch(
            (16.1, 2.2),
            0.5,
            0.45,
            boxstyle="round,pad=0.02",
            facecolor=trained,
            edgecolor=COLOUR["e2"],
        )
    )
    axis.text(
        16.8,
        2.42,
        "E2 trains every UNet weight (860 M)",
        fontsize=8.5,
        va="center",
        color=INK,
    )
    axis.add_patch(
        FancyBboxPatch(
            (16.25, 1.48),
            0.22,
            0.18,
            boxstyle="square,pad=0",
            facecolor=COLOUR["e3"],
            edgecolor="none",
        )
    )
    axis.text(
        16.8,
        1.57,
        "E3 trains only rank-4 LoRA in q, k, v, out\nof each attn block (0.8 M)",
        fontsize=8.5,
        va="center",
        color=INK,
    )
    axis.add_patch(
        FancyBboxPatch(
            (16.1, 0.62),
            0.5,
            0.45,
            boxstyle="round,pad=0.02",
            facecolor=frozen,
            edgecolor=MUTED,
        )
    )
    axis.text(16.8, 0.84, "frozen in both", fontsize=8.5, va="center", color=INK)
    save(figure, output / "architecture.png")


def progression(output):
    ssim = json.loads((DOCS / "e3-ssim-result-2026-10-17.json").read_text())
    groups = ssim["groups"]
    steps = [
        ("sd", "base", "SD,\nno training"),
        ("e0", "adapted", "First LoRA\n(E0)"),
        ("e1", "e1e10", "+ full fine-tune\n+ recipe (E1)"),
        ("e2", "e2e10", "+ remove 2\nbad scans (E2)"),
        ("e3", "e3e10", "E2 recipe, small\nadapter (E3)"),
    ]
    values = [groups[key]["all"]["nearest_ssim"]["mean"] for _, key, _ in steps]
    real = ssim["real_validation_to_training_feature_neighbor"]["mean"]
    figure, axis = plt.subplots(figsize=(8, 3.8))
    bars = axis.bar(
        range(len(steps)), values, width=0.6, color=[COLOUR[m] for m, _, _ in steps]
    )
    for bar, value in zip(bars, values):
        inside = value > 0.15
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.04 if inside else value + 0.012,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
            color="white" if inside else INK,
            fontweight="bold",
        )
    axis.axhline(0.6, color=MUTED, linestyle="--", linewidth=1)
    axis.text(
        -0.4, 0.585, "course target 0.6", ha="left", va="top", fontsize=8, color=MUTED
    )
    axis.axhline(real, color=INK, linestyle=":", linewidth=1)
    axis.text(-0.4, real + 0.012, f"real slices {real:.3f}", fontsize=8, color=INK)
    axis.set_xlim(-0.55, len(steps) - 0.45)
    axis.set_xticks(range(len(steps)), [label for _, _, label in steps], fontsize=8.5)
    axis.set(
        ylabel="feature-nearest SSIM (validation)",
        ylim=(0, 0.8),
        title="Each step's change: the recipe and data cleaning carry the gain, not model size",
    )
    axis.grid(axis="x", visible=False)
    save(figure, output / "progression-ssim.png")


def read_epochs(name):
    with (DOCS / "training-curves" / f"{name}-epochs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    return (
        [int(r["epoch"]) for r in rows],
        [float(r["train_loss"]) for r in rows],
        [float(r["val_loss_ema"]) for r in rows],
    )


def loss_curves(output):
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=False)
    figure.subplots_adjust(wspace=0.32)
    limits = {"e2": None, "e3": None, "b3": (0.03, 0.13)}
    for axis, name in zip(axes, ("e2", "e3", "b3")):
        epochs, train, validation = read_epochs(name)
        axis.plot(epochs, train, color=COLOUR[name], linewidth=2, label="training loss")
        axis.plot(
            epochs,
            validation,
            color=COLOUR[name],
            linewidth=2,
            linestyle="--",
            label="validation loss (EMA)",
        )
        best = min(range(len(validation)), key=validation.__getitem__)
        axis.scatter(
            [epochs[best]],
            [validation[best]],
            s=36,
            color=COLOUR[name],
            zorder=3,
            edgecolor="white",
            linewidth=1.5,
        )
        axis.annotate(
            f"validation lowest\nat epoch {epochs[best]}",
            (epochs[best], validation[best]),
            textcoords="offset points",
            xytext=(-8, 28) if name == "e3" else (-8, 14),
            ha="right" if name != "b3" else "left",
            fontsize=8,
            color=INK,
        )
        axis.set(title=NAME[name], xlabel="epoch", ylabel="noise-prediction MSE")
        if limits[name]:
            axis.set_ylim(*limits[name])
            axis.text(
                0.98,
                0.04,
                "epoch 1 (training 0.39) off scale",
                transform=axis.transAxes,
                ha="right",
                fontsize=7.5,
                color=MUTED,
            )
        axis.legend(fontsize=8, loc="upper right")
    figure.suptitle(
        "Fine-tuned models keep training and validation loss together; the model trained from scratch "
        "overfits after epoch 23",
        fontsize=10,
        y=1.02,
    )
    save(figure, output / "loss-curves.png")


ORDER = ("sd", "e0", "e2", "r1e2", "e3", "r1e3", "b3", "b1", "b2")


def ssim_box(output):
    values = json.loads((DOCS / "test-ssim-per-sample-2026-10-19.json").read_text())[
        "models"
    ]
    floor = json.loads((DOCS / "test-ssim-2026-10-19.json").read_text())["real_floor"][
        "mean"
    ]
    figure, axis = plt.subplots(figsize=(10, 4))
    data = [values[m] for m in ORDER]
    parts = axis.boxplot(
        data,
        orientation="vertical",
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        flierprops={
            "marker": ".",
            "markersize": 3,
            "alpha": 0.4,
            "markeredgecolor": MUTED,
        },
        medianprops={"color": INK, "linewidth": 1.5},
    )
    for patch, model in zip(parts["boxes"], ORDER):
        patch.set(
            facecolor=COLOUR[model],
            alpha=0.55 if model.startswith("r1") else 0.85,
            edgecolor=MUTED,
        )
    axis.axhline(0.6, color=MUTED, linestyle="--", linewidth=1)
    axis.text(
        9.45,
        0.605,
        "course target 0.6",
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    axis.axhline(floor, color=INK, linestyle=":", linewidth=1)
    axis.text(0.55, floor + 0.01, f"real slices {floor:.3f}", fontsize=8, color=INK)
    axis.set_xticks(
        range(1, len(ORDER) + 1),
        [NAME[m] for m in ORDER],
        rotation=25,
        ha="right",
        fontsize=8.5,
    )
    axis.set(
        ylabel="feature-nearest SSIM per sample",
        ylim=(0, 0.85),
        title="Test SSIM per sample (1,024 per model): the blurry VAE scores highest, so SSIM alone cannot rank models",
    )
    axis.grid(axis="x", visible=False)
    save(figure, output / "ssim-box.png")


def kid_coverage(output):
    metrics = json.loads((DOCS / "test-metrics-2026-10-19.json").read_text())
    floor = metrics["floor"]["inception"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    positions = range(len(ORDER))
    kid = [100 * metrics["models"][m]["inception"]["kid"]["mean"] for m in ORDER]
    kid_sd = [100 * metrics["models"][m]["inception"]["kid"]["sd"] for m in ORDER]
    left.bar(
        positions,
        kid,
        yerr=kid_sd,
        width=0.6,
        color=[COLOUR[m] for m in ORDER],
        capsize=2,
        error_kw={"ecolor": MUTED, "linewidth": 1},
    )
    left.set_yscale("log")
    left.axhline(100 * floor["kid"]["mean"], color=INK, linestyle=":", linewidth=1)
    left.text(
        0.99,
        0.97,
        f"dotted line: real slices {100 * floor['kid']['mean']:.2f}",
        transform=left.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=INK,
    )
    for x, value in zip(positions, kid):
        left.text(
            x,
            value * 1.25,
            f"{value:.2f}" if value < 10 else f"{value:.0f}",
            ha="center",
            fontsize=7.5,
            color=INK,
        )
    left.set(
        ylabel="KID x 100 (log scale, lower is better)",
        title="Distance from the real distribution",
    )
    coverage = [
        metrics["models"][m]["inception"]["prdc"]["coverage"]["mean"] for m in ORDER
    ]
    coverage_sd = [
        metrics["models"][m]["inception"]["prdc"]["coverage"]["sd"] for m in ORDER
    ]
    right.bar(
        positions,
        coverage,
        yerr=coverage_sd,
        width=0.6,
        color=[COLOUR[m] for m in ORDER],
        capsize=2,
        error_kw={"ecolor": MUTED, "linewidth": 1},
    )
    right.axhline(
        floor["prdc"]["coverage"]["mean"], color=INK, linestyle=":", linewidth=1
    )
    right.text(
        -0.4,
        floor["prdc"]["coverage"]["mean"] + 0.02,
        f"real slices {floor['prdc']['coverage']['mean']:.2f}",
        fontsize=8,
        color=INK,
    )
    for x, value in zip(positions, coverage):
        right.text(
            x, value + 0.02, f"{value:.2f}", ha="center", fontsize=7.5, color=INK
        )
    right.set(
        ylabel="coverage (higher is better)",
        ylim=(0, 1),
        title="Share of real neighbourhoods the samples reach",
    )
    for axis in (left, right):
        axis.set_xticks(
            list(positions),
            [NAME[m] for m in ORDER],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        axis.grid(axis="x", visible=False)
    save(figure, output / "kid-coverage.png")


def mia_timestep(output):
    membership = json.loads((DOCS / "test-metrics-2026-10-19.json").read_text())[
        "membership"
    ]
    figure, axis = plt.subplots(figsize=(8, 3.8))
    for model in ("b3", "e2", "r1e2", "e3", "r1e3", "sd"):
        by_t = membership[model]["auc_by_timestep"]
        steps = sorted(int(t) for t in by_t)
        axis.plot(
            steps,
            [by_t[str(t)] for t in steps],
            color=COLOUR[model],
            linewidth=2,
            linestyle="--" if model.startswith("r1") else "-",
            marker="o" if not model.startswith("r1") else None,
            markersize=4,
            label=NAME[model],
        )
    axis.axhline(0.5, color=MUTED, linewidth=1)
    axis.text(
        60, 0.48, "0.5 = no leak (chance)", ha="left", va="top", fontsize=8, color=MUTED
    )
    axis.set(
        xlabel="noise step t (of 1,000)",
        ylabel="membership AUC",
        ylim=(0.4, 1.02),
        title="Where the loss gives training slices away: mostly at low noise",
    )
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(figure, output / "mia-timestep.png")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=DOCS / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    architecture(args.output)
    progression(args.output)
    loss_curves(args.output)
    ssim_box(args.output)
    kid_coverage(args.output)
    mia_timestep(args.output)


if __name__ == "__main__":
    main()
