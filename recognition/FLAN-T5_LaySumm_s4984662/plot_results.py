"""
plot_results.py - draw the README figures: the pipeline diagram and the results overview.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
Course : COMP3710 Pattern Recognition, UQ, Semester 2 2026 - Project 2.6
         (FLAN-T5 lay summarisation of radiology reports, LoRA vs full fine-tuning)

Description
    pipeline.png  how a report becomes a lay summary, the three configurations compared and
                  how they are evaluated (static drawing).
    results.png   four panels read from the `metrics.json` files that train.py wrote:
                    (a) ROUGE-Lsum on the three validation tiers,
                    (b) dev ROUGE-Lsum after each epoch,
                    (c) peak GPU memory while training,
                    (d) generation time per report.
                  Every number comes from a file; nothing is typed in by hand.

Usage
    python plot_results.py --runs_dir runs --out_dir figures

References
    Colours: the first categorical slots of the dataviz skill palette (blue, orange, aqua) in
    fixed order, plus grey for the untuned baseline. Bars carry their values and the lines are
    labelled directly, so colour is never the only cue.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# Run folders written by slurm/submit_final.sh: label -> (folder, colour).
RUNS = {"Zero-shot": ("zeroshot", "#8a8984"), "LoRA r=8": ("lora_r8_lr3e-3", "#2a78d6"),
        "LoRA r=16": ("lora_r16_lr3e-3", "#eb6834"), "Full FT": ("full_lr3e-4", "#1baf7a")}
TIERS = (("full", "Full validation\n(10,000)"), ("unseen_report", "Report not in train\n(5,494)"),
         ("clean", "Clean: new report and\nstudy (4,037)"))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_metrics(runs_dir: str) -> Dict[str, dict]:
    """metrics.json of each run, keyed by the label used in the figures."""
    out = {}
    for label, (folder, _) in RUNS.items():
        with open(os.path.join(runs_dir, folder, "metrics.json")) as handle:
            out[label] = json.load(handle)
    return out


def style(ax, ylabel: str, title: str) -> None:
    """Light, recessive axes: no top/right spines, faint horizontal grid."""
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def bar_labels(ax, bars: Sequence, fmt: str = "{:.1f}") -> None:
    """Write the value on top of every bar."""
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(bar.get_height()),
                ha="center", va="bottom", fontsize=8, color=INK)


def plot_results(metrics: Dict[str, dict], path: str) -> None:
    """The four-panel results overview."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    (ax_rouge, ax_dev), (ax_vram, ax_time) = axes

    width = 0.2
    for k, (label, (_, colour)) in enumerate(RUNS.items()):
        values = [metrics[label]["validation"]["rouge"][tier]["rougeLsum"] for tier, _ in TIERS]
        xs = [i + (k - 1.5) * width for i in range(len(TIERS))]
        bar_labels(ax_rouge, ax_rouge.bar(xs, values, width * 0.92, color=colour, label=label))
    ax_rouge.set_xticks(range(len(TIERS)), [name for _, name in TIERS])
    ax_rouge.set_ylim(0, 85)
    ax_rouge.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    style(ax_rouge, "ROUGE-Lsum (x100)", "(a) ROUGE-Lsum by validation tier")

    for label, (_, colour) in list(RUNS.items())[1:]:
        history = metrics[label]["training"]["history"]
        scores = [r["rougeLsum"] for r in history["dev_rouge"]]
        ax_dev.plot(history["dev_epoch"], scores, marker="o", color=colour, linewidth=2)
        shift = {"LoRA r=8": -11, "LoRA r=16": 3}.get(label, -3)    # keep close lines readable
        ax_dev.annotate(f"{label}  {scores[-1]:.1f}", (history["dev_epoch"][-1], scores[-1]),
                        xytext=(6, shift), textcoords="offset points", fontsize=8, color=INK)
    ax_dev.set_xticks([0, 1, 2, 3])
    ax_dev.set_xlim(-0.1, 4.2)
    ax_dev.set_xlabel("epoch (0 = before training)", color=MUTED, fontsize=9)
    style(ax_dev, "dev ROUGE-Lsum (x100)", "(b) Dev set during training")

    tuned = list(RUNS.items())[1:]
    vram = [metrics[label]["training"]["peak_vram_gib"] for label, _ in tuned]
    bar_labels(ax_vram, ax_vram.bar([label for label, _ in tuned], vram, 0.55,
                                    color=[c for _, (_, c) in tuned]), "{:.1f}")
    ax_vram.axhline(40, color=MUTED, linewidth=1, linestyle=":")
    ax_vram.text(2.45, 40.6, "A100 40 GB", ha="right", fontsize=8, color=MUTED)
    ax_vram.set_ylim(0, 46)
    style(ax_vram, "GiB", "(c) Peak GPU memory, training (micro-batch 8)")

    seconds = [metrics[label]["validation"]["inference"]["seconds_per_report"] for label in RUNS]
    bar_labels(ax_time, ax_time.bar(list(RUNS), seconds, 0.55, color=[c for _, c in RUNS.values()]),
               "{:.3f}")
    ax_time.set_ylim(0, 0.45)
    style(ax_time, "seconds per report", "(d) Generation time (beam 4, batch 32)")

    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def box(ax, x: float, y: float, w: float, h: float, text: str, face: str = "#ffffff",
        edge: str = "#52514e", size: int = 9, bold: bool = False) -> None:
    """A rounded box with centred text."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=face, edgecolor=edge, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    """An arrow between two points."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.4))


def fan(ax, x_from: float, y_from: float, xs: Sequence[float], y_line: float, y_to: float
        ) -> None:
    """One arrow from (x_from, y_from) down to a horizontal bar that ticks onto every x in xs."""
    ax.plot([x_from, x_from], [y_from, y_line], color=MUTED, linewidth=1.4)
    ax.plot([min(xs), max(xs)], [y_line, y_line], color=MUTED, linewidth=1.4)
    for x in xs:
        arrow(ax, x, y_line, x, y_to)


def plot_pipeline(path: str) -> None:
    """The pipeline diagram (static)."""
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    top = 4.15
    box(ax, 0.1, top, 2.3, 1.1, "Radiology report\n(BioLaySumm 2025\nLaymanRRG, 150k pairs)")
    box(ax, 3.0, top, 2.5, 1.1, "Task prompt + report\ntruncate to 512 tokens\nlabels: pad = -100")
    box(ax, 6.1, top, 3.1, 1.1, "FLAN-T5-base (247.6M)\nencoder  ->  decoder\nbeam search, 4 beams",
        face="#eaf2fd", edge="#2a78d6", bold=True)
    box(ax, 9.8, top, 2.1, 1.1, "Lay summary\n(plain language)")
    for x0, x1 in ((2.4, 3.0), (5.5, 6.1), (9.2, 9.8)):
        arrow(ax, x0 + 0.02, top + 0.55, x1 - 0.02, top + 0.55)

    ax.text(0.1, 3.5, "Three configurations, same split, prompt and decoding:", fontsize=10,
            color=INK, fontweight="bold")
    mid = 2.2
    box(ax, 0.1, mid, 3.6, 1.0, "Zero-shot baseline\nall weights frozen\n0 trainable parameters",
        face="#f1f0ed")
    box(ax, 4.2, mid, 3.6, 1.0,
        "LoRA  (peft, q and v)\nr = 8 or 16, alpha = 2r\n0.36% / 0.72% trainable",
        face="#eaf2fd", edge="#2a78d6")
    box(ax, 8.3, mid, 3.6, 1.0, "Full fine-tuning\nall weights updated\n100% trainable",
        face="#e7f6ef", edge="#1baf7a")
    fan(ax, 7.65, top - 0.05, (1.9, 6.0, 10.1), 3.35, mid + 1.02)

    ax.text(0.1, 1.65, "Evaluation:", fontsize=10, color=INK, fontweight="bold")
    box(ax, 0.1, 0.35, 3.6, 1.1,
        "ROUGE-1 / 2 / L / Lsum on validation\nfull, report-not-in-train,\nclean tiers")
    box(ax, 4.2, 0.35, 3.6, 1.1, "Resources\nparameters, peak VRAM,\ntokens/s, latency")
    box(ax, 8.3, 0.35, 3.6, 1.1,
        "Clinical safety autopsy\n5 cases read by hand:\nhallucination, omission, readability",
        face="#fdeee8", edge="#eb6834")
    for x in (1.9, 6.0, 10.1):                         # all three are evaluated alike
        ax.plot([x, x], [mid - 0.03, 1.9], color=MUTED, linewidth=1.4)
    ax.plot([1.9, 10.1], [1.9, 1.9], color=MUTED, linewidth=1.4)
    arrow(ax, 6.0, 1.9, 6.0, 1.48)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> None:
    """Draw both figures."""
    parser = argparse.ArgumentParser(description="Draw the README figures")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    plot_pipeline(os.path.join(args.out_dir, "pipeline.png"))
    plot_results(load_metrics(args.runs_dir), os.path.join(args.out_dir, "results.png"))
    print(f"wrote pipeline.png and results.png to {args.out_dir}")


if __name__ == "__main__":
    main()
