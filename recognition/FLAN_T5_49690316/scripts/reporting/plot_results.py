"""Render report figures from downloaded results and transcribed training logs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    with path.open() as file:
        return json.load(file)


def main():
    training = read_json(ROOT / "report_data" / "training_measurements.json")
    lora = read_json(ROOT / "results" / "lora_all_final_eval" / "metrics.json")
    full = read_json(ROOT / "results" / "full_all_final_eval" / "metrics.json")
    assert lora["baseline"] == full["baseline"], "Baseline results differ"
    profiles = [
        read_json(ROOT / "results" / "inference_profiles" / f"{name}.json")
        for name in ("baseline", "lora_all", "full_all")
    ]
    assert len({row["reports_sha256"] for row in profiles}) == 1

    colors = ["#778899", "#2171b5", "#d46a24"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    positions = np.arange(4)
    for offset, (label, values, color) in enumerate(zip(
        ["Untuned", "LoRA", "Full"],
        [lora["baseline"], lora["trained"], full["trained"]], colors
    )):
        axes[0, 0].bar(positions + (offset - 1) * .24,
                       [values[key] for key in metrics], width=.24,
                       color=color, label=label)
    axes[0, 0].set_xticks(positions, ["R-1", "R-2", "R-L", "R-Lsum"])
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Mean per-example F1")
    axes[0, 0].set_title("Final reference overlap (10,000 reports)")
    axes[0, 0].legend(frameon=False)

    for mode, label, color in [("lora", "LoRA", colors[1]), ("full", "Full", colors[2])]:
        for split, style in [("training_loss", "--"), ("development_loss", "-")]:
            axes[0, 1].plot([1, 2, 3], training[mode][split], style,
                            marker="o", color=color,
                            label=f"{label} {'train' if split == 'training_loss' else 'dev'}")
    axes[0, 1].set_xticks([1, 2, 3])
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Token-weighted cross-entropy")
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].set_title("Full-data learning curves")
    axes[0, 1].legend(frameon=False, fontsize=9)

    peak_values = [max(training[mode]["peak_cuda_allocated_gib"]) for mode in ("lora", "full")]
    bars = axes[1, 0].bar(["LoRA", "Full"], peak_values, color=colors[1:])
    axes[1, 0].bar_label(bars, fmt="%.2f")
    axes[1, 0].set_ylim(0, 7.5)
    axes[1, 0].set_ylabel("Peak allocated CUDA memory (GiB)")
    axes[1, 0].set_title("Training + development peak (batch size 2)")

    speeds = [profile["generated_tokens_per_second"] for profile in profiles]
    bars = axes[1, 1].bar(["Untuned", "LoRA", "Full"], speeds, color=colors)
    axes[1, 1].bar_label(bars, fmt="%.1f")
    axes[1, 1].set_ylim(0, 110)
    axes[1, 1].set_ylabel("Generated tokens/second")
    axes[1, 1].set_title("Inference: 32 shared dev reports, batch size 1")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    output = ROOT / "figures" / "results_summary.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print("Saved", output)


if __name__ == "__main__":
    main()
