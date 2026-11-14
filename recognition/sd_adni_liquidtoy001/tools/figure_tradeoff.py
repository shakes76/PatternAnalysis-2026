"""README figure: every model on two axes of the trilemma (local).

Left: coverage (higher covers more of the real data) against training
preference M2 (below zero means closer to training slices than held-out slices
would be), with 95% intervals. Right: membership-inference AUC above the SD
control, for the diffusion models. The real floor marks what real slices score.

    python tools/figure_tradeoff.py --metrics outputs/final-metrics-<date>/metrics.json \
        --output docs/figures/tradeoff-test.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {
    "sd": "SD zero-shot",
    "e0": "E0",
    "e2": "E2 full UNet",
    "r1e2": "E2 seed 2",
    "e3": "E3 adapter",
    "r1e3": "E3 seed 2",
    "b1": "B1 VAE",
    "b2": "B2 GAN",
    "b2b": "B2b GANs",
    "b3": "B3 scratch",
}

# Label offsets (points) for models that would otherwise overlap.
OFFSETS = {"r1e2": (6, -12), "b2": (6, -12), "r1e3": (6, -12), "e0": (-18, 6)}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

    for name, entry in metrics["models"].items():
        coverage = entry["inception"]["prdc"]["coverage"]
        preference = entry["m2_training_preference"]["pixel"]
        low, high = preference["difference_ci95"]
        left.errorbar(
            coverage["mean"],
            preference["difference"],
            xerr=coverage["sd"],
            yerr=[[preference["difference"] - low], [high - preference["difference"]]],
            fmt="o",
            capsize=3,
        )
        left.annotate(
            NAMES.get(name, name),
            (coverage["mean"], preference["difference"]),
            textcoords="offset points",
            xytext=OFFSETS.get(name, (6, 4)),
            fontsize=8,
        )
    floor = metrics["floor"]["inception"]["prdc"]["coverage"]["mean"]
    left.axvline(floor, color="grey", linestyle=":", label=f"real floor {floor:.2f}")
    left.axhline(0, color="grey", linewidth=0.8)
    left.set(
        xlabel="coverage (Inception features, k = 5)",
        ylabel="training preference M2 (log ratio; < 0 means memorisation)",
        title="Coverage against sample-level memorisation",
    )
    left.legend(fontsize=8, loc="lower right")

    membership = metrics.get("membership", {})
    models = [m for m in membership if m != "sd"]
    for i, name in enumerate(models):
        block = membership[name]
        low, high = block["auc_minus_control_ci95"]
        right.errorbar(
            i,
            block["auc_minus_control"],
            yerr=[
                [block["auc_minus_control"] - low],
                [high - block["auc_minus_control"]],
            ],
            fmt="s",
            capsize=4,
        )
    right.axhline(0, color="grey", linewidth=0.8)
    right.set_xticks(range(len(models)), [NAMES.get(m, m) for m in models], rotation=20)
    right.set(
        ylabel="membership AUC minus SD control",
        title="Does the model reveal its training patients?",
    )
    figure.text(
        0.01,
        0.01,
        f"Split: {'test' if metrics['test_split_read'] else 'validation (rehearsal)'}. "
        "Error bars: 95% bootstrap intervals (coverage: sd over 10 real draws).",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
