"""Plot aggregate seed-42 learning curves; no MRI or patient records are used."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("training_curves.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    with args.data.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    models = ("CNN-A", "CNN-B", "ConvNeXt-F")
    colours = {"train": "#2463A6", "validation": "#C66B20"}
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 4.7), sharex="col")
    for col, model in enumerate(models):
        ax = axes[0, col]
        metric_ax = axes[1, col]
        selected = sorted((r for r in rows if r["model"] == model), key=lambda r: int(r["epoch"]))
        if len(selected) != 30 or [int(r["epoch"]) for r in selected] != list(range(1, 31)):
            raise ValueError(f"Expected 30 consecutive epochs for {model}")
        epochs = [int(r["epoch"]) for r in selected]
        for key, label in (("train_unweighted_loss", "Train"), ("validation_loss", "Validation")):
            ax.plot(epochs, [float(r[key]) for r in selected], label=label,
                    color=colours["train" if label == "Train" else "validation"], linewidth=1.9)
        ax.set_title(model, fontsize=11, weight="bold")
        ax.set_xlim(1, 30)
        ax.set_xticks((1, 10, 20, 30))
        ax.grid(alpha=0.18, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        metric_ax.plot(epochs, [float(r["validation_patient_macro_f1"]) for r in selected],
                       color="#4B8268", linewidth=1.9)
        metric_ax.set_ylim(0, 1)
        metric_ax.set_yticks((0, 0.5, 1))
        metric_ax.set_xlabel("Epoch")
        metric_ax.grid(alpha=0.18, linewidth=0.7)
        metric_ax.spines[["top", "right"]].set_visible(False)
        metric_ax.tick_params(labelsize=8)
    axes[0, 0].set_ylabel("Slice cross-entropy", fontsize=9)
    axes[1, 0].set_ylabel("Validation patient macro-F1", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Training trajectory · seed 42", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.0, h_pad=1.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "training_curves.png", dpi=240, bbox_inches="tight")
    svg = args.output_dir / "training_curves.svg"
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
