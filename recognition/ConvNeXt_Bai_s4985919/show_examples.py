"""Example inputs and outputs of the final model, for the README.

Picks one AD scan and one CN scan from the VALIDATION subjects (official
split, fixed random seed, no test data), takes the middle slice of each, and
shows the three input channels the model sees with `context3` (previous,
current and next slice of the same scan, after resizing and normalisation)
together with the ensemble's AD probability. No scan or subject IDs are
written to the figure.

Run from the project folder on Rangpur (CPU is enough):
    python show_examples.py
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch

from dataset import load_datasets, split_slice_name
from modules import ConvNeXt

DEFAULT_CHECKPOINTS = [
    Path(f"checkpoints/convnext/official_cosine_ema_context3_s{seed}/best.pth")
    for seed in (42, 43, 44)
]
CLASS_NAMES = {0: "AD", 1: "CN"}


def pick_examples(val_dataset, seed):
    """Return one dataset index per class: the middle slice of a random scan."""
    source, indices = val_dataset.dataset, val_dataset.indices
    scans = defaultdict(list)  # (label, scan id) -> [(slice number, index)]
    for index in indices:
        path, label = source.samples[index]
        scan_id, number = split_slice_name(path)
        scans[(label, scan_id)].append((number, index))
    rng = random.Random(seed)
    picks = {}
    for label in (0, 1):
        keys = sorted(k for k in scans if k[0] == label)
        slices = sorted(scans[rng.choice(keys)])
        picks[label] = slices[len(slices) // 2][1]
    return picks


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("figures/example_inputs.png"))
    args = parser.parse_args()

    _, val_dataset, _ = load_datasets("official", "context3")
    models = []
    for checkpoint in args.checkpoint:
        model = ConvNeXt(num_classes=2, in_channels=3)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        models.append(model.eval())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = pick_examples(val_dataset, args.seed)
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.4))
    for row, (label, index) in enumerate(sorted(picks.items())):
        image, _ = val_dataset.dataset[index]  # 3x224x224, normalised to [-1, 1]
        with torch.no_grad():
            prob_ad = torch.stack([torch.softmax(m(image.unsqueeze(0)), dim=1)[0, 0]
                                   for m in models]).mean().item()
        predicted = "AD" if prob_ad >= args.threshold else "CN"
        print(f"Example {row + 1}: true {CLASS_NAMES[label]}, P(AD)={prob_ad:.3f}, "
              f"predicted {predicted}")
        for col, name in enumerate(("previous slice", "current slice", "next slice")):
            ax = axes[row][col]
            ax.imshow((image[col] * 0.5 + 0.5).numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(name, fontsize=8)
            ax.axis("off")
        axes[row][0].text(-0.08, 0.5, f"true {CLASS_NAMES[label]}\nP(AD)={prob_ad:.2f}\n"
                          f"predicted {predicted}", transform=axes[row][0].transAxes,
                          ha="right", va="center", fontsize=9)
    fig.suptitle("Input channels (context3) and ensemble output, validation subjects",
                 fontsize=10)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130, bbox_inches="tight")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
