"""
Run a trained model on test scans, print per-class Dice and save comparison figures.

Example:
    python predict.py --run-dir /content/drive/MyDrive/comp3710_runs/unet2d \
        --cache-dir /content/cache --num-cases 3

Each figure shows the axial slice through the centre of the true prostate:
input image | ground truth | prediction | errors (false positive red, false negative blue).
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import LABEL_NAMES, PROSTATE, load_case
from modules import build_model
from utils import dice_iou_per_class, predict_volume, save_json

CLASS_COLOURS = ["#00000000", "#d9a066", "#4caf50", "#f06292", "#7e57c2", "#ffd600"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="output folder of train.py (has best.pt, split.json)")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--num-cases", type=int, default=3, help="how many test scans to show")
    p.add_argument("--case", default=None, help="a specific case id, e.g. 004_w2")
    return p.parse_args()


def label_rgb(label_slice):
    """Colour an integer label slice (H, W) for plotting."""
    from matplotlib.colors import to_rgba
    lut = np.array([to_rgba(c) for c in CLASS_COLOURS])
    return lut[label_slice]


def save_figure(image, label, pred, case_id, dice, path):
    """Four panels at the axial slice through the centre of the true prostate."""
    zs = np.where((label == PROSTATE).any(axis=(1, 2)))[0]
    z = int(zs.mean()) if len(zs) else image.shape[0] // 2
    # transpose + origin="upper": front of the body (bladder) at the top, as radiologists view axial MR
    img, gt, pr = image[z].T, label[z].T, pred[z].T

    fp = (pr == PROSTATE) & (gt != PROSTATE)
    fn = (gt == PROSTATE) & (pr != PROSTATE)
    errors = np.zeros(img.shape + (4,))
    errors[fp] = [1, 0, 0, 0.8]
    errors[fn] = [0, 0.4, 1, 0.8]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    panels = [("input MR", None), ("ground truth", label_rgb(gt)),
              ("prediction", label_rgb(pr)), ("prostate errors: FP red, FN blue", errors)]
    for ax, (title, overlay) in zip(axes, panels):
        ax.imshow(img, cmap="gray", origin="upper")
        if overlay is not None:
            ax.imshow(overlay, origin="upper", alpha=0.6 if title != panels[-1][0] else 1.0)
        if title.startswith("prostate errors"):
            ax.contour(gt == PROSTATE, levels=[0.5], colors="yellow", linewidths=0.8, origin="upper")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"case {case_id}, slice {z}  |  prostate Dice {dice[PROSTATE]:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(args.run_dir, "best.pt"), map_location=device, weights_only=False)
    model, mode = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    with open(os.path.join(args.run_dir, "split.json")) as f:
        split = json.load(f)
    case_ids = [args.case] if args.case else split["test_cases"][:args.num_cases]

    out_dir = os.path.join(args.run_dir, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    print(f"Model {ckpt['config']['model']} (best epoch {ckpt['epoch']})")
    print("case      " + "".join(f"{n:>11}" for n in LABEL_NAMES))
    for cid in case_ids:
        image, label = load_case(args.cache_dir, cid)
        pred = predict_volume(model, image, mode, device, scale=ckpt["config"].get("scale", 2))
        dice, iou = dice_iou_per_class(pred, label)
        results[cid] = {"dice": dice, "iou": iou}
        print(f"{cid:<10}" + "".join(f"{d:>11.3f}" for d in dice))
        save_figure(image.numpy(), label.numpy(), pred.numpy(), cid, dice,
                    os.path.join(out_dir, f"{cid}.png"))
    save_json(results, os.path.join(out_dir, "predict_metrics.json"))
    print(f"\nFigures saved to {out_dir}")


if __name__ == "__main__":
    main()
