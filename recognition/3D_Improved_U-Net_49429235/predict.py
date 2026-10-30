import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import CLASS_NAMES, HipMRI3DDataset, load_splits
from modules import build_model
from utils import dice_iou_per_class, slice_thirds_dice, third_slice_indices

PROSTATE = CLASS_NAMES.index("prostate")
THIRD_NAMES = ["third 1 (one end)", "third 2 (middle)", "third 3 (other end)"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=False, help="unused if --split-file holds full paths")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split-file", required=True, help="splits.json written by train.py")
    p.add_argument("--out-dir", default="predictions")
    p.add_argument("--n-worst", type=int, default=4)
    p.add_argument("--cases", nargs="*", default=None,
                   help="draw these volumes instead of the worst ones, e.g. J026_Week1 T014_Week2 "
                        "(use the same list for both models to compare identical cases)")
    p.add_argument("--slice-axis", type=int, default=2)
    return p.parse_args()


def take_slice(vol, idx, axis):
    return torch.select(vol, axis, idx).cpu().numpy()

def overlay_error(img2d, gt2d, pred2d):
    lo, hi = img2d.min(), img2d.max()
    grey = (img2d - lo) / (hi - lo + 1e-8)
    rgb = np.stack([grey] * 3, axis=-1)
    fp = (pred2d == PROSTATE) & (gt2d != PROSTATE)
    fn = (gt2d == PROSTATE) & (pred2d != PROSTATE)
    rgb[fp] = [1.0, 0.0, 0.0]
    rgb[fn] = [0.0, 0.3, 1.0]
    return rgb


def save_case_figure(img, gt, pred, title, path, axis):
    idxs = third_slice_indices(gt, PROSTATE, axis)
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for r, (idx, name) in enumerate(zip(idxs, THIRD_NAMES)):
        im = take_slice(img[0], idx, axis)
        g = take_slice(gt, idx, axis)
        p = take_slice(pred, idx, axis)
        axes[r, 0].imshow(im, cmap="gray")
        axes[r, 1].imshow(g, cmap="nipy_spectral", vmin=0, vmax=len(CLASS_NAMES) - 1)
        axes[r, 2].imshow(p, cmap="nipy_spectral", vmin=0, vmax=len(CLASS_NAMES) - 1)
        axes[r, 3].imshow(overlay_error(im, g, p))
        axes[r, 0].set_ylabel(f"{name}\nslice {idx}")
    for c, t in enumerate(["Input", "Ground truth", "Prediction", "Error (red=FP, blue=FN)"]):
        axes[0, c].set_title(t)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    a = ckpt["args"]
    num_classes = len(CLASS_NAMES)
    model = build_model(a["model"], 1, num_classes, a["base_channels"], a["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {a['model']} model (epoch {ckpt['epoch']}, "
          f"val prostate Dice {ckpt['val_prostate_dice']:.4f})")

    splits = load_splits(args.split_file)
    ds = HipMRI3DDataset(splits["test"], tuple(a["target_size"]), num_classes, augment=False)

    rows = []
    for i in range(len(ds)):
        img, lab = ds[i]
        pred = model(img[None].to(device)).argmax(1)[0].cpu()
        d, _ = dice_iou_per_class(pred[None], lab[None], num_classes)
        thirds = slice_thirds_dice(pred, lab, PROSTATE, args.slice_axis)
        rows.append({"i": i, "case": splits["test"][i]["patient"] + " " + splits["test"][i]["week"],
                     "prostate_dice": d[0, PROSTATE].item(), "thirds": thirds,
                     "img": img, "lab": lab, "pred": pred})
        print(f"{rows[-1]['case']:<22} prostate Dice {rows[-1]['prostate_dice']:.4f} | "
              f"thirds {[round(t, 3) for t in thirds]}")

    valid = [r for r in rows if r["prostate_dice"] == r["prostate_dice"]]
    print(f"\nMean test prostate Dice over {len(valid)} volumes: "
          f"{np.mean([r['prostate_dice'] for r in valid]):.4f}")

    if args.cases:
        wanted = {c.replace("_", " ") for c in args.cases}
        worst = [r for r in rows if r["case"].replace("_", " ") in wanted]
        missing = wanted - {r["case"].replace("_", " ") for r in worst}
        if missing:
            print("WARNING: not found in the test split:", sorted(missing))
    else:
        worst = sorted(valid, key=lambda r: r["prostate_dice"])[:args.n_worst]
    for r in worst:
        fname = f"worst_{r['case'].replace(' ', '_')}.png"
        save_case_figure(r["img"], r["lab"], r["pred"],
                         f"{r['case']} - prostate Dice {r['prostate_dice']:.3f}",
                         os.path.join(args.out_dir, fname), args.slice_axis)
        print("saved", os.path.join(args.out_dir, fname))


if __name__ == "__main__":
    main()