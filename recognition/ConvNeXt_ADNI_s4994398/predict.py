"""
Example usage of a trained model: load saved weights, run inference, print results
and save a visualisation.

    # evaluate on the whole ADNI test split and plot random examples
    python predict.py --ckpt checkpoints/convnext_best.pt

    # classify specific slices
    python predict.py --ckpt checkpoints/convnext_best.pt --images D:/ADNI/AD_NC/test/AD/1003730_80.jpeg

Predictions whose (temperature-scaled) confidence falls below ``--threshold`` are
reported as "REFER" - i.e. routed to a human reader instead of being automated
(see the README for how the threshold was chosen).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import CLASSES, DEFAULT_ROOT, TensorBatcher, load_split, preprocess
from modules import build_model
from utils import classification_report, softmax_np, subject_level

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["arch"], ckpt.get("variant", "tiny"), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def infer(model, images_u8, mean, std, device, batch_size=128):
    """uint8 [N,H,W] tensor -> logits [N,2] numpy."""
    out = []
    dummy = torch.zeros(len(images_u8), dtype=torch.long)
    for xb, _, _ in TensorBatcher(images_u8, dummy, batch_size, False, device):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(model(preprocess(xb, mean, std)).float().cpu())
    return torch.cat(out).numpy()


def describe(prob, threshold):
    conf = prob.max()
    label = CLASSES[prob.argmax()] if conf >= threshold else "REFER"
    return label, conf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "convnext_best.pt"))
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--images", nargs="*", help="specific slice images to classify")
    ap.add_argument("--temperature", type=float, default=None,
                    help="softmax temperature (default: value stored by evaluate.py, else 1)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="refer cases below this confidence (default: value stored by evaluate.py, else 0.5)")
    ap.add_argument("--n-show", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "predict_examples.png"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.ckpt, device)
    T = args.temperature or ckpt.get("temperature", 1.0)
    thr = args.threshold or ckpt.get("threshold", 0.5)
    print(f"Loaded {ckpt['arch']} from epoch {ckpt['epoch']} (val acc {ckpt['val_acc']:.4f}); T={T:.3f}, "
          f"refer below confidence {thr:.3f}")

    if args.images:
        imgs = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert("L")) for p in args.images]))
        probs = softmax_np(infer(model, imgs, ckpt["mean"], ckpt["std"], device), T)
        for p, pr in zip(args.images, probs):
            label, conf = describe(pr, thr)
            print(f"{os.path.basename(p):>22s}: P(AD)={pr[1]:.3f} -> {label} (confidence {conf:.3f})")
        labels = None
    else:
        imgs, labels, paths = load_split("test", args.root)
        probs = softmax_np(infer(model, imgs, ckpt["mean"], ckpt["std"], device), T)
        labels = labels.numpy()
        rep = classification_report(probs, labels)
        print(f"\nTest slices: {len(labels)}")
        for k in ("accuracy", "macro_f1", "auroc", "sensitivity", "specificity", "ece"):
            print(f"  {k:12s} {rep[k]:.4f}")
        for c in CLASSES:
            print(f"  {c}: precision {rep[f'precision_{c}']:.4f} recall {rep[f'recall_{c}']:.4f} "
                  f"F1 {rep[f'f1_{c}']:.4f}")
        print(f"  confusion [[TN, FP], [FN, TP]] = {rep['confusion']}")
        sp, sl, _ = subject_level(probs, labels, paths)
        print(f"Subject-level accuracy (mean of 20 slices): {(sp.argmax(1) == sl).mean():.4f} "
              f"over {len(sl)} subjects")
        keep = probs.max(1) >= thr
        print(f"Reject option @ {thr:.3f}: automated {keep.mean():.1%} of slices with accuracy "
              f"{(probs.argmax(1)[keep] == labels[keep]).mean():.4f}; referred {(~keep).sum()}")

    # visualise a random selection
    rng = np.random.default_rng(0)
    idx = rng.choice(len(imgs), size=min(args.n_show, len(imgs)), replace=False)
    cols = 6
    rows = -(-len(idx) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, idx):
        label, conf = describe(probs[i], thr)
        ax.imshow(imgs[i].numpy(), cmap="gray")
        title = f"pred {label} ({conf:.2f})"
        colour = "black"
        if labels is not None:
            title += f"\ntrue {CLASSES[labels[i]]}"
            colour = "orange" if label == "REFER" else ("green" if label == CLASSES[labels[i]] else "red")
        ax.set_title(title, fontsize=9, color=colour)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Saved visualisation to {args.out}")


if __name__ == "__main__":
    main()
