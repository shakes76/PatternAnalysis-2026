"""Find and plot the most confidently wrong test scans for a trained model.

Example:
  python failure_cases.py --ckpt ~/adni_work/gfnet_clean_r4_stem_s2.pth --conv-stem \
      --preds ~/adni_work/gfnet_clean_r4_stem_s2_test_preds.json --name final

Picks the AD scans given the lowest AD probability (missed AD) and the NC scans given the highest
AD probability (false alarms), then shows 8 evenly spaced slices of each with the per-slice
probability of AD. Saves failure_cases_<name>.png and prints a table for the README.
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import get_loaders
from modules import BaselineCNN, GFNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preds", required=True, help="test preds json from predict.py or train.py")
    ap.add_argument("--model", default="gfnet", choices=["baseline", "gfnet"])
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--conv-stem", action="store_true")
    ap.add_argument("--n-fn", type=int, default=3, help="missed AD scans to show")
    ap.add_argument("--n-fp", type=int, default=2, help="false alarm scans to show")
    ap.add_argument("--name", default="final")
    args = ap.parse_args()

    d = json.load(open(os.path.expanduser(args.preds)))
    p, y = d["scan_prob"], d["scan_label"]
    wrong = [s for s in p if (p[s] > 0.5) != bool(y[s])]
    fn = sorted([s for s in wrong if y[s] == 1], key=lambda s: p[s])
    fp = sorted([s for s in wrong if y[s] == 0], key=lambda s: -p[s])
    print(f"{len(p)} test scans, {len(wrong)} wrong: {len(fn)} missed AD, {len(fp)} false alarms")
    chosen = fn[:args.n_fn] + fp[:args.n_fp]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "gfnet":
        model = GFNet(dim=args.dim, depth=args.depth, patch=args.patch, conv_stem=args.conv_stem)
    else:
        model = BaselineCNN()
    model.load_state_dict(torch.load(os.path.expanduser(args.ckpt), map_location=device))
    model.to(device).eval()

    _, _, test_dl = get_loaders(64)
    imgs, probs = defaultdict(list), defaultdict(list)
    want = set(chosen)
    with torch.no_grad():
        for x, _, sid in test_dl:
            pr = model(x.to(device)).softmax(1)[:, 1].cpu().tolist()
            for i, s in enumerate(sid):
                if s in want:
                    imgs[s].append(x[i, 0].numpy())
                    probs[s].append(pr[i])

    cols = 8
    fig, axes = plt.subplots(len(chosen), cols, figsize=(2 * cols, 2.2 * len(chosen)), squeeze=False)
    print("\n| scan | true | predicted | scan P(AD) | slices correct |\n|---|---|---|---|---|")
    for r, s in enumerate(chosen):
        sl, pr = imgs[s], np.array(probs[s])
        true = "AD" if y[s] == 1 else "NC"
        pred = "AD" if p[s] > 0.5 else "NC"
        frac = float(((pr > 0.5) == bool(y[s])).mean())
        print(f"| {s} | {true} | {pred} | {p[s]:.2f} | {frac:.0%} of {len(pr)} |")
        pick = np.linspace(0, len(sl) - 1, cols).astype(int)
        for c, k in enumerate(pick):
            ax = axes[r][c]
            ax.imshow(sl[k], cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"P(AD)={pr[k]:.2f}", fontsize=8)
        axes[r][0].set_ylabel(f"true {true}\npred {pred}\nscan {p[s]:.2f}", fontsize=9)
    fig.tight_layout()
    out = f"failure_cases_{args.name}.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
