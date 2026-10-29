"""Count slice-level failure modes over the whole test set for all three models.

dropout slice : true prostate present but the model predicts < 10% of its area
over slice    : predicted prostate area > 2x the true area
Reported for the base, middle and apex thirds, plus the number of volumes with
a run of >= 2 consecutive dropout slices.
"""
import numpy as np
import torch

from dataset import PROSTATE, load_case, load_split
from predict import load_model, predict_volume, prostate_thirds

# Load the three trained models (first runs) and the test cases.
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split = load_split()
cases = sorted(c for c, s in split.items() if s == "test")
names = ("unet2d", "improved2d", "improved3d")
models = {m: load_model(m, dev) for m in names}
stats = {m: {p: [0, 0, 0] for p in ("base", "middle", "apex")} for m in names}  # slices, dropout, over
runs = {m: 0 for m in names}

# For every test volume, compare each model's predicted prostate area with the true area, slice by slice.
for case in cases:
    img, lab = load_case(case)
    gt = lab == PROSTATE
    thirds, zs = prostate_thirds(lab)
    true_px = gt.sum(axis=(1, 2))      # true prostate pixels on each slice
    for m in names:
        pr = predict_volume(models[m], m, img, dev) == PROSTATE
        pred_px = pr.sum(axis=(1, 2))
        # dropout: the true prostate is there but under 10% of it is predicted; over: more than 2x the true area
        drop = (true_px > 0) & (pred_px < 0.1 * true_px)
        over = (true_px > 0) & (pred_px > 2 * true_px)
        for part, idx in thirds.items():
            s = stats[m][part]
            s[0] += len(idx)
            s[1] += int(drop[idx].sum())
            s[2] += int(over[idx].sum())
        # a volume counts once if two neighbouring slices both drop out
        d = drop[zs].astype(int)
        if np.any(d[:-1] & d[1:]):
            runs[m] += 1

# Report the counts per prostate third (README Section 7.2).
print(f"{len(cases)} test volumes")
for m in names:
    print(f"\n{m}: volumes with a run of >=2 dropout slices: {runs[m]}")
    for part, (n, d, o) in stats[m].items():
        print(f"  {part:7s} slices {n:4d} | dropout {d:3d} ({100 * d / n:4.1f}%) | over-segmented {o:3d} ({100 * o / n:4.1f}%)")
