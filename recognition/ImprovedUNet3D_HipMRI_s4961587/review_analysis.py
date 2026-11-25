"""
review_analysis.py: extra analyses requested in the PR #74 code review.

Author: Angus Slater (49615878), COMP3710 2026

Reads the test JSONs written by predict.py and prints:
  1. organ-interface errors per model: prostate leakage into the bladder and rectum,
     prostate lost to them, and signed apex/base extent errors;
  2. test results for all three training seeds of each model;
  3. per-patient (rather than per-scan) prostate Dice, since test patients contribute
     1-8 correlated scans each;
  4. how often the review flags proposed in README Section 8.3 fire: predicted prostate
     volume outside the training range, and more than one connected component.

Usage (on Rangpur, after predict.py has been run for every checkpoint):
    python review_analysis.py | tee results/review_analysis.txt
"""
import json
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from dataset import ROOT, TARGET_SHAPE, list_pairs, split_by_patient

MAIN = {"2D U-Net": "unet2d_v1", "3D U-Net": "unet3d_baseline_v2", "Improved 3D": "improved_v1"}
SEEDS = {"2D U-Net": ["unet2d_v1", "unet2d_v1_s1", "unet2d_v1_s2"],
         "3D U-Net": ["unet3d_baseline_v2", "unet3d_baseline_s1", "unet3d_baseline_s2"],
         "Improved 3D": ["improved_v1", "improved_v1_s1", "improved_v1_s2"]}
AUTOPSY = ["K008_Week0", "K018_Week7", "T014_Week2", "T014_Week7", "V027_Week7"]
PROSTATE = 5


def load(run):
    """Per-volume rows from results/test_<run>.json."""
    with open(f"results/test_{run}.json") as f:
        return json.load(f)["per_volume"]


def mean_std(x):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):6.2f} ± {np.nanstd(x):5.2f}"


main = {name: load(run) for name, run in MAIN.items()}

print("1. ORGAN-INTERFACE ERRORS (test set, mean ± std over 35 scans; median in brackets)")
for k in main["Improved 3D"][0]["interface"]:
    print(f"  {k}")
    for name, rows in main.items():
        v = np.array([r["interface"][k] for r in rows], float)
        extra = f"  |mean abs| {np.nanmean(np.abs(v)):5.2f}" if "extent" in k else ""
        print(f"    {name:12s} {mean_std(v)}  (median {np.nanmedian(v):5.2f}){extra}")

print("\n2. TEST RESULTS FOR ALL THREE SEEDS (mean over 35 scans per seed)")
for name, runs in SEEDS.items():
    per_seed = []
    for run in runs:
        rows = load(run)
        per_seed.append([np.mean([r["prostate_dice"] for r in rows]),
                         *[np.mean([r["prostate_dice_thirds"][t] for r in rows])
                           for t in ("apex", "mid", "base")],
                         np.nanmean([r["prostate_hd95_mm"] for r in rows])])
        print(f"  {run:22s} Dice {per_seed[-1][0]:.3f} | apex {per_seed[-1][1]:.3f} "
              f"mid {per_seed[-1][2]:.3f} base {per_seed[-1][3]:.3f} | HD95 {per_seed[-1][4]:.2f} mm")
    a = np.array(per_seed)
    print(f"  {name + ' (3 seeds)':22s} Dice {a[:,0].mean():.3f} ± {a[:,0].std():.3f} | "
          f"apex {a[:,1].mean():.3f} ± {a[:,1].std():.3f} mid {a[:,2].mean():.3f} ± {a[:,2].std():.3f} "
          f"base {a[:,3].mean():.3f} ± {a[:,3].std():.3f} | HD95 {a[:,4].mean():.2f} ± {a[:,4].std():.2f} mm\n")

print("3. PER-PATIENT PROSTATE DICE (each patient's scans averaged first)")
for name, rows in main.items():
    by_pat = defaultdict(list)
    for r in rows:
        by_pat[r["case"].split("_")[0]].append(r)
    pats = sorted(by_pat)
    dice = [np.mean([r["prostate_dice"] for r in by_pat[p]]) for p in pats]
    apex = [np.nanmean([r["prostate_dice_thirds"]["apex"] for r in by_pat[p]]) for p in pats]
    base = [np.nanmean([r["prostate_dice_thirds"]["base"] for r in by_pat[p]]) for p in pats]
    print(f"  {name:12s} {len(pats)} patients | Dice {np.mean(dice):.3f} ± {np.std(dice):.3f} | "
          f"apex {np.mean(apex):.3f} | base {np.mean(base):.3f}")
    print("    " + ", ".join(f"{p} ({len(by_pat[p])} scans) {d:.3f}" for p, d in zip(pats, dice)))

print("\n4. REVIEW FLAGS (README Section 8.3)")
train = split_by_patient(list_pairs(ROOT), seed=42)["train"]
vols = []
for _, lab_path in train:
    lab = nib.load(lab_path).get_fdata(dtype=np.float32)[:, :, :128]
    lab = F.interpolate(torch.from_numpy(lab)[None, None], size=TARGET_SHAPE, mode="nearest")
    vols.append(int((lab[0, 0].round() == PROSTATE).sum()))
lo, hi = min(vols), max(vols)
print(f"  training prostate volume range: {lo}-{hi} voxels (downsampled grid, {len(vols)} scans)")
for name, rows in main.items():
    flagged = []
    for r in rows:
        why = []
        if not lo <= r["prostate_pred_voxels"] <= hi:
            why.append(f"volume {r['prostate_pred_voxels']}")
        if r.get("prostate_components", 1) > 1:
            why.append(f"{r['prostate_components']} components")
        if why:
            flagged.append((r["case"], r["prostate_dice"], ", ".join(why)))
    fd = [d for _, d, _ in flagged]
    rest = [r["prostate_dice"] for r in rows if r["case"] not in {c for c, _, _ in flagged}]
    print(f"  {name:12s} flagged {len(flagged)}/{len(rows)} scans | mean Dice flagged "
          f"{np.mean(fd) if fd else float('nan'):.3f} vs not flagged {np.mean(rest):.3f}")
    for c, d, why in flagged:
        print(f"      {c:12s} Dice {d:.3f}  ({why})")
    caught = [c for c in AUTOPSY if c in {f[0] for f in flagged}]
    print(f"    autopsy cases caught: {caught if caught else 'none'}")
