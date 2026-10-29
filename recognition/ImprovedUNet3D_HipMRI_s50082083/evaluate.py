"""
Investigate the 2D-versus-3D dilemma on the held-out test patients.

For every trained model and every test scan this script measures, at full resolution:
    - per-class Dice / IoU
    - prostate Dice in the apex, mid and base thirds of the gland
    - extent error (mm): how far the predicted gland starts/ends from the true apex/base
    - leakage (mL): predicted prostate inside the true rectum and bladder
    - HD95 (mm): 95th-percentile boundary distance of the prostate
    - anatomy checks that need no ground truth (could flag cases in deployment):
        A. "background" enclosed inside the body on an axial slice (impossible holes)
        B. predicted bladder missing or more than 10 mm from the predicted prostate
        C. predicted prostate split into several pieces
    - inference time and peak inference VRAM per scan

Examples:
    # 1. choose the anatomy-check thresholds on the VALIDATION patients (never on test)
    python evaluate.py --runs-dir RUNS --cache-dir CACHE --out-dir RUNS/eval_val --split val --calibrate
    # 2. evaluate on the TEST patients using those saved thresholds
    python evaluate.py --runs-dir RUNS --cache-dir CACHE --out-dir RUNS/evaluation \
        --thresholds RUNS/eval_val/thresholds.json

Writes per_case.csv, summary.json, summary.md (README-ready tables) and figures.
Metrics are averaged over each patient's scans first, then mean +/- std across patients.
"""

import argparse
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from dataset import LABEL_NAMES, PROSTATE, load_case, patient_of
from modules import build_model
from utils import dice_iou_per_class, predict_volume, save_json

BLADDER, RECTUM = 3, 4
GAP_LIMIT_MM = 10.0        # check B: bladder must lie within this distance of the prostate
HOLE_LIMIT_ML = 1.0        # check A default; replaced per model by validation calibration
CALIBRATION_PERCENTILE = 90  # check A threshold = 90th percentile of holes in ACCEPTABLE validation scans
PIECE_MIN_ML = 0.1         # check C: prostate pieces smaller than this are ignored
BAD_PROSTATE, BAD_BLADDER = 0.7, 0.5   # a scan is "clinically unacceptable" below these Dice

# Colour-blind-safe categorical slots (fixed order), plus text/grid ink
MODEL_STYLE = {"unet2d": ("2D U-Net", "#2a78d6"),
               "iunet2d": ("2D Improved U-Net", "#eb6834"),
               "iunet3d": ("3D Improved U-Net (x2)", "#1baf7a"),
               "iunet3d_full": ("3D Improved U-Net (full res)", "#eda100")}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", required=True, help="folder containing one sub-folder per trained model")
    p.add_argument("--models", nargs="+", default=["unet2d", "iunet2d", "iunet3d"])
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--figure-cases", nargs="+", default=["015_w6", "038_w0", "014_w1"],
                   help="scans to draw slice-by-slice Dice plots for")
    p.add_argument("--split", choices=["val", "test"], default="test", help="which patients to evaluate")
    p.add_argument("--calibrate", action="store_true",
                   help="(with --split val) choose check-A thresholds and save thresholds.json")
    p.add_argument("--thresholds", default=None, help="thresholds.json from a validation calibration run")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Geometry helpers (NumPy/SciPy; evaluation only, not part of the model)
# --------------------------------------------------------------------------- #
def gland_regions(label):
    """Split the true prostate's slice extent into apex / mid / base thirds.

    The base is the end nearer the bladder. Returns dict region -> slice indices,
    the apex and base end slices, and +1/-1 = direction (in slice index) towards the base.
    """
    zs = np.where((label == PROSTATE).any(axis=(1, 2)))[0]
    bladder_z = np.where((label == BLADDER).any(axis=(1, 2)))[0]
    to_base = 1 if (len(bladder_z) == 0 or bladder_z.mean() >= zs.mean()) else -1
    ordered = zs if to_base == 1 else zs[::-1]                    # apex ... base
    thirds = np.array_split(ordered, 3)
    return {"apex": thirds[0], "mid": thirds[1], "base": thirds[2]}, ordered[0], ordered[-1], to_base


def dice(a, b):
    s = a.sum() + b.sum()
    return np.nan if s == 0 else 2 * (a & b).sum() / s


def hd95(a, b, spacing):
    """95th-percentile symmetric surface distance (mm) between two binary masks."""
    if not a.any() or not b.any():
        return np.nan
    idx = np.argwhere(a | b)
    lo, hi = np.maximum(idx.min(0) - 2, 0), idx.max(0) + 3
    box = tuple(slice(l, h) for l, h in zip(lo, hi))
    a, b = a[box], b[box]
    sa, sb = a ^ ndimage.binary_erosion(a), b ^ ndimage.binary_erosion(b)
    da = ndimage.distance_transform_edt(~sa, sampling=spacing)
    db = ndimage.distance_transform_edt(~sb, sampling=spacing)
    return float(np.percentile(np.concatenate([da[sb], db[sa]]), 95))


def enclosed_background_ml(seg, voxel_ml):
    """Check A: background fully surrounded by body on an axial slice (a 'hole')."""
    holes = 0
    for z in range(seg.shape[0]):
        body = seg[z] != 0
        if body.any():
            holes += (ndimage.binary_fill_holes(body) & ~body).sum()
    return holes * voxel_ml


def bladder_gap_mm(seg, spacing):
    """Check B: shortest distance from the bladder to the prostate (inf if either is missing)."""
    if not (seg == BLADDER).any() or not (seg == PROSTATE).any():
        return float("inf")
    dist_to_prostate = ndimage.distance_transform_edt(seg != PROSTATE, sampling=spacing)
    return float(dist_to_prostate[seg == BLADDER].min())


def prostate_pieces(seg, voxel_ml):
    """Check C: number of prostate pieces at least PIECE_MIN_ML in volume."""
    lab, n = ndimage.label(seg == PROSTATE)
    if n == 0:
        return 0
    sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1)) * voxel_ml
    return int((sizes >= PIECE_MIN_ML).sum())


def anatomy_checks(seg, spacing, voxel_ml, hole_limit=HOLE_LIMIT_ML):
    holes = enclosed_background_ml(seg, voxel_ml)
    gap = bladder_gap_mm(seg, spacing)
    pieces = prostate_pieces(seg, voxel_ml)
    flags = {"A_holes": bool(holes > hole_limit), "B_bladder": bool(gap > GAP_LIMIT_MM), "C_pieces": pieces > 1}
    return {"holes_ml": float(holes), "bladder_gap_mm": gap, "prostate_pieces": pieces,
            **flags, "flagged": any(flags.values())}


def case_metrics(pred, label, spacing, hole_limit=HOLE_LIMIT_ML):
    """All per-scan metrics for one prediction (numpy (D, H, W) integer maps)."""
    voxel_ml = float(np.prod(spacing)) / 1000.0
    dice_all, iou_all = dice_iou_per_class(torch.from_numpy(pred), torch.from_numpy(label))
    gt, pr = label == PROSTATE, pred == PROSTATE
    regions, apex_z, base_z, to_base = gland_regions(label)

    m = {f"dice_{n}": d for n, d in zip(LABEL_NAMES, dice_all)}
    m.update({f"iou_{n}": v for n, v in zip(LABEL_NAMES, iou_all)})
    for name, zs in regions.items():
        m[f"prostate_dice_{name}"] = dice(gt[zs], pr[zs])

    pz = np.where(pr.any(axis=(1, 2)))[0]
    if len(pz):
        pred_apex = pz.min() if to_base == 1 else pz.max()
        pred_base = pz.max() if to_base == 1 else pz.min()
        # positive = predicted gland extends beyond the true end; negative = stops short
        m["base_extent_mm"] = float((pred_base - base_z) * to_base * spacing[0])
        m["apex_extent_mm"] = float((apex_z - pred_apex) * to_base * spacing[0])
    else:
        m["base_extent_mm"] = m["apex_extent_mm"] = np.nan

    m["leak_rectum_ml"] = float((pr & (label == RECTUM)).sum() * voxel_ml)
    m["leak_bladder_ml"] = float((pr & (label == BLADDER)).sum() * voxel_ml)
    m["prostate_hd95_mm"] = hd95(pr, gt, spacing)
    m["unacceptable"] = bool(dice_all[PROSTATE] < BAD_PROSTATE or dice_all[BLADDER] < BAD_BLADDER)
    m.update(anatomy_checks(pred, spacing, voxel_ml, hole_limit))

    g, p, inter = gt.sum((1, 2)), pr.sum((1, 2)), (gt & pr).sum((1, 2))
    slice_dice = np.where(g + p > 0, 2 * inter / np.maximum(g + p, 1), np.nan)
    return m, {"slice_dice": slice_dice, "true_area": g, "base_low": to_base == 1}


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def per_patient(rows, key):
    by_p = {}
    for r in rows:
        v = r[key]
        if v is not None and np.isfinite(v):
            by_p.setdefault(r["patient"], []).append(v)
    vals = np.array([np.mean(v) for v in by_p.values()])
    return (float(vals.mean()), float(vals.std())) if len(vals) else (np.nan, np.nan)


def fmt(ms, digits=3):
    m, s = ms
    return "–" if not np.isfinite(m) else f"{m:.{digits}f} ± {s:.{digits}f}"


def write_markdown(summary, models, path, split="test", hole_limits=None):
    name = lambda k: MODEL_STYLE.get(k, (k,))[0]
    lines = [f"## Evaluation summary ({split} patients; per-patient mean ± std)", ""]
    lines += ["### Prostate accuracy by region", "",
              "| Model | Overall Dice | Apex Dice | Mid Dice | Base Dice | HD95 (mm) |",
              "| --- | --- | --- | --- | --- | --- |"]
    for k in models:
        s = summary[k]
        lines.append(f"| {name(k)} | {fmt(s['dice_prostate'])} | {fmt(s['prostate_dice_apex'])} | "
                     f"{fmt(s['prostate_dice_mid'])} | {fmt(s['prostate_dice_base'])} | {fmt(s['prostate_hd95_mm'], 1)} |")
    lines += ["", "### Extent error and leakage", "",
              "Extent error: + = predicted gland extends past the true end, − = stops short.", "",
              "| Model | Apex extent (mm) | Base extent (mm) | |Apex| (mm) | |Base| (mm) | Leak into rectum (mL) | Leak into bladder (mL) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for k in models:
        s = summary[k]
        lines.append(f"| {name(k)} | {fmt(s['apex_extent_mm'], 1)} | {fmt(s['base_extent_mm'], 1)} | "
                     f"{fmt(s['abs_apex_extent_mm'], 1)} | {fmt(s['abs_base_extent_mm'], 1)} | "
                     f"{fmt(s['leak_rectum_ml'], 2)} | {fmt(s['leak_bladder_ml'], 2)} |")
    lines += ["", "### Other organs (Dice)", "", "| Model | Bladder | Rectum | Bone | Body |", "| --- | --- | --- | --- | --- |"]
    for k in models:
        s = summary[k]
        lines.append(f"| {name(k)} | {fmt(s['dice_bladder'])} | {fmt(s['dice_rectum'])} | "
                     f"{fmt(s['dice_bone'])} | {fmt(s['dice_body'])} |")
    lines += ["", "### Resources", "",
              "| Model | Parameters | Inference per scan (s) | Peak inference VRAM (GB) | Train time per epoch (s) | Peak train VRAM (GB) | Epochs (best) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for k in models:
        r = summary[k]["resources"]
        lines.append(f"| {name(k)} | {r['parameters']:,} | {r['infer_s_median']:.2f} | {r['infer_vram_gb']:.2f} | "
                     f"{r['train_epoch_s_median']:.0f} | {r['train_vram_gb']:.2f} | {r['epochs']} ({r['best_epoch']}) |")
    lines += ["", "Training times come from different GPUs for different runs; compare them within a GPU type only.", "",
              "### Anatomy checks (no ground truth needed)", "",
              "A: background hole inside the body larger than the model's threshold (" +
              ", ".join(f"{name(k)} {hole_limits.get(k, HOLE_LIMIT_ML):.1f} mL" for k in models) +
              f") · B: bladder missing or > {GAP_LIMIT_MM:.0f} mm "
              f"from prostate · C: prostate in several pieces. 'Unacceptable' = prostate Dice < {BAD_PROSTATE} or "
              f"bladder Dice < {BAD_BLADDER}.", "",
              "| Model | Scans flagged (A / B / C) | Unacceptable scans | Unacceptable and flagged | Flagged but acceptable |",
              "| --- | --- | --- | --- | --- |"]
    for k in models:
        c = summary[k]["checks"]
        lines.append(f"| {name(k)} | {c['flagged']} ({c['A']} / {c['B']} / {c['C']}) of {c['n']} | {c['bad']} | "
                     f"{c['bad_flagged']} of {c['bad']} | {c['ok_flagged']} |")
    g = summary["_ground_truth_checks"]
    lines += ["", f"The same checks applied to the ground-truth labels flag {g['flagged']} of {g['n']} scans "
              f"(A {g['A']}, B {g['B']}, C {g['C']}), i.e. their false-alarm rate on correct anatomy."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_regional(summary, models, path):
    regions = ["apex", "mid", "base"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    width = 0.8 / len(models)
    for i, k in enumerate(models):
        label, colour = MODEL_STYLE.get(k, (k, INK_2))
        means = [summary[k][f"prostate_dice_{r}"][0] for r in regions]
        stds = [summary[k][f"prostate_dice_{r}"][1] for r in regions]
        x = np.arange(3) + (i - (len(models) - 1) / 2) * width
        ax.bar(x, means, width * 0.92, color=colour, label=label, yerr=stds,
               error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    ax.set_xticks(range(3), ["Apex third", "Mid third", "Base third"], color=INK)
    ax.set_ylabel("Prostate Dice (test, per-patient mean)", color=INK_2)
    ax.set_ylim(0, 1)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_slices(case, per_model, path):
    first = next(iter(per_model.values()))
    area, base_low = first["true_area"], first["base_low"]
    zs = np.where(area > 0)[0]
    z = np.arange(max(zs.min() - 3, 0), min(zs.max() + 4, len(area)))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax2 = ax.twinx()
    ax2.bar(z, area[z], color="#ece9df", width=0.9, zorder=0)
    ax2.set_yticks([])
    for side in ax2.spines.values():
        side.set_visible(False)
    ax2.set_ylabel("true prostate area per slice (shaded)", color=INK_2, fontsize=8)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    for k, d in per_model.items():
        label, colour = MODEL_STYLE.get(k, (k, INK_2))
        ax.plot(z, d["slice_dice"][z], "-o", color=colour, linewidth=2, markersize=4, label=label)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("axial slice index", color=INK_2)
    ax.set_ylabel("prostate Dice on the slice", color=INK_2)
    lo, hi = ("apex", "base") if base_low else ("base", "apex")
    ax.set_title(f"{case}: slice-by-slice prostate Dice   ({lo} ← → {hi})", color=INK, fontsize=10,
                 loc="left", pad=28)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, ncol=len(per_model), loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.text(0.01, 0.01, "Dice 0 outside the shaded gland = prostate predicted on slices where there is none.",
             color=INK_2, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_patient(rows, models, path):
    patients = sorted({r["patient"] for r in rows})
    base = {p: np.mean([r["dice_prostate"] for r in rows if r["patient"] == p and r["model"] == models[0]])
            for p in patients}
    patients.sort(key=lambda p: base[p])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, k in enumerate(models):
        label, colour = MODEL_STYLE.get(k, (k, INK_2))
        vals = [np.mean([r["dice_prostate"] for r in rows if r["patient"] == p and r["model"] == k]) for p in patients]
        ax.scatter(np.arange(len(patients)) + (i - (len(models) - 1) / 2) * 0.15, vals, s=50, color=colour, label=label,
                   edgecolors="white", linewidths=1.5, zorder=3)
    ax.set_xticks(range(len(patients)), [f"patient {p}" for p in patients], color=INK, fontsize=9)
    ax.set_ylabel("prostate Dice (mean over scans)", color=INK_2)
    ax.set_ylim(0.3, 1.02)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def calibrate(rows, models, path):
    """Pick each model's check-A threshold from the validation scans it got RIGHT:
    the 90th percentile of their hole volume, so ~10% of acceptable scans are flagged.
    Thresholds are fixed here and only then applied to the test patients."""
    limits = {}
    print("\nCheck-A calibration on validation scans (hole volume, mL)")
    print(f"{'model':<14}{'acceptable: median / p90 / max':<34}{'threshold':>10}   unacceptable values")
    for k in models:
        ok = [r["holes_ml"] for r in rows if r["model"] == k and not r["unacceptable"]]
        bad = [r["holes_ml"] for r in rows if r["model"] == k and r["unacceptable"]]
        limits[k] = max(HOLE_LIMIT_ML, float(np.percentile(ok, CALIBRATION_PERCENTILE))) if ok else HOLE_LIMIT_ML
        okd = f"{np.median(ok):.1f} / {np.percentile(ok, 90):.1f} / {max(ok):.1f}" if ok else "-"
        badd = ", ".join(f"{v:.1f}" for v in bad) or "-"
        print(f"{k:<14}{okd:<34}{limits[k]:>10.1f}   {badd}")
    save_json({"hole_limit_ml": limits, "percentile": CALIBRATION_PERCENTILE}, path)
    print(f"Saved {path}")


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.cache_dir, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(args.runs_dir, args.models[0], "split.json")) as f:
        test_ids = json.load(f)[f"{args.split}_cases"]
    hole_limits = {}
    if args.thresholds:
        with open(args.thresholds) as f:
            hole_limits = json.load(f)["hole_limit_ml"]
    print(f"Device: {device} | {len(test_ids)} {args.split} scans | models: {args.models} | "
          f"check-A thresholds: {hole_limits or 'default'}")

    rows, slices, summary, gt_checks = [], {}, {}, []
    labels = {}
    for cid in test_ids:                      # ground truth once, plus its anatomy checks
        _, lab = load_case(args.cache_dir, cid)
        labels[cid] = lab.numpy()
        spacing = np.array(meta[cid]["spacing_dhw"])
        gt_checks.append(anatomy_checks(labels[cid], spacing, float(np.prod(spacing)) / 1000.0))

    for k in args.models:
        run = os.path.join(args.runs_dir, k)
        ckpt = torch.load(os.path.join(run, "best.pt"), map_location=device, weights_only=False)
        model, mode = build_model(ckpt["config"]["model"])
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        scale = ckpt["config"].get("scale", 2)
        times, vrams = [], []
        predict_volume(model, load_case(args.cache_dir, test_ids[0])[0], mode, device, scale=scale)   # warm-up
        for cid in test_ids:
            image, _ = load_case(args.cache_dir, cid)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            pred = predict_volume(model, image, mode, device, scale=scale)
            if device.type == "cuda":
                torch.cuda.synchronize()
                vrams.append(torch.cuda.max_memory_allocated() / 1e9)
            times.append(time.perf_counter() - t0)
            m, s = case_metrics(pred.numpy(), labels[cid], np.array(meta[cid]["spacing_dhw"]),
                                hole_limits.get(k, HOLE_LIMIT_ML))
            rows.append({"model": k, "case": cid, "patient": patient_of(cid), **m})
            slices.setdefault(cid, {})[k] = s
            print(f"  {k:8s} {cid}: prostate {m['dice_prostate']:.3f} | apex {m['prostate_dice_apex']:.3f} "
                  f"mid {m['prostate_dice_mid']:.3f} base {m['prostate_dice_base']:.3f} | "
                  f"flags {'A' if m['A_holes'] else '-'}{'B' if m['B_bladder'] else '-'}{'C' if m['C_pieces'] else '-'}")

        mrows = [r for r in rows if r["model"] == k]
        for r in mrows:
            r["abs_apex_extent_mm"], r["abs_base_extent_mm"] = abs(r["apex_extent_mm"]), abs(r["base_extent_mm"])
        keys = ([f"dice_{n}" for n in LABEL_NAMES] + [f"prostate_dice_{x}" for x in ("apex", "mid", "base")] +
                ["apex_extent_mm", "base_extent_mm", "abs_apex_extent_mm", "abs_base_extent_mm",
                 "leak_rectum_ml", "leak_bladder_ml", "prostate_hd95_mm"])
        summary[k] = {key: per_patient(mrows, key) for key in keys}
        with open(os.path.join(run, "history.json")) as f:
            hist = json.load(f)
        summary[k]["resources"] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "infer_s_median": float(np.median(times)),
            "infer_vram_gb": float(max(vrams)) if vrams else float("nan"),
            "train_epoch_s_median": float(np.median([h["epoch_time_s"] for h in hist])),
            "train_vram_gb": float(max(h["peak_vram_gb"] for h in hist)),
            "epochs": len(hist), "best_epoch": ckpt["epoch"]}
        count = lambda cond: int(sum(bool(cond(r)) for r in mrows))
        summary[k]["checks"] = {
            "n": len(mrows), "flagged": count(lambda r: r["flagged"]),
            "A": count(lambda r: r["A_holes"]), "B": count(lambda r: r["B_bladder"]),
            "C": count(lambda r: r["C_pieces"]), "bad": count(lambda r: r["unacceptable"]),
            "bad_flagged": count(lambda r: r["unacceptable"] and r["flagged"]),
            "ok_flagged": count(lambda r: (not r["unacceptable"]) and r["flagged"])}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary["_ground_truth_checks"] = {"n": len(gt_checks),
                                       **{key: int(sum(bool(c[f]) for c in gt_checks)) for key, f in
                                          [("flagged", "flagged"), ("A", "A_holes"), ("B", "B_bladder"), ("C", "C_pieces")]}}

    with open(os.path.join(args.out_dir, "per_case.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json(summary, os.path.join(args.out_dir, "summary.json"))
    write_markdown(summary, args.models, os.path.join(args.out_dir, "summary.md"), args.split, hole_limits)
    if args.calibrate:
        calibrate(rows, args.models, os.path.join(args.out_dir, "thresholds.json"))
    plot_regional(summary, args.models, os.path.join(args.out_dir, "regional_dice.png"))
    plot_per_patient(rows, args.models, os.path.join(args.out_dir, "per_patient_prostate.png"))
    for cid in args.figure_cases:
        if cid in slices:
            plot_slices(cid, slices[cid], os.path.join(args.out_dir, f"slice_dice_{cid}.png"))
    with open(os.path.join(args.out_dir, "summary.md")) as f:
        print("\n" + f.read())
    print(f"Outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
