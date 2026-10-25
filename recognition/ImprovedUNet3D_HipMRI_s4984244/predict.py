"""
Evaluate trained models on the held-out test patients and draw the figures
used in README.md.

    python predict.py --checkpoints runs/unet3d/best.pt runs/unet2d/best.pt

For every checkpoint this reports
    * per-class Dice / IoU and boundary distances (HD95, ASSD in mm),
    * prostate Dice by axial position from apex to base, and prostate
      predicted on slices outside the true gland ("spill"),
    * prostate over-contoured into the rectum / bladder and missed prostate,
    * the number of connected components of the predicted prostate,
    * parameter count, inference latency and peak inference VRAM.
With several checkpoints it also compares the first with each of the others
using paired Wilcoxon tests (per scan and per patient) on the prostate zone
and boundary metrics and on the rectum and bladder.

    --split val          evaluate the validation patients instead of the test patients
    --postprocess lcc    keep only the largest connected piece of each organ

NumPy / SciPy are used here only for evaluation and plotting.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from scipy import ndimage, stats

from dataset import (CLASS_NAMES, DEFAULT_ROOT, NUM_CLASSES, find_pairs,
                     key_to_name, load_split_scans, name_to_patient)
from modules import count_parameters
from utils import dice_iou, load_checkpoint, predict_volume

BLADDER, RECTUM, PROSTATE = 3, 4, 5
DISPLAY_NAMES = {"unet3d": "3D Improved U-Net", "improved2d": "2D Improved U-Net",
                 "unet2d": "2D U-Net (baseline)"}
ORGANS = (BLADDER, RECTUM, PROSTATE)  # single connected structures
LABEL_CMAP = ListedColormap([(0, 0, 0, 0), (0.95, 0.85, 0.6, 0.12), (1, 1, 1, 0.55),
                             (1, 0.85, 0, 0.6), (0.55, 0.3, 0.1, 0.7), (0.9, 0.1, 0.2, 0.75)])
ZONES = (("apex", 0, 1 / 3), ("mid", 1 / 3, 2 / 3), ("base", 2 / 3, 1.0001))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="checkpoints from train.py; the first is the main model")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--out", default="images", help="directory for figures")
    parser.add_argument("--split", choices=("test", "val"), default="test",
                        help="patients to evaluate (val for decisions, test for reporting)")
    parser.add_argument("--postprocess", choices=("none", "lcc"), default="none",
                        help="lcc: keep only the largest connected piece of bladder, "
                             "rectum and prostate")
    parser.add_argument("--limit", type=int, help="evaluate only this many scans")
    parser.add_argument("--failures", type=int, default=4,
                        help="number of lowest-Dice scans to list")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def voxel_spacing(image_path, shape):
    """Voxel size (mm) of the resampled (D, H, W) grid, from the Nifti header."""
    header = nib.load(str(image_path)).header
    zooms = header.get_zooms()[:3][::-1]           # (z, y, x)
    original = header.get_data_shape()[:3][::-1]
    return tuple(float(z) * o / s for z, o, s in zip(zooms, original, shape))


def surface_distances(pred, gt, spacing):
    """Symmetric boundary-to-boundary distances in mm, or None if a mask is empty."""
    if not pred.any() or not gt.any():
        return None
    # Every border voxel lies inside the joint bounding box, so the distances are the
    # same when computed there; a one-voxel margin keeps the erosion unchanged.
    box = tuple(slice(max(idx.min() - 1, 0), idx.max() + 2)
                for idx in np.nonzero(pred | gt))
    pred, gt = pred[box], gt[box]
    pred_border = pred ^ ndimage.binary_erosion(pred)
    gt_border = gt ^ ndimage.binary_erosion(gt)
    to_gt = ndimage.distance_transform_edt(~gt_border, sampling=spacing)
    to_pred = ndimage.distance_transform_edt(~pred_border, sampling=spacing)
    return np.concatenate([to_gt[pred_border], to_pred[gt_border]])


def superior_is_up(gt):
    """True if larger slice indices are superior (the bladder sits above the prostate)."""
    z = np.arange(gt.shape[0])
    def centre(c):
        return (z * (gt == c).sum(axis=(1, 2))).sum() / max((gt == c).sum(), 1)
    return centre(BLADDER) > centre(PROSTATE)


def prostate_profile(pred, gt, margin=3):
    """
    Per-slice prostate Dice against normalised position t (0 = apex, 1 = base).

    Slices up to ``margin`` beyond the gland are included (t < 0 or t > 1);
    any prostate predicted there is spill into non-prostate anatomy.
    """
    p, g = pred == PROSTATE, gt == PROSTATE
    zs = np.where(g.any(axis=(1, 2)))[0]
    if len(zs) < 2:
        return []
    z0, z1 = zs.min(), zs.max()
    up = superior_is_up(gt)
    rows = []
    for z in range(max(z0 - margin, 0), min(z1 + margin, gt.shape[0] - 1) + 1):
        t = (z - z0) / (z1 - z0) if up else (z1 - z) / (z1 - z0)
        inter, total = (p[z] & g[z]).sum(), p[z].sum() + g[z].sum()
        rows.append({"z": int(z), "t": float(t), "pred_voxels": int(p[z].sum()),
                     "dice": float(2 * inter / total) if total else float("nan")})
    return rows


def scan_metrics(pred, gt, spacing):
    """All per-scan metrics for one prediction (NumPy label maps)."""
    dice, iou = dice_iou(torch.from_numpy(pred), torch.from_numpy(gt), NUM_CLASSES)
    hd95, assd = [], []
    for c in range(1, NUM_CLASSES):
        d = surface_distances(pred == c, gt == c, spacing)
        hd95.append(float(np.percentile(d, 95)) if d is not None else float("nan"))
        assd.append(float(d.mean()) if d is not None else float("nan"))

    ml = float(np.prod(spacing)) / 1000.0
    p, g = pred == PROSTATE, gt == PROSTATE
    profile = prostate_profile(pred, gt)
    inside = [r for r in profile if 0 <= r["t"] <= 1]
    zones = {}
    for name, lo, hi in ZONES:
        values = [r["dice"] for r in inside if lo <= r["t"] < hi]
        zones[name] = float(np.nanmean(values)) if values else float("nan")
    return {
        "dice": dice.tolist(), "iou": iou.tolist(), "hd95": hd95, "assd": assd,
        "zone_dice": zones,
        "spill_slices": sum(r["pred_voxels"] > 0 for r in profile if not 0 <= r["t"] <= 1),
        "prostate_ml": float(g.sum() * ml),
        "over_rectum_ml": float((p & (gt == RECTUM)).sum() * ml),
        "over_bladder_ml": float((p & (gt == BLADDER)).sum() * ml),
        "missed_prostate_ml": float((g & ~p).sum() * ml),
        "extra_prostate_ml": float((p & ~g).sum() * ml),
        "components": int(ndimage.label(p)[1]),
        "profile": profile,
    }


@torch.no_grad()
def profile_inference(model, model_name, image, device, repeats=10):
    """Mean latency (s) and peak VRAM (GB) for one volume."""
    amp = device.type == "cuda"
    run = lambda: predict_volume(model, image, model_name)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        for _ in range(2):
            run()
        if amp:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        tic = time.perf_counter()
        for _ in range(repeats):
            run()
        if amp:
            torch.cuda.synchronize()
    latency = (time.perf_counter() - tic) / repeats
    peak = torch.cuda.max_memory_allocated() / 1e9 if amp else None
    return latency, peak


def keep_largest_components(probs):
    """
    Label map from class probabilities (C, D, H, W), keeping only the largest
    connected piece of each organ in ORGANS. Voxels of smaller pieces go to
    their next most likely class.
    """
    probs = probs.copy()
    for _ in range(3):  # reassigned voxels can rarely form new pieces; repeat
        pred = probs.argmax(axis=0)
        changed = False
        for c in ORGANS:
            pieces, n = ndimage.label(pred == c)
            if n > 1:
                sizes = np.bincount(pieces.ravel())
                sizes[0] = 0
                probs[c][(pieces > 0) & (pieces != sizes.argmax())] = -1.0
                changed = True
        if not changed:
            break
    return probs.argmax(axis=0)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def show(ax, image, label=None, title=None, origin="upper"):
    """
    Draw a slice with an optional label overlay. The volumes are LPS, so
    origin="upper" shows axial slices in the radiological convention
    (anterior up, patient's right on the left); sagittal slices (z, y) need
    origin="lower" to put superior up.
    """
    ax.imshow(image, cmap="gray", origin=origin)
    if label is not None:
        ax.imshow(label, cmap=LABEL_CMAP, vmin=0, vmax=NUM_CLASSES - 1,
                  origin=origin, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)


def error_map(pred, gt):
    """RGBA overlay: prostate false positives red, false negatives blue."""
    rgba = np.zeros(gt.shape + (4,))
    rgba[(pred == PROSTATE) & (gt != PROSTATE)] = (1, 0.1, 0.1, 0.8)
    rgba[(pred != PROSTATE) & (gt == PROSTATE)] = (0.1, 0.4, 1, 0.8)
    return rgba


def prostate_box(masks, pad=12):
    """Crop window (y0, y1, x0, x1) around the union of prostate masks (any shape ending in H, W)."""
    ys, xs = np.where(np.any([m.reshape(-1, *m.shape[-2:]).any(axis=0) for m in masks], axis=0))
    return (max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad)


def slice_dice(pred, gt, z):
    """Prostate Dice on axial slice z (NaN if neither has prostate there)."""
    p, g = pred[z] == PROSTATE, gt[z] == PROSTATE
    total = p.sum() + g.sum()
    return 2 * (p & g).sum() / total if total else float("nan")


def plot_dice_bars(results, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(results)
    x = np.arange(1, NUM_CLASSES)
    for i, (label, res) in enumerate(results.items()):
        d = np.array([s["dice"] for s in res["scans"]])[:, 1:]
        ax.bar(x + (i - (len(results) - 1) / 2) * width, np.nanmean(d, 0), width,
               yerr=np.nanstd(d, 0), capsize=3, label=label)
    ax.axhline(0.7, color="k", ls="--", lw=1, label="target 0.7")
    ax.set_xticks(x, CLASS_NAMES[1:])
    ax.set(ylabel="test Dice", ylim=(0, 1.05))
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_profiles(results, path, bins=10):
    """Left: slice Dice from apex to base. Right: spill beyond the gland."""
    edges = np.linspace(0, 1, bins + 1)
    edges[-1] += 1e-6  # include the base slice (t = 1) in the last bin
    centres = (edges[:-1] + edges[1:]) / 2
    fig, (ax, ax_spill) = plt.subplots(1, 2, figsize=(11, 4),
                                       gridspec_kw={"width_ratios": [3, 1.2]})
    for name, lo, hi in ZONES:
        ax.axvspan(lo, min(hi, 1), alpha=0.04 if name == "mid" else 0.1, color="k")
        ax.text((lo + min(hi, 1)) / 2, 0.03, name, ha="center", fontsize=8)

    width = 0.8 / len(results)
    for i, (label, res) in enumerate(results.items()):
        rows = [r for s in res["scans"] for r in s["profile"]]
        inside = [r for r in rows if 0 <= r["t"] <= 1]
        t = np.array([r["t"] for r in inside])
        d = np.array([r["dice"] for r in inside])
        means = [np.nanmean(d[m]) if (m := (t >= a) & (t < b)).any() else np.nan
                 for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(centres, means, marker="o", label=label)

        # Fraction of slices just outside the gland that still contain prostate.
        below = [r["pred_voxels"] > 0 for r in rows if r["t"] < 0]
        above = [r["pred_voxels"] > 0 for r in rows if r["t"] > 1]
        rates = [np.mean(below) if below else 0, np.mean(above) if above else 0]
        ax_spill.bar(np.arange(2) + (i - (len(results) - 1) / 2) * width, rates, width,
                     label=label)

    ax.set(xlabel="normalised axial position within gland (0 = apex, 1 = base)",
           ylabel="mean slice Dice", ylim=(0, 1.02), title="Prostate Dice by slice position")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax_spill.set_xticks(range(2), ["below apex", "above base"])
    ax_spill.set(ylabel="fraction of slices with predicted prostate", ylim=(0, 1),
                 title="Spill outside the gland")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_examples(images, gts, preds, names, path, view="axial"):
    """Rows = scans; columns = MR, ground truth, then one column per model."""
    labels = list(preds)
    fig, axes = plt.subplots(len(names), 2 + len(labels),
                             figsize=(2.6 * (2 + len(labels)), 2.6 * len(names)), squeeze=False)
    for row, name in zip(axes, names):
        gt = gts[name]
        zc, yc, xc = (int(v) for v in ndimage.center_of_mass(gt == PROSTATE))
        take = (lambda v: v[zc]) if view == "axial" else (lambda v: v[:, :, xc])
        origin = "upper" if view == "axial" else "lower"
        show(row[0], take(images[name]), title=f"{name}", origin=origin)
        show(row[1], take(images[name]), take(gt), "ground truth", origin)
        for ax, label in zip(row[2:], labels):
            show(ax, take(images[name]), take(preds[label][name]), label, origin)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def select_failures(results, preds, gts):
    """
    Pick one test case for each failure type the brief asks about, taking a
    new patient for each type where possible (a patient's weekly scans fail
    alike). Types, judged on the main (first) model unless stated:

    * apex / base   - lowest zone Dice; shown at the worst slice of that zone
    * spill         - most slices with prostate beyond the true apex/base
    * over-contour  - largest prostate volume drawn outside the true gland
    * fragments     - most disconnected prostate pieces in any model; shown
                      at the slice through the smallest piece
    * baseline fails - largest prostate Dice lead of the main model over the
                      worst other model, i.e. where context helps most; shown
                      at the slice where the two differ most

    Returns a list of dicts with keys kind, name, z and why.
    """
    labels = list(results)
    main = labels[0]
    scans = results[main]["scans"]
    used, cases = set(), []

    def take(kind, scored):
        """scored: (score, name, z, why); keep the highest score from an unused patient."""
        scored = sorted((c for c in scored if c[2] is not None), key=lambda c: -c[0])
        if not scored:
            return
        fresh = [c for c in scored if name_to_patient(c[1]) not in used]
        _, name, z, why = (fresh or scored)[0]
        used.add(name_to_patient(name))
        cases.append({"kind": kind, "name": name, "z": int(z), "why": why})

    def worst_in(scan, lo, hi):
        rows = [r for r in scan["profile"] if lo <= r["t"] <= hi and not np.isnan(r["dice"])]
        return min(rows, key=lambda r: r["dice"])["z"] if rows else None

    for kind, lo, hi in (("apex", 0, 1 / 3), ("base", 2 / 3, 1)):
        take(kind, [(-s["zone_dice"][kind], s["name"], worst_in(s, lo, hi),
                     f"{main} {kind} Dice {s['zone_dice'][kind]:.2f}")
                    for s in scans if not np.isnan(s["zone_dice"][kind])])

    spill = []
    for s in scans:
        outside = [r for r in s["profile"] if not 0 <= r["t"] <= 1 and r["pred_voxels"] > 0]
        if outside:
            z = max(outside, key=lambda r: r["pred_voxels"])["z"]
            spill.append((s["spill_slices"], s["name"], z,
                          f"{main} prostate on {s['spill_slices']} slices beyond the gland"))
    take("spill", spill)

    over = []
    for s in scans:
        extra = (preds[main][s["name"]] == PROSTATE) & (gts[s["name"]] != PROSTATE)
        if extra.any():
            z = int(extra.sum(axis=(1, 2)).argmax())
            over.append((s["extra_prostate_ml"], s["name"], z,
                         f"{main} draws {s['extra_prostate_ml']:.1f} mL outside the gland"))
    take("over-contour", over)

    fragments = []
    for label in labels:
        for s in results[label]["scans"]:
            if s["components"] > 1:
                pieces, n = ndimage.label(preds[label][s["name"]] == PROSTATE)
                smallest = 1 + int(np.bincount(pieces.ravel())[1:].argmin())
                z = int(ndimage.center_of_mass(pieces == smallest)[0])
                fragments.append((s["components"], s["name"], z,
                                  f"{label} predicts {s['components']} separate prostate pieces"))
    take("fragments", fragments)

    lead = []
    for s in scans:
        others = [next(x for x in results[l]["scans"] if x["name"] == s["name"]) for l in labels[1:]]
        if not others:
            break
        worst = min(range(len(others)), key=lambda i: others[i]["dice"][PROSTATE])
        gap = s["dice"][PROSTATE] - others[worst]["dice"][PROSTATE]
        if gap > 0:
            gt, a, b = gts[s["name"]], preds[main][s["name"]], preds[labels[1 + worst]][s["name"]]
            zs = [r["z"] for r in s["profile"] if 0 <= r["t"] <= 1]
            z = max(zs, key=lambda z: np.nan_to_num(slice_dice(a, gt, z) - slice_dice(b, gt, z)))
            lead.append((gap, s["name"], z, f"{main} Dice {s['dice'][PROSTATE]:.2f} vs "
                         f"{labels[1 + worst]} {others[worst]['dice'][PROSTATE]:.2f}"))
    take("baseline fails", lead)
    return cases


def plot_failures(images, gts, preds, results, cases, path):
    """
    One row per selected failure case, cropped around the prostate on the
    chosen axial slice. Columns: MR, ground truth, then for every model its
    predicted labels and its prostate errors.
    """
    labels = list(preds)
    columns = 2 + 2 * len(labels)
    fig, axes = plt.subplots(len(cases), columns,
                             figsize=(2.6 * columns, 2.9 * len(cases)), squeeze=False)
    for row, case in zip(axes, cases):
        name, z, gt = case["name"], case["z"], gts[case["name"]]
        y0, y1, x0, x1 = prostate_box([gt == PROSTATE]
                                      + [preds[label][name][z] == PROSTATE for label in labels])
        crop = (lambda v: v[z, y0:y1, x0:x1])
        show(row[0], crop(images[name]), title=f"{case['kind']}: {name}, z={z}")
        show(row[1], crop(images[name]), crop(gt), "ground truth")
        for i, label in enumerate(labels):
            pred = preds[label][name]
            scan = next(x for x in results[label]["scans"] if x["name"] == name)
            show(row[2 + 2 * i], crop(images[name]), crop(pred), f"{label}\nprediction")
            ax = row[3 + 2 * i]
            show(ax, crop(images[name]))
            ax.imshow(error_map(crop(pred), crop(gt)), origin="upper", interpolation="nearest")
            ax.set_title(f"prostate errors\nslice Dice {slice_dice(pred, gt, z):.2f}, "
                         f"3D Dice {scan['dice'][PROSTATE]:.2f}", fontsize=8)
    fig.suptitle("Prostate errors: red = false positive, blue = false negative", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))  # leave room for the title
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(results):
    """Print markdown tables of the headline numbers."""
    print("\n| Model | " + " | ".join(CLASS_NAMES[1:]) + " | mean fg |")
    print("|---" * (NUM_CLASSES + 1) + "|")
    for label, res in results.items():
        d = np.array([s["dice"] for s in res["scans"]])[:, 1:]
        cells = [f"{m:.3f} +/- {sd:.3f}" for m, sd in zip(np.nanmean(d, 0), np.nanstd(d, 0))]
        print(f"| {label} | " + " | ".join(cells) + f" | {np.nanmean(d):.3f} |")

    print("\n| Model | prostate HD95 (mm) | apex Dice | mid Dice | base Dice | spill slices "
          "| into rectum (mL) | into bladder (mL) | missed (mL) | extra (mL) | components |")
    print("|---" * 11 + "|")
    for label, res in results.items():
        s = res["scans"]
        col = lambda f: np.nanmean([f(x) for x in s])
        print(f"| {label} | {col(lambda x: x['hd95'][PROSTATE - 1]):.2f} "
              + " ".join(f"| {col(lambda x, z=z: x['zone_dice'][z]):.3f}" for z, _, _ in ZONES)
              + f" | {col(lambda x: x['spill_slices']):.2f} "
              f"| {col(lambda x: x['over_rectum_ml']):.2f} | {col(lambda x: x['over_bladder_ml']):.2f} "
              f"| {col(lambda x: x['missed_prostate_ml']):.2f} | {col(lambda x: x['extra_prostate_ml']):.2f} "
              f"| {col(lambda x: x['components']):.2f} |")

    print("\n| Model | parameters | inference latency (s/volume) | peak inference VRAM (GB) "
          "| peak training VRAM (GB) | training time (h) |")
    print("|---" * 6 + "|")
    for label, res in results.items():
        r = res["resources"]
        fmt = lambda v, f: format(v, f) if v is not None else "n/a"
        print(f"| {label} | {r['parameters']:,} | {r['latency_s']:.3f} "
              f"| {fmt(r['peak_inference_vram_gb'], '.2f')} "
              f"| {fmt(r.get('peak_train_vram_gb'), '.2f')} | {fmt(r.get('train_hours'), '.2f')} |")


PAIRED_METRICS = {  # metric -> value for one scan; compared between the first two models
    "prostate Dice": lambda s: s["dice"][PROSTATE],
    "prostate apex Dice": lambda s: s["zone_dice"]["apex"],
    "prostate base Dice": lambda s: s["zone_dice"]["base"],
    "prostate HD95 (mm)": lambda s: s["hd95"][PROSTATE - 1],
    "prostate spill slices": lambda s: s["spill_slices"],
    "prostate over-contoured (mL)": lambda s: s["extra_prostate_ml"],
    "prostate missed (mL)": lambda s: s["missed_prostate_ml"],
    "rectum Dice": lambda s: s["dice"][RECTUM],
    "rectum HD95 (mm)": lambda s: s["hd95"][RECTUM - 1],
    "bladder HD95 (mm)": lambda s: s["hd95"][BLADDER - 1],
}


def wilcoxon_p(diff):
    """Two-sided Wilcoxon signed-rank p-value; 1.0 if every difference is zero."""
    diff = diff[~np.isnan(diff)]
    return float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0


def compare(name_a, scans_a, name_b, scans_b):
    """
    Paired comparison of two models on every metric in PAIRED_METRICS: mean
    of each, mean difference (a - b) and two-sided Wilcoxon signed-rank
    p-values per scan and per patient. Scans of one patient are correlated, so
    the per-patient test (on patient means) is the honest one; with 6 patients
    its smallest possible p is 0.031. No correction for multiple comparisons
    is applied, so treat single p-values near 0.05 as exploratory.
    """
    sa = {s["name"]: s for s in scans_a}
    sb = {s["name"]: s for s in scans_b}
    names = sorted(sa)
    owner = np.array([name_to_patient(n) for n in names])
    patients = sorted(set(owner))

    print(f"\nPaired comparison, {name_a} vs {name_b} ({len(names)} scans, {len(patients)} patients):")
    print(f"| Metric | {name_a} | {name_b} | difference | p (scans) | p (patients) |")
    print("|---" * 6 + "|")
    out = {}
    for metric, value in PAIRED_METRICS.items():
        va = np.array([value(sa[n]) for n in names], dtype=float)
        vb = np.array([value(sb[n]) for n in names], dtype=float)
        diff = va - vb
        patient_diff = np.array([np.nanmean(diff[owner == p]) for p in patients])
        row = {"mean_a": float(np.nanmean(va)), "mean_b": float(np.nanmean(vb)),
               "mean_difference": float(np.nanmean(diff)),
               "per_scan_p": wilcoxon_p(diff),
               "per_patient_p": wilcoxon_p(patient_diff) if len(patients) >= 5 else None}
        out[metric] = row
        p_patient = f"{row['per_patient_p']:.3f}" if row["per_patient_p"] is not None else "n/a"
        print(f"| {metric} | {row['mean_a']:.3f} | {row['mean_b']:.3f} | "
              f"{row['mean_difference']:+.3f} | {row['per_scan_p']:.3f} | {p_patient} |")
    return out


def paired_test(results):
    """Compare the main (first) model with each other model."""
    (main, rm), *others = results.items()
    return {f"{main} vs {other}": compare(main, rm["scans"], other, ro["scans"])
            for other, ro in others}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [load_checkpoint(path, device) for path in args.checkpoints]
    shapes = {tuple(ckpt["args"]["shape"]) for _, ckpt in models}
    if len(shapes) != 1:
        raise ValueError(f"Checkpoints were trained at different shapes: {shapes}")
    shape = shapes.pop()

    scans, names = load_split_scans(args.root, args.cache, shape)
    test_names = names[args.split][:args.limit] if args.limit else names[args.split]
    paths = {key_to_name((p["case"], p["week"])): p["image"] for p in find_pairs(args.root)}

    images = {n: scans[n][0].float().numpy() for n in test_names}
    gts = {n: scans[n][1].numpy().astype(np.int64) for n in test_names}
    spacing = {n: voxel_spacing(paths[n], shape) for n in test_names}

    results, preds = {}, {}
    for (model, ckpt), path in zip(models, args.checkpoints):
        label = DISPLAY_NAMES.get(ckpt["model"], ckpt["model"])
        if label in results:
            label = f"{label} ({path})"
        print(f"Evaluating {label} on {len(test_names)} {args.split} scans"
              + (" with largest-component post-processing" if args.postprocess == "lcc" else ""))
        preds[label], per_scan = {}, []
        for n in test_names:
            image = scans[n][0].float()[None].to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                probs = predict_volume(model, image, ckpt["model"])
            if args.postprocess == "lcc":
                pred = keep_largest_components(probs.float().cpu().numpy())
            else:
                pred = probs.argmax(dim=0).cpu().numpy()
            preds[label][n] = pred
            per_scan.append({"name": n, **scan_metrics(pred, gts[n], spacing[n])})

        latency, peak = profile_inference(model, ckpt["model"],
                                          scans[test_names[0]][0].float()[None].to(device), device)
        resources = {"parameters": count_parameters(model), "latency_s": latency,
                     "peak_inference_vram_gb": peak}
        train_metrics = Path(path).with_name("test_metrics.json")
        if train_metrics.exists():
            saved = json.loads(train_metrics.read_text())
            resources.update(train_hours=saved["train_hours"],
                             peak_train_vram_gb=saved["peak_train_vram_gb"])
        results[label] = {"checkpoint": str(path), "split": args.split,
                          "postprocess": args.postprocess, "resources": resources,
                          "scans": per_scan}

    summarise(results)
    model_results = dict(results)  # models only; summary entries are added to results below
    if len(model_results) >= 2:
        results["paired_test"] = paired_test(model_results)

    plot_dice_bars(model_results, out_dir / "dice_comparison.png")
    plot_profiles(model_results, out_dir / "prostate_profile.png")
    # Example figures show the first scan of three different patients.
    firsts = {}
    for n in test_names:
        firsts.setdefault(name_to_patient(n), n)
    examples = list(firsts.values())[:3]
    plot_examples(images, gts, preds, examples, out_dir / "examples_axial.png", "axial")
    plot_examples(images, gts, preds, examples, out_dir / "examples_sagittal.png", "sagittal")
    cases = select_failures(model_results, preds, gts)
    plot_failures(images, gts, preds, model_results, cases, out_dir / "failures.png")
    results["failure_cases"] = cases
    print("\nFailure gallery:")
    for c in cases:
        print(f"  {c['kind']:<12} {c['name']} z={c['z']}: {c['why']}")

    worst = sorted(next(iter(model_results.values()))["scans"], key=lambda s: s["dice"][PROSTATE])
    print("\nLowest prostate Dice (main model):")
    for s in worst[:args.failures]:
        print(f"  {s['name']}: Dice {s['dice'][PROSTATE]:.3f}, HD95 {s['hd95'][PROSTATE - 1]:.1f} mm, "
              f"zones {', '.join(f'{k} {v:.2f}' for k, v in s['zone_dice'].items())}, "
              f"spill {s['spill_slices']}, components {s['components']}")

    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nFigures and results.json written to {out_dir}")


if __name__ == "__main__":
    main()
