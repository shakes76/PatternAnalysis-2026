"""
data_audit.py -- Pre-training data audit for the HipMRI 2D prostate dataset.

Run this BEFORE training anything. It answers four questions that decide how
the rest of the pipeline must be built:

    1. Leakage   -- does any patient (case_XXX) appear in more than one of the
                    provided train / validate / test folders? If yes, the
                    provided split is unsafe and we must re-split by patient.
    2. Pairing   -- does every MR slice have exactly one matching mask?
    3. Geometry  -- what image sizes exist? (Mixed sizes => must resize/pad.)
    4. Labels    -- which label values exist, and how imbalanced are they?
                    (The prostate is expected to be < 2% of pixels.)

Outputs (written to --out):
    audit_report.json       machine-readable summary of everything below
    audit_samples.png       example slices with mask overlays
    audit_label_balance.png per-class pixel share (log scale)

Usage (Rangpur, CPU node -- no GPU needed):
    python data_audit.py --root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data --out audit_out

NumPy is used here deliberately: this is a standalone analysis helper, not part
of the model (modules.py stays PyTorch-only, as the specification requires).
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: Rangpur compute nodes have no display
import matplotlib.pyplot as plt

# Expected label meaning for HipMRI (per the dataset description: body outline,
# bone, bladder, rectum, prostate). The audit VERIFIES which values actually
# occur rather than assuming this mapping is correct.
EXPECTED_LABELS = {0: "background", 1: "body", 2: "bone", 3: "bladder", 4: "rectum", 5: "prostate"}

# Images are named  case_004_week_0_slice_12.nii.gz  and masks are typically
# named  seg_004_week_0_slice_12.nii.gz  -- accept either prefix so both parse
# to the same key: patient "004", week "0", slice 12.
NAME_RE = re.compile(r"(?:case|seg)_(\d+)_week_(\d+)_slice_(\d+)")

SPLITS = {
    "train":    ("keras_slices_train",    "keras_slices_seg_train"),
    "validate": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test":     ("keras_slices_test",     "keras_slices_seg_test"),
}


def parse_key(filename):
    """Return (patient, week, slice) parsed from a filename, or None if it doesn't match."""
    m = NAME_RE.search(filename)
    if m is None:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def load_2d(path):
    """Load a Nifti slice as a 2D array (some HipMRI files carry a trailing singleton dim)."""
    arr = nib.load(path).get_fdata(caching="unchanged")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def audit_split(root, img_dir, seg_dir):
    """Collect pairing, patient, geometry and label statistics for one split."""
    img_path, seg_path = os.path.join(root, img_dir), os.path.join(root, seg_dir)
    img_files = sorted(f for f in os.listdir(img_path) if f.endswith(".nii.gz"))
    seg_files = sorted(f for f in os.listdir(seg_path) if f.endswith(".nii.gz"))

    img_keys = {parse_key(f): f for f in img_files}
    seg_keys = {parse_key(f): f for f in seg_files}
    unparsed = [f for f in img_files + seg_files if parse_key(f) is None]
    paired = sorted(k for k in img_keys if k is not None and k in seg_keys)

    patients = Counter(k[0] for k in paired)
    scans = {(k[0], k[1]) for k in paired}

    shape_counts = Counter()
    label_pixels = Counter()
    slices_with_label = Counter()
    mismatched_shapes = 0
    prostate_area_by_position = defaultdict(list)  # relative slice position -> prostate pixel count

    # slices per scan, to express each slice's position within its scan (0 = first, 1 = last)
    slices_per_scan = defaultdict(int)
    for k in paired:
        slices_per_scan[(k[0], k[1])] = max(slices_per_scan[(k[0], k[1])], k[2] + 1)

    for k in paired:
        img_shape = nib.load(os.path.join(img_path, img_keys[k])).shape[:2]
        seg = load_2d(os.path.join(seg_path, seg_keys[k])).astype(np.int64)
        shape_counts[str(tuple(img_shape))] += 1
        if tuple(img_shape) != seg.shape:
            mismatched_shapes += 1
        values, counts = np.unique(seg, return_counts=True)
        for v, c in zip(values.tolist(), counts.tolist()):
            label_pixels[int(v)] += int(c)
            slices_with_label[int(v)] += 1
        n_slices = slices_per_scan[(k[0], k[1])]
        rel = round(k[2] / max(n_slices - 1, 1), 1)
        prostate_area_by_position[rel].append(int((seg == 5).sum()))

    intensities = []
    for k in paired[:: max(1, len(paired) // 200)]:  # ~200 evenly spaced slices is enough for range stats
        intensities.append(load_2d(os.path.join(img_path, img_keys[k])))
    flat = np.concatenate([a.ravel() for a in intensities]) if intensities else np.array([0.0])

    return {
        "n_image_files": len(img_files),
        "n_mask_files": len(seg_files),
        "n_paired": len(paired),
        "images_without_mask": len([k for k in img_keys if k not in seg_keys]),
        "masks_without_image": len([k for k in seg_keys if k not in img_keys]),
        "unparsed_filenames": unparsed[:10],
        "n_patients": len(patients),
        "patients": sorted(patients),
        "n_scans": len(scans),
        "shapes": dict(shape_counts),
        "image_mask_shape_mismatches": mismatched_shapes,
        "label_pixels": {str(k): v for k, v in sorted(label_pixels.items())},
        "slices_containing_label": {str(k): v for k, v in sorted(slices_with_label.items())},
        "intensity": {"min": float(flat.min()), "max": float(flat.max()),
                      "mean": float(flat.mean()), "std": float(flat.std())},
        "prostate_area_by_relative_position": {
            str(p): {"mean_px": float(np.mean(v)), "frac_with_prostate": float(np.mean(np.array(v) > 0))}
            for p, v in sorted(prostate_area_by_position.items())
        },
        "_paired_keys": paired,
        "_img_keys": img_keys,
        "_seg_keys": seg_keys,
    }


def plot_samples(root, report, out_dir, n=6):
    """Show training slices that contain the prostate, with the mask overlaid."""
    img_dir, seg_dir = SPLITS["train"]
    stats = report["train"]
    candidates = []
    for k in stats["_paired_keys"]:
        seg = load_2d(os.path.join(root, seg_dir, stats["_seg_keys"][k]))
        if (seg == 5).any():
            candidates.append(k)
        if len(candidates) >= 200:
            break
    if not candidates:
        candidates = stats["_paired_keys"][:n]
    if not candidates:
        print("WARNING: no paired training slices -- skipping sample figure")
        return
    n = min(n, len(candidates))
    picks = [candidates[i] for i in np.linspace(0, len(candidates) - 1, n).astype(int)]

    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 6))
    for col, k in enumerate(picks):
        img = load_2d(os.path.join(root, img_dir, stats["_img_keys"][k]))
        seg = load_2d(os.path.join(root, seg_dir, stats["_seg_keys"][k]))
        axes[0, col].imshow(img.T, cmap="gray", origin="lower")
        axes[0, col].set_title(f"case {k[0]} wk {k[1]}\nslice {k[2]}", fontsize=9)
        axes[1, col].imshow(img.T, cmap="gray", origin="lower")
        axes[1, col].imshow(np.ma.masked_where(seg.T == 0, seg.T), cmap="tab10",
                            vmin=0, vmax=9, alpha=0.5, origin="lower")
        for r in range(2):
            axes[r, col].axis("off")
    axes[0, 0].set_ylabel("MR slice")
    axes[1, 0].set_ylabel("mask overlay")
    plt.suptitle("HipMRI training slices containing prostate (label 5)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "audit_samples.png"), dpi=120)
    plt.close()


def plot_label_balance(report, out_dir):
    """Bar chart of each label's share of all training pixels (log scale shows the imbalance)."""
    pix = {int(k): v for k, v in report["train"]["label_pixels"].items()}
    total = sum(pix.values())
    labels = sorted(pix)
    shares = [100 * pix[l] / total for l in labels]
    names = [f"{l}\n{EXPECTED_LABELS.get(l, '?')}" for l in labels]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, shares, color="steelblue")
    plt.yscale("log")
    plt.ylabel("% of training pixels (log)")
    plt.title("HipMRI label imbalance (training split)")
    for b, s in zip(bars, shares):
        plt.text(b.get_x() + b.get_width() / 2, s, f"{s:.2f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "audit_label_balance.png"), dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")
    parser.add_argument("--out", default="audit_out")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    report = {}
    for split, (img_dir, seg_dir) in SPLITS.items():
        print(f"Auditing {split} ...", flush=True)
        report[split] = audit_split(args.root, img_dir, seg_dir)
        r = report[split]
        print(f"   {r['n_image_files']} images, {r['n_mask_files']} masks, {r['n_paired']} matched pairs", flush=True)
        if r["n_paired"] == 0:
            # Fail loudly with the evidence needed to fix the filename pattern,
            # rather than crashing later with an unrelated-looking error.
            img_dir_path, seg_dir_path = os.path.join(args.root, img_dir), os.path.join(args.root, seg_dir)
            print(f"\nERROR: no image/mask pairs matched in '{split}'. Filename pattern not recognised.")
            print("   example image names:", sorted(os.listdir(img_dir_path))[:3])
            print("   example mask names: ", sorted(os.listdir(seg_dir_path))[:3])
            print("   Unparsed filenames:", r["unparsed_filenames"][:5])
            raise SystemExit(1)

    # ---- Leakage check: a patient must belong to exactly one split ----
    sets = {s: set(report[s]["patients"]) for s in SPLITS}
    leakage = {
        "train&validate": sorted(sets["train"] & sets["validate"]),
        "train&test": sorted(sets["train"] & sets["test"]),
        "validate&test": sorted(sets["validate"] & sets["test"]),
    }
    report["leakage"] = leakage
    report["provided_split_is_patient_level"] = not any(leakage.values())

    plot_samples(args.root, report, args.out)
    plot_label_balance(report, args.out)

    # strip internal bookkeeping before saving
    for s in SPLITS:
        for k in ("_paired_keys", "_img_keys", "_seg_keys"):
            report[s].pop(k)
    with open(os.path.join(args.out, "audit_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- Human-readable summary ----
    print("\n" + "=" * 64)
    print("HipMRI 2D DATA AUDIT")
    print("=" * 64)
    for s in SPLITS:
        r = report[s]
        print(f"\n[{s}] slices={r['n_paired']}  patients={r['n_patients']}  scans={r['n_scans']}")
        print(f"   unpaired: images_without_mask={r['images_without_mask']}  "
              f"masks_without_image={r['masks_without_image']}  shape_mismatches={r['image_mask_shape_mismatches']}")
        print(f"   shapes: {r['shapes']}")
        print(f"   intensity: min={r['intensity']['min']:.1f} max={r['intensity']['max']:.1f} "
              f"mean={r['intensity']['mean']:.1f} std={r['intensity']['std']:.1f}")
        total = sum(r["label_pixels"].values())
        shares = {k: f"{100 * v / total:.3f}%" for k, v in r["label_pixels"].items()}
        print(f"   label pixel share: {shares}")
        print(f"   slices containing each label: {r['slices_containing_label']}")
    print("\nPatient overlap between splits:")
    for pair, ids in leakage.items():
        print(f"   {pair}: {len(ids)} patients {ids[:10]}")
    verdict = "SAFE (patient-level)" if report["provided_split_is_patient_level"] else "LEAKY -- must re-split by patient"
    print(f"\nProvided split verdict: {verdict}")
    print("\nProstate presence by relative slice position (train; 0=first slice, 1=last):")
    for p, v in report["train"]["prostate_area_by_relative_position"].items():
        print(f"   pos {p}: {100 * v['frac_with_prostate']:.0f}% of slices contain prostate, mean area {v['mean_px']:.0f}px")
    print(f"\nSaved: {args.out}/audit_report.json, audit_samples.png, audit_label_balance.png")


if __name__ == "__main__":
    main()
