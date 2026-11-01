"""
This script was produced with the help of AI.

Audit the ADNI dataset and create the patient-level validation split.

Expected layout: <root>/AD_NC/{train,test}/{AD,NC}/<patientID>_<sliceNo>.jpeg

Steps:
  1. Count images and unique patients per split (train/test) and class (AD/NC).
  2. Check image size, colour mode and intensity range on a random sample.
  3. Assert that no patient ID appears in both train and test, or in both classes.
  4. Report slices per patient (minimum, maximum, most common).
  5. Split the training patients into train and validation, stratified by class,
     using a fixed seed, and save the patient lists to a JSON file. Every later
     run reads this file, so the split never changes.

Limitation: patients are separated by ID only. If one person appears under
several IDs (e.g. repeat visits), this check cannot detect it.

Usage (from the project folder):
    python scripts/audit_data.py --root /home/groups/comp3710/ADNI --val-frac 0.15 --seed 42
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

SPLITS = ["train", "test"]
CLASSES = ["AD", "NC"]
IMG_EXTS = {".jpeg", ".jpg", ".png"}


def parse_name(path: Path):
    """Split '<patientID>_<sliceNo>.jpeg' into (patient_id, slice_no), both as strings.

    Returns (None, None) if the file name has no underscore.
    """
    stem = path.stem
    if "_" not in stem:
        return None, None
    pid, slice_no = stem.rsplit("_", 1)
    return pid, slice_no


def collect(root: Path):
    """Index every slice image by split, class and patient.

    Args:
        root: ADNI data folder containing AD_NC/.

    Returns:
        index: {split: {class: {patient_id: [paths]}}}
        bad_names: image files whose names do not match <patientID>_<sliceNo>

    Raises:
        FileNotFoundError: if any split/class folder is missing.
    """
    index = {s: {c: defaultdict(list) for c in CLASSES} for s in SPLITS}
    bad_names = []
    for s in SPLITS:
        for c in CLASSES:
            folder = root / "AD_NC" / s / c
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing folder: {folder}")
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() not in IMG_EXTS:
                    continue
                pid, _ = parse_name(p)
                if pid is None:
                    bad_names.append(p)
                    continue
                index[s][c][pid].append(p)
    return index, bad_names


def image_stats(paths, n=200, seed=0):
    """Image size, colour mode and intensity range over a random sample of images.

    Args:
        paths: all image paths to sample from.
        n: maximum number of images to check.
        seed: seed for the random sample.

    Returns:
        (sizes, modes, (min_intensity, max_intensity), number_checked), where
        sizes and modes are Counters of (width, height) and PIL mode.
    """
    rng = random.Random(seed)
    sample = rng.sample(paths, min(n, len(paths)))
    sizes, modes = Counter(), Counter()
    lo, hi = 255, 0
    for p in sample:
        with Image.open(p) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
            ext = im.convert("L").getextrema()
            lo, hi = min(lo, ext[0]), max(hi, ext[1])
    return sizes, modes, (lo, hi), len(sample)


def main():
    """Run the audit checks, print a report, and save the validation split."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root", default="/home/groups/comp3710/ADNI", help="ADNI data folder"
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="fraction of training patients (per class) moved to validation",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="seed for the split and the image sample"
    )
    ap.add_argument(
        "--out", default="splits/val_split.json", help="where to save the patient lists"
    )
    args = ap.parse_args()

    index, bad_names = collect(Path(args.root))

    # 1. Counts
    print("\n== Counts ==")
    print(f"{'split':6} {'class':5} {'images':>8} {'patients':>9}")
    for s in SPLITS:
        for c in CLASSES:
            n_img = sum(len(v) for v in index[s][c].values())
            print(f"{s:6} {c:5} {n_img:8d} {len(index[s][c]):9d}")
    if bad_names:
        print(
            f"\nWARNING: {len(bad_names)} files did not match <pid>_<slice>, e.g. {bad_names[:3]}"
        )

    # 2. Image properties
    all_paths = [
        p for s in SPLITS for c in CLASSES for v in index[s][c].values() for p in v
    ]
    sizes, modes, (lo, hi), n = image_stats(all_paths, seed=args.seed)
    print(f"\n== Image properties (sample of {n}) ==")
    print("sizes:", dict(sizes))
    print("modes:", dict(modes))
    print(f"intensity range: {lo}..{hi}")

    # 3. Leakage checks
    print("\n== Leakage checks ==")
    pids = {(s, c): set(index[s][c]) for s in SPLITS for c in CLASSES}
    train_ids = pids[("train", "AD")] | pids[("train", "NC")]
    test_ids = pids[("test", "AD")] | pids[("test", "NC")]
    overlap_tt = train_ids & test_ids
    overlap_cls = {s: pids[(s, "AD")] & pids[(s, "NC")] for s in SPLITS}
    print("patients in both train and test:", len(overlap_tt))
    for s in SPLITS:
        print(f"patients in both AD and NC ({s}):", len(overlap_cls[s]))
    assert not overlap_tt, f"Train/test patient leakage: {sorted(overlap_tt)[:10]}"
    assert not any(overlap_cls.values()), "A patient appears in both classes"
    print("PASS: no patient-level leakage")

    # 4. Slices per patient
    print("\n== Slices per patient ==")
    for s in SPLITS:
        for c in CLASSES:
            counts = Counter(len(v) for v in index[s][c].values())
            vals = list(counts.elements())
            print(
                f"{s:6} {c:5} min={min(vals)} max={max(vals)} most common={counts.most_common(3)}"
            )

    # 5. Patient-level, stratified validation split from train
    rng = random.Random(args.seed)
    split = {"seed": args.seed, "val_frac": args.val_frac, "train": {}, "val": {}}
    for c in CLASSES:
        ids = sorted(index["train"][c])
        rng.shuffle(ids)
        n_val = round(len(ids) * args.val_frac)
        split["val"][c] = sorted(ids[:n_val])
        split["train"][c] = sorted(ids[n_val:])

    tr = set(split["train"]["AD"] + split["train"]["NC"])
    va = set(split["val"]["AD"] + split["val"]["NC"])
    assert not tr & va, "Train/val patient overlap"

    print("\n== Validation split (patients) ==")
    for part in ["train", "val"]:
        ad, nc = len(split[part]["AD"]), len(split[part]["NC"])
        print(f"{part:5} AD={ad} NC={nc} total={ad + nc} AD share={ad / (ad + nc):.2f}")
    t_ad, t_nc = len(pids[("test", "AD")]), len(pids[("test", "NC")])
    print(
        f"test  AD={t_ad} NC={t_nc} total={t_ad + t_nc} AD share={t_ad / (t_ad + t_nc):.2f}"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2))
    print(f"\nSaved patient lists to {out}")


if __name__ == "__main__":
    main()
