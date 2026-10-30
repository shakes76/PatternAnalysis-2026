"""
Data audit including per-split counts, slice shapes,
intensity ranges, near-empty slices and patient (ID) overlap
"""
import os
import re
import nibabel as nib
import numpy as np
from collections import Counter

# CONSTANTS - Directories
TRAIN_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
TEST_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
VAL_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"

CASE_RE = re.compile(r"case_(\d+)", re.IGNORECASE)
SAMPLE_PER_SPLIT = 200  # slices loaded per split for intensity stats

splits = {"train": TRAIN_DIR, "test": TEST_DIR, "validate": VAL_DIR}

def load_slice(path):
    arr = nib.load(path).get_fdata(caching="unchanged")
    if arr.ndim == 3:  # HipMRI slices sometimes carry an extra trailing dim
        arr = arr[:, :, 0]
    return arr.astype(np.float32)

def case_id(path):
    m = CASE_RE.search(os.path.basename(path))
    return m.group(1) if m else None

def list_nifti(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if ".nii" in f)

def audit(sample_per_split, rng):

    report = {}
    cases = {}
    for name, folder in splits.items():
        files = list_nifti(folder)
 
        # Verify if resizing is required in data preprocessing
        shapes = Counter(tuple(nib.load(f).shape[:2]) for f in files)  # header only
        ids = {case_id(f) for f in files} - {None}
        cases[name] = ids
 
        pick = rng.choice(len(files), min(sample_per_split, len(files)), replace=False)
        mins, maxs, means, empty = [], [], [], 0
        for i in pick:
            a = load_slice(files[i])
            mins.append(a.min()); maxs.append(a.max()); means.append(a.mean())
            if a.max() - a.min() < 1e-6 or (a > a.max() * 0.05).mean() < 0.01:
                empty += 1  # blank or almost entirely background
 
        report[name] = {
            "n_slices": len(files),
            "n_cases": len(ids),
            "shapes": {f"{h}x{w}": c for (h, w), c in shapes.most_common()},
            "n_sampled": len(pick),
            "min": float(np.min(mins)),
            "max": float(np.max(maxs)),
            "median_slice_max": float(np.median(maxs)),
            "mean_of_means": float(np.mean(means)),
            "near_empty_fraction": empty / len(pick),
        }
 
    # Patient leakage check across splits
    overlap = {}
    names = list(cases)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = sorted(cases[names[i]] & cases[names[j]])
            overlap[f"{names[i]} & {names[j]}"] = shared
 
    return report, overlap


def main():
    rng = np.random.default_rng(0)  # fixed seed so the sample is reproducible
    report, overlap = audit(SAMPLE_PER_SPLIT, rng)
 
    print("== Data audit")
    for name, e in report.items():
        print(f"{name}: {e['n_slices']} slices, {e['n_cases']} patients")
        print(f"  shapes: {e['shapes']}")
        print(f"  intensity ({e['n_sampled']} sampled): min {e['min']:.2f}, max {e['max']:.2f}, "
              f"median slice max {e['median_slice_max']:.2f}, mean {e['mean_of_means']:.2f}")
        print(f"  near-empty slices: {e['near_empty_fraction']:.1%}")
 
    print("== Patient overlap between splits (should be 0)")
    for pair, shared in overlap.items():
        print(f"{pair}: {len(shared)} shared patients {shared[:10]}")


if __name__ == "__main__":
    main()