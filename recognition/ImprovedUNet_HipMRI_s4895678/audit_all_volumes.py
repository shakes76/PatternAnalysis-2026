"""Audit every HipMRI 3D label volume: shape, voxel size, labels, prostate size."""
import csv
import os
from collections import Counter

import nibabel as nib
import numpy as np
from tqdm import tqdm

DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open"
LABEL_DIR = f"{DATA_DIR}/semantic_labels_only"
PROSTATE = 5  # our inference from the first volume; this script tests it

# One row per label volume; file names look like B040_Week0_SEMANTIC.nii.gz.
rows = []
for name in tqdm(sorted(os.listdir(LABEL_DIR))):
    case = name.replace("_SEMANTIC.nii.gz", "")
    patient, week = case.split("_")[0], case.split("_")[1]

    # read the label volume; its header holds the voxel size
    lab = nib.load(f"{LABEL_DIR}/{name}")
    y = np.asarray(lab.dataobj).astype(np.int16)
    if y.ndim == 4:                      # some files carry an extra dimension
        y = y[..., 0]

    # voxel size in mm (x, y, z) and the volume of one voxel in mL (1 mL = 1000 mm^3)
    zooms = tuple(round(float(z), 3) for z in lab.header.get_zooms()[:3])
    voxel_ml = float(np.prod(zooms)) / 1000.0
    # prostate size in voxels, in mL, and the number of slices that contain it
    prostate = (y == PROSTATE)
    n_vox = int(prostate.sum())
    slices_with_prostate = int(prostate.any(axis=(0, 1)).sum())

    rows.append({
        "case": case, "patient": patient, "week": week,
        "shape": str(y.shape), "zooms": str(zooms),
        "labels": str(sorted(np.unique(y).tolist())),
        "prostate_vox": n_vox,
        "prostate_ml": round(n_vox * voxel_ml, 1),
        "prostate_slices": slices_with_prostate,
    })

# Save the table, then print a summary: shapes, voxel sizes, label values, prostate sizes.
os.makedirs("results", exist_ok=True)
with open("results/audit.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("\nVolumes audited:", len(rows))
print("Shapes:", Counter(r["shape"] for r in rows))
print("Voxel sizes:", Counter(r["zooms"] for r in rows))
print("Label sets:", Counter(r["labels"] for r in rows))
print("Volumes with NO prostate label:", [r["case"] for r in rows if r["prostate_vox"] == 0])
ml = np.array([r["prostate_ml"] for r in rows])
sl = np.array([r["prostate_slices"] for r in rows])
print(f"Prostate volume (mL): min {ml.min():.1f} | median {np.median(ml):.1f} | max {ml.max():.1f}")
print(f"Slices containing prostate (of 128): min {sl.min()} | median {int(np.median(sl))} | max {sl.max()}")
