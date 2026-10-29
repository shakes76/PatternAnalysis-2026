"""Data loading for the HipMRI prostate segmentation project.

Provides a leakage-free, patient-level stratified train/val/test split, NIfTI
volume loading (following the COMP3710 task sheet, Appendix B), the 2D and 3D
Dataset classes, and the random-flip augmentation used during training.
"""
import csv
import os
import random
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

# Where the HipMRI data lives on Rangpur: MRI volumes (LFOV) and the matching label
# volumes (SEMANTIC). Change DATA_DIR if the data is stored somewhere else.
DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open"
IMAGE_DIR = f"{DATA_DIR}/semantic_MRs"
LABEL_DIR = f"{DATA_DIR}/semantic_labels_only"
# The saved split: every script reads this file, so all models use identical patients.
SPLIT_CSV = "results/split.csv"


def list_cases():
    """Return sorted case names such as 'B040_Week3' (one per scan)."""
    return sorted(f.replace("_LFOV.nii.gz", "")
                  for f in os.listdir(IMAGE_DIR) if f.endswith("_LFOV.nii.gz"))


def image_path(case):
    """Path of the MRI file for a case name such as 'B040_Week3'."""
    return f"{IMAGE_DIR}/{case}_LFOV.nii.gz"


def label_path(case):
    """Path of the label file that matches the MRI of this case."""
    # Images and labels pair by swapping LFOV -> SEMANTIC in the filename.
    return f"{LABEL_DIR}/{case}_SEMANTIC.nii.gz"


def patient_of(case):
    """Patient ID is the text before the first underscore."""
    return case.split("_")[0]


def make_split(val_frac=0.15, test_frac=0.15, seed=42):
    """Split by PATIENT so no patient appears in more than one split.

    Patients are grouped by how many scans they have (full course = 8,
    single = 1, otherwise partial) and each group is split separately, so
    train/val/test get a similar mix. Returns {case_name: 'train'|'val'|'test'}.
    """
    # 1. Group the scans (cases) by patient.
    by_patient = defaultdict(list)
    for case in list_cases():
        by_patient[patient_of(case)].append(case)

    # 2. Put each patient in a stratum by number of scans, so that each split gets a
    #    similar mix of full-course, single-scan and partial-course patients.
    strata = defaultdict(list)
    for patient, cases in by_patient.items():
        n = len(cases)
        key = "full" if n == 8 else ("single" if n == 1 else "partial")
        strata[key].append(patient)

    # 3. Inside each stratum, shuffle the patients with a fixed seed (reproducible), then
    #    the first n_test go to test, the next n_val to validation and the rest to train.
    #    max(1, ...) makes sure each stratum gives at least one patient to val and test.
    rng = random.Random(seed)
    patient_split = {}
    for key in sorted(strata):
        patients = sorted(strata[key])
        rng.shuffle(patients)
        n_val = max(1, round(val_frac * len(patients)))
        n_test = max(1, round(test_frac * len(patients)))
        for i, p in enumerate(patients):
            if i < n_test:
                patient_split[p] = "test"
            elif i < n_test + n_val:
                patient_split[p] = "val"
            else:
                patient_split[p] = "train"

    # 4. Every scan inherits the split of its patient, so no patient is ever shared.
    return {c: patient_split[patient_of(c)] for c in list_cases()}


def save_split(split, path=SPLIT_CSV):
    """Write the split to a CSV with columns case, patient, split."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "patient", "split"])
        for case, s in split.items():
            w.writerow([case, patient_of(case), s])


def load_split(path=SPLIT_CSV):
    """Read the saved split back as {case_name: 'train'|'val'|'test'}."""
    with open(path) as f:
        return {r["case"]: r["split"] for r in csv.DictReader(f)}


def check_no_leakage(split):
    """Assert that every patient sits in exactly one split."""
    seen = defaultdict(set)
    for case, s in split.items():
        seen[patient_of(case)].add(s)
    bad = {p: s for p, s in seen.items() if len(s) > 1}
    assert not bad, f"Patients in more than one split: {bad}"


DEPTH = 128    # every volume is cropped or padded to this many slices
PROSTATE = 5   # label value for the prostate (inferred from the audit)


def fix_depth(arr, depth=DEPTH):
    """Centre-crop or zero-pad the slice axis (axis 2) to `depth` slices."""
    d = arr.shape[2]
    if d > depth:      # too many slices: keep the middle `depth` ones
        start = (d - depth) // 2
        return arr[:, :, start:start + depth]
    if d < depth:      # too few slices: add empty (zero) slices equally at both ends
        pad = depth - d
        before = pad // 2
        return np.pad(arr, ((0, 0), (0, 0), (before, pad - before)))
    return arr


def load_case(case):
    """Load one scan: returns (image float32 [D,H,W] z-scored, label uint8 [D,H,W])."""
    # nibabel gives the arrays in (H, W, D) order; the label values are the integers 0-5
    img = np.asarray(nib.load(image_path(case)).dataobj).astype(np.float32)
    lab = np.asarray(nib.load(label_path(case)).dataobj).astype(np.uint8)
    if img.ndim == 4:        # some files carry an extra trailing dimension
        img = img[..., 0]
    if lab.ndim == 4:
        lab = lab[..., 0]
    # the same crop/pad for image and label keeps them aligned
    img, lab = fix_depth(img), fix_depth(lab)
    # z-score per volume (zero mean, unit variance): MRI intensities have no fixed scale,
    # so this puts every scan on a comparable scale; 1e-8 avoids dividing by zero
    img = (img - img.mean()) / (img.std() + 1e-8)
    # (H, W, D) -> (D, H, W) so that img[z] is one axial slice
    return (np.ascontiguousarray(img.transpose(2, 0, 1)),
            np.ascontiguousarray(lab.transpose(2, 0, 1)))


def random_flip(x, y):
    """Augmentation: flip each sample along the height axis with probability 0.5.

    Works on a batch for both models: x is [B,1,H,W] (2D) or [B,1,D,H,W] (3D)
    and y is [B,H,W] or [B,D,H,W]. Image and label are always flipped together,
    so they stay aligned. The height axis is the patient's left-right direction.
    """
    # one random yes/no per sample in the batch (True = flip this sample)
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    ax = x.ndim - 2     # height axis of x: 2 for [B,1,H,W], 3 for [B,1,D,H,W]
    # flip.view(...) reshapes the yes/no vector so it broadcasts over all other axes;
    # torch.where takes the flipped copy only for samples marked True. The label has no
    # channel axis, so its height axis is one lower (ax - 1).
    x = torch.where(flip.view(-1, *([1] * (x.ndim - 1))), x.flip(ax), x)
    y = torch.where(flip.view(-1, *([1] * (y.ndim - 1))), y.flip(ax - 1), y)
    return x, y


class HipMRI3D(Dataset):
    """Whole volumes for the 3D model: image [1,D,H,W], label [D,H,W]."""

    def __init__(self, split, name):
        self.cases = sorted(c for c, s in split.items() if s == name)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        img, lab = load_case(self.cases[i])
        return torch.from_numpy(img)[None], torch.from_numpy(lab.astype(np.int64))


class HipMRI2D(Dataset):
    """Axial slices for the 2D baseline: image [1,H,W], label [H,W].

    empty_ratio=None keeps EVERY slice (use for val/test, so predictions can be
    stacked back into whole volumes). A number such as 0.25 keeps all prostate
    slices plus that many randomly chosen prostate-free slices per prostate
    slice (use for training, so the model also learns what 'no prostate' looks like).
    """

    def __init__(self, split, name, empty_ratio=None, seed=0):
        self.cases = sorted(c for c, s in split.items() if s == name)
        self.imgs, self.labs, self.index = [], [], []
        rng = random.Random(seed)
        for k, case in enumerate(self.cases):
            img, lab = load_case(case)
            self.imgs.append(img.astype(np.float16))   # halves memory
            self.labs.append(lab)
            # which slices of this volume contain at least one prostate voxel
            has = (lab == PROSTATE).any(axis=(1, 2))
            pos = np.flatnonzero(has).tolist()
            neg = np.flatnonzero(~has).tolist()
            if empty_ratio is not None:    # training: keep only some prostate-free slices
                neg = rng.sample(neg, min(len(neg), int(empty_ratio * len(pos))))
            # index maps a dataset position to (volume number, slice number)
            self.index += [(k, z) for z in sorted(pos + neg)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        k, z = self.index[i]
        x = torch.from_numpy(self.imgs[k][z].astype(np.float32))[None]
        y = torch.from_numpy(self.labs[k][z].astype(np.int64))
        return x, y




class HipMRI3DCached(Dataset):
    """Whole volumes for the 3D model, loaded into RAM once (image float16, label uint8).

    Returns image [1,D,H,W] float32 and label [D,H,W] int64.
    """

    def __init__(self, split, name):
        self.cases = sorted(c for c, s in split.items() if s == name)
        self.imgs, self.labs = [], []
        for case in self.cases:
            img, lab = load_case(case)
            self.imgs.append(img.astype(np.float16))   # halves memory
            self.labs.append(lab)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        x = torch.from_numpy(self.imgs[i].astype(np.float32))[None]
        y = torch.from_numpy(self.labs[i].astype(np.int64))
        return x, y


if __name__ == "__main__":
    # Build the split, check there is no leakage, save it and print its sizes.
    split = make_split()
    check_no_leakage(split)
    save_split(split)
    for s in ("train", "val", "test"):
        cases = [c for c, v in split.items() if v == s]
        print(f"{s}: {len(cases)} volumes from {len({patient_of(c) for c in cases})} patients")
