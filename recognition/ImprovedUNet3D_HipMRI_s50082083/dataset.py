"""
Data loading, patient-level splitting, preprocessing and augmentation for the
HipMRI 3D dataset (CSIRO "Labelled weekly MR images of the male pelvis").

Pipeline:
    1. find_cases()      pair every MR volume with its label volume (patient + week)
    2. split_patients()  leakage-free split by PATIENT (repeat weekly scans of one
                         patient never appear in two splits)
    3. preprocess()      one-off: z-score normalise, fix depth to 128 slices, store
                         each case as uncompressed .npy so slices can be read fast
    4. HipMRISlices2D    PyTorch Dataset of full-resolution axial slices (2D models)
    5. HipMRIVolumes3D   PyTorch Dataset of whole volumes downsampled x2 (3D models)

Array layout after preprocessing is (D, H, W) = (axial slice, x, y):
    D = superior-inferior (128 axial slices), H = left-right, W = anterior-posterior.
"""

import glob
import json
import os
import random
import re

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

LABEL_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
NUM_CLASSES = len(LABEL_NAMES)
PROSTATE = 5

# File-name rules, checked against the data audit (211 pairs, 0 unparsed).
PATIENT_RE = re.compile(r"case[_\-]?(\d+)", re.IGNORECASE)
WEEK_RE = re.compile(r"week[_\-]?(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# 1. Finding and pairing files
# --------------------------------------------------------------------------- #
def _is_label(path):
    """A file is a label volume if its name or folder says so (e.g. *_SEMANTIC_*)."""
    name = os.path.basename(path).lower()
    folder = os.path.basename(os.path.dirname(path)).lower()
    return any(k in name for k in ("label", "seg", "semantic")) or any(
        k in folder for k in ("label", "seg")
    )


def find_cases(root):
    """Return a sorted list of cases, one dict per (patient, week) image/label pair."""
    files = sorted(glob.glob(os.path.join(root, "**", "*.nii*"), recursive=True))
    images, labels = {}, {}
    for f in files:
        name = os.path.basename(f)
        patient = PATIENT_RE.search(name)
        if patient is None:
            continue
        week = WEEK_RE.search(name)
        key = (patient.group(1).zfill(3), week.group(1) if week else "0")
        (labels if _is_label(f) else images)[key] = f

    keys = sorted(set(images) & set(labels))
    if not keys:
        raise RuntimeError(f"No image/label pairs found under {root}")
    return [
        {"case_id": f"{p}_w{w}", "patient": p, "week": w,
         "image": images[(p, w)], "label": labels[(p, w)]}
        for p, w in keys
    ]


# --------------------------------------------------------------------------- #
# 2. Leakage-free split by patient
# --------------------------------------------------------------------------- #
def split_patients(patients, n_val=6, n_test=6, seed=42):
    """Shuffle unique patient IDs with a fixed seed and split them train/val/test.

    Splitting by patient (not by scan) matters because each patient has up to
    8 near-identical weekly scans; a scan-level split would leak test anatomy
    into training and inflate the scores.
    """
    patients = sorted(set(patients))
    rng = random.Random(seed)
    rng.shuffle(patients)
    test = sorted(patients[:n_test])
    val = sorted(patients[n_test:n_test + n_val])
    train = sorted(patients[n_test + n_val:])

    assert train and val and test, "Not enough patients for the requested split"
    assert set(train).isdisjoint(val), "Patient leakage: train/val overlap"
    assert set(train).isdisjoint(test), "Patient leakage: train/test overlap"
    assert set(val).isdisjoint(test), "Patient leakage: val/test overlap"
    return train, val, test


def case_ids_for(cases, patients):
    """All case IDs (every weekly scan) belonging to the given patients."""
    wanted = set(patients)
    return [c["case_id"] for c in cases if c["patient"] in wanted]


def patient_of(case_id):
    """'004_w3' -> '004'."""
    return case_id.split("_w")[0]


# --------------------------------------------------------------------------- #
# 3. One-off preprocessing into a fast cache
# --------------------------------------------------------------------------- #
def _fit_depth(volume, depth):
    """Centre-crop or zero-pad axis 2 (the slice axis) to exactly `depth` slices.

    Only one volume in the dataset (256x256x144) needs cropping.
    """
    d = volume.shape[2]
    if d > depth:
        start = (d - depth) // 2
        return volume[:, :, start:start + depth]
    if d < depth:
        before = (depth - d) // 2
        pad = [(0, 0), (0, 0), (before, depth - d - before)]
        return np.pad(volume, pad)
    return volume


def _cache_paths(cache_dir, case_id):
    return (os.path.join(cache_dir, f"{case_id}_img.npy"),
            os.path.join(cache_dir, f"{case_id}_lab.npy"))


def preprocess(cases, cache_dir, depth=128):
    """Normalise each case once and save it as .npy in (D, H, W) layout.

    - image: z-score normalised per volume (mean 0, std 1), stored as float16
    - label: integer class map 0..5, stored as uint8
    Returns metadata per case (patient, voxel spacing in mm, original depth).
    Already-cached cases are skipped, so re-running is cheap.
    """
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    todo = [c for c in cases
            if c["case_id"] not in meta
            or not all(os.path.exists(p) for p in _cache_paths(cache_dir, c["case_id"]))]
    for c in tqdm(todo, desc="Preprocessing"):
        img_nii, lab_nii = nib.load(c["image"]), nib.load(c["label"])
        img = img_nii.get_fdata(dtype=np.float32)
        lab = np.rint(np.asarray(lab_nii.dataobj)).astype(np.int16)
        while img.ndim > 3:          # drop any trailing singleton dimensions
            img = img[..., 0]
        while lab.ndim > 3:
            lab = lab[..., 0]
        assert img.shape == lab.shape, f"{c['case_id']}: image {img.shape} vs label {lab.shape}"
        assert lab.min() >= 0 and lab.max() < NUM_CLASSES, f"{c['case_id']}: unexpected label values"

        orig_depth = img.shape[2]
        img, lab = _fit_depth(img, depth), _fit_depth(lab, depth)
        img = (img - img.mean()) / max(float(img.std()), 1e-6)

        # (H, W, D) -> (D, H, W) so each axial slice is one contiguous block on disk
        img = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float16)
        lab = np.ascontiguousarray(lab.transpose(2, 0, 1)).astype(np.uint8)
        img_path, lab_path = _cache_paths(cache_dir, c["case_id"])
        np.save(img_path, img)
        np.save(lab_path, lab)

        sx, sy, sz = (float(z) for z in img_nii.header.get_zooms()[:3])
        meta[c["case_id"]] = {"patient": c["patient"], "spacing_dhw": [sz, sx, sy],
                              "orig_depth": int(orig_depth)}

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def load_case(cache_dir, case_id):
    """Full-resolution (image, label) of one cached case as tensors of shape (D, H, W)."""
    img_path, lab_path = _cache_paths(cache_dir, case_id)
    img = torch.from_numpy(np.load(img_path).astype(np.float32))
    lab = torch.from_numpy(np.load(lab_path).astype(np.int64))
    return img, lab


# --------------------------------------------------------------------------- #
# 4. Augmentation (training split only)
# --------------------------------------------------------------------------- #
def augment(image, label):
    """Random left-right flip (same for image and label) plus mild intensity jitter.

    image: (1, ..., H, W) float tensor; label: (..., H, W) long tensor.
    H is the left-right axis, and the pelvis is roughly left-right symmetric,
    so a flip produces a realistic new example without moving organs
    relative to each other.
    """
    if random.random() < 0.5:
        image = torch.flip(image, dims=[-2])
        label = torch.flip(label, dims=[-2])
    scale = random.uniform(0.9, 1.1)
    shift = random.uniform(-0.1, 0.1)
    return image * scale + shift, label


# --------------------------------------------------------------------------- #
# 5. 2D slice dataset
# --------------------------------------------------------------------------- #
class HipMRISlices2D(Dataset):
    """Every axial slice of the given cases at full 256x256 resolution.

    Returns (image (1, H, W) float32, label (H, W) int64).
    Volumes are memory-mapped, so a slice is read from disk without loading
    the whole volume into RAM.
    """

    def __init__(self, cache_dir, case_ids, augment=False):
        self.cache_dir = cache_dir
        self.augment = augment
        self.index = []          # (case_id, slice number)
        self.has_prostate = []   # used for prostate-biased sampling
        for cid in case_ids:
            lab = np.load(_cache_paths(cache_dir, cid)[1], mmap_mode="r")
            present = (lab == PROSTATE).any(axis=(1, 2))
            for z in range(lab.shape[0]):
                self.index.append((cid, z))
                self.has_prostate.append(bool(present[z]))
        self._mmaps = {}         # opened lazily inside each DataLoader worker

    def _volume(self, case_id, which):
        key = (case_id, which)
        if key not in self._mmaps:
            path = _cache_paths(self.cache_dir, case_id)[0 if which == "img" else 1]
            self._mmaps[key] = np.load(path, mmap_mode="r")
        return self._mmaps[key]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        cid, z = self.index[i]
        image = torch.from_numpy(np.array(self._volume(cid, "img")[z], dtype=np.float32))[None]
        label = torch.from_numpy(np.array(self._volume(cid, "lab")[z], dtype=np.int64))
        if self.augment:
            image, label = augment(image, label)
        return image, label

    def sample_weights(self, prostate_fraction=0.5):
        """Per-slice weights so that about `prostate_fraction` of sampled slices
        contain prostate (only ~20% of slices do, and the prostate is 0.1% of voxels)."""
        n_pos = sum(self.has_prostate)
        n_neg = len(self.has_prostate) - n_pos
        if n_pos == 0 or n_neg == 0:
            return torch.ones(len(self.has_prostate), dtype=torch.double)
        w_pos, w_neg = prostate_fraction / n_pos, (1 - prostate_fraction) / n_neg
        return torch.tensor([w_pos if p else w_neg for p in self.has_prostate], dtype=torch.double)


# --------------------------------------------------------------------------- #
# 6. 3D volume dataset
# --------------------------------------------------------------------------- #
def downsample(image, label, scale):
    """Shrink a (D, H, W) image/label pair by `scale` in every direction.

    Image: trilinear (smooth averaging). Label: nearest-neighbour, so every
    voxel keeps a real class value (averaging labels would invent classes).
    """
    if scale == 1:
        return image, label
    img = F.interpolate(image[None, None], scale_factor=1 / scale, mode="trilinear", align_corners=False)[0, 0]
    if label is None:
        return img, None
    lab = F.interpolate(label[None, None].float(), scale_factor=1 / scale, mode="nearest")[0, 0]
    return img, lab.long()


class HipMRIVolumes3D(Dataset):
    """Whole volumes downsampled by `scale` (x2: 128x256x256 -> 64x128x128).

    All volumes are downsampled once and kept in RAM (~3 MB each), so training
    never waits on disk. Returns (image (1, D, H, W) float32, label (D, H, W) int64).
    """

    def __init__(self, cache_dir, case_ids, scale=2, augment=False):
        self.case_ids = list(case_ids)
        self.augment = augment
        self.images, self.labels = [], []
        for cid in tqdm(self.case_ids, desc=f"Loading volumes (x{scale} downsampled)", leave=False):
            image, label = load_case(cache_dir, cid)
            image, label = downsample(image, label, scale)
            self.images.append(image.half())
            self.labels.append(label.to(torch.uint8))

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, i):
        image = self.images[i].float()[None]
        label = self.labels[i].long()
        if self.augment:
            image, label = augment(image, label)
        return image, label
