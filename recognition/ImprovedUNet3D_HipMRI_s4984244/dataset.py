"""
Data loading and pre-processing for the HipMRI Study 3D prostate dataset.

The dataset contains 211 pelvic MR volumes from 38 patients. Most patients
were scanned in eight treatment weeks, but eleven have a single scan. Files
are named ``<patient>_Week<n>_LFOV.nii.gz`` (e.g. ``B006_Week0_LFOV.nii.gz``)
and every volume has a semantic label map with six classes (background,
body, bone, bladder, rectum, prostate).

Pipeline:
    1. Pair every MR volume with its label map by (patient, week).
    2. Split by *patient* so that no patient appears in more than one of
       train / validation / test. Scans of the same patient from different
       weeks are almost identical, so a volume-level split would leak test
       anatomy into training. Single-scan and multi-scan patients are split
       separately, so each subset gets a fair share of scans.
    3. Normalise each volume (z-score) and downsample it to a fixed grid.
       Labels are downsampled by averaging their one-hot encoding and taking
       the arg-max, which keeps thin structures better than nearest-neighbour.
    4. Cache the pre-processed tensors so later runs skip the slow Nifti I/O.

Both the 3D model and the 2D baseline read the same cache and the same split,
so they are trained and evaluated on identical patients and voxels.

Run ``python dataset.py --root <data dir>`` for a data audit.
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

DEFAULT_ROOT = "/home/groups/comp3710/HipMRI_Study_open"
IMAGE_DIR = "semantic_MRs"
LABEL_DIR = "semantic_labels_only"

CLASS_NAMES = ("background", "body", "bone", "bladder", "rectum", "prostate")
NUM_CLASSES = len(CLASS_NAMES)

# Volumes are stored as (D, H, W) after loading, where D is the axial slice
# axis. Raw scans are 128 x 256 x 256 at about 1.56 x 1.68 x 1.68 mm. The 128
# axial slices are kept as acquired - the prostate spans only 17-41 of them
# (median 25), and the apex/base slices are exactly what we want to study -
# while the in-plane resolution is halved to fit the 3D model in memory.
TARGET_SHAPE = (128, 128, 128)

SPLIT_FILE = Path(__file__).with_name("splits.json")

_KEY_PATTERN = re.compile(r"^([A-Z]\d+)_Week(\d+)_")


# ---------------------------------------------------------------------------
# Locating files and splitting by patient
# ---------------------------------------------------------------------------

def scan_key(path):
    """Return the (patient, week) key encoded in a HipMRI file name."""
    match = _KEY_PATTERN.match(Path(path).name)
    if match is None:
        raise ValueError(f"Unrecognised HipMRI file name: {path}")
    return match.group(1), int(match.group(2))


def key_to_name(key):
    case, week = key
    return f"{case}_Week{week}"


def name_to_patient(name):
    """Patient ID of a scan name such as ``B006_Week3``."""
    return name.rsplit("_Week", 1)[0]


def find_pairs(root):
    """
    Pair every MR volume with its semantic label map.

    Returns a list of dicts with keys ``case``, ``week``, ``image`` and
    ``label``, sorted by (case, week).
    """
    root = Path(root)
    images = {scan_key(p): p for p in (root / IMAGE_DIR).glob("*.nii*")}
    labels = {scan_key(p): p for p in (root / LABEL_DIR).glob("*.nii*")}
    if not images:
        raise FileNotFoundError(f"No Nifti files found in {root / IMAGE_DIR}")

    missing = sorted(set(images) ^ set(labels))
    if missing:
        names = ", ".join(key_to_name(k) for k in missing[:5])
        raise RuntimeError(f"{len(missing)} scans lack an image or label: {names}")

    return [
        {"case": k[0], "week": k[1], "image": images[k], "label": labels[k]}
        for k in sorted(images)
    ]


def patient_split(scans_per_patient, fractions=(0.7, 0.15, 0.15), seed=3710):
    """
    Randomly assign whole patients to train / val / test.

    ``scans_per_patient`` maps patient ID to its number of scans. Patients
    with one scan and with several scans are shuffled and split separately
    (stratified), otherwise a test set drawn mostly from single-scan patients
    could end up with very few scans. Returns a dict mapping split name to a
    sorted list of patient IDs.
    """
    rng = random.Random(seed)
    strata = defaultdict(list)
    for case, count in sorted(scans_per_patient.items()):
        strata[count > 1].append(case)

    split = {"test": [], "val": [], "train": []}
    for multi in sorted(strata):
        cases = strata[multi]
        rng.shuffle(cases)
        n_test = round(fractions[2] * len(cases))
        n_val = round(fractions[1] * len(cases))
        split["test"] += cases[:n_test]
        split["val"] += cases[n_test:n_test + n_val]
        split["train"] += cases[n_test + n_val:]
    return {name: sorted(cases) for name, cases in split.items()}


def load_or_create_split(pairs, split_file=SPLIT_FILE):
    """
    Load the committed patient split, or create and save it on first use.

    Keeping the split in version control guarantees that every experiment
    (and every re-run by a reviewer) uses exactly the same test patients.
    """
    split_file = Path(split_file)
    if split_file.exists():
        split = json.loads(split_file.read_text())
    else:
        split = patient_split(Counter(p["case"] for p in pairs))
        split_file.write_text(json.dumps(split, indent=2) + "\n")
        print(f"Created new patient split at {split_file}")

    train, val, test = (set(split[s]) for s in ("train", "val", "test"))
    if train & val or train & test or val & test:
        raise RuntimeError(f"A patient appears in more than one split: {split_file}")
    if {p["case"] for p in pairs} != train | val | test:
        raise RuntimeError("Patient split does not match the cases on disk; "
                           f"delete {split_file} to regenerate it.")
    return split


# ---------------------------------------------------------------------------
# Loading and pre-processing volumes
# ---------------------------------------------------------------------------

def load_nifti(path):
    """Load a Nifti volume as a float32 tensor with shape (D, H, W)."""
    data = nib.load(str(path)).get_fdata(caching="unchanged", dtype="float32")
    volume = torch.from_numpy(data)
    if volume.ndim == 4:  # a few HipMRI files carry a trailing singleton axis
        volume = volume[..., 0]
    # Nifti stores (x, y, z); put the slice axis z first.
    return volume.permute(2, 1, 0).contiguous()


def preprocess(image, label, shape=TARGET_SHAPE):
    """
    Normalise an MR volume and resample image and label to ``shape``.

    Returns (image float16 [D, H, W], label uint8 [D, H, W]).
    """
    image = (image - image.mean()) / (image.std() + 1e-8)
    image = F.interpolate(image[None, None], size=shape, mode="trilinear",
                          align_corners=False)[0, 0]

    label = label.round().long().clamp_(0, NUM_CLASSES - 1)
    one_hot = torch.zeros(NUM_CLASSES, *label.shape).scatter_(0, label[None], 1.0)
    soft = F.interpolate(one_hot[None], size=shape, mode="area")[0]
    label = soft.argmax(dim=0)

    return image.half(), label.to(torch.uint8)


def build_cache(pairs, cache_path, shape=TARGET_SHAPE):
    """
    Pre-process every scan once and store the tensors in ``cache_path``.

    The cache is a dict ``name -> (image, label)``; it is reused as long as
    the target shape matches.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        cache = torch.load(cache_path)
        if cache["shape"] == tuple(shape) and len(cache["scans"]) == len(pairs):
            return cache["scans"]
        print("Cache is stale, rebuilding")

    scans = {}
    for i, pair in enumerate(pairs, 1):
        name = key_to_name((pair["case"], pair["week"]))
        image = load_nifti(pair["image"])
        label = load_nifti(pair["label"])
        if image.shape != label.shape:
            raise RuntimeError(f"{name}: image {tuple(image.shape)} vs label "
                               f"{tuple(label.shape)}")
        scans[name] = preprocess(image, label, shape)
        print(f"\rPre-processing {i}/{len(pairs)}", end="", flush=True)
    print()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"shape": tuple(shape), "scans": scans}, cache_path)
    return scans


def load_split_scans(root, cache_dir, shape=TARGET_SHAPE):
    """
    Load (or build) the cache for ``shape`` in ``cache_dir`` and group scan
    names by split.

    Returns (scans, names_by_split) where ``scans`` maps name -> (image, label).
    """
    pairs = find_pairs(root)
    split = load_or_create_split(pairs)
    cache_path = Path(cache_dir) / f"hipmri_{'x'.join(map(str, shape))}.pt"
    scans = build_cache(pairs, cache_path, shape)

    case_to_split = {c: s for s, cases in split.items() for c in cases}
    names = defaultdict(list)
    for p in pairs:
        names[case_to_split[p["case"]]].append(key_to_name((p["case"], p["week"])))
    return scans, dict(names)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def random_affine(image, label, max_rotate=10.0, scale=(0.9, 1.1)):
    """
    Rotate (in the axial plane) and scale an image/label pair together.

    ``image`` has shape (C, *spatial) and ``label`` has shape (*spatial);
    works for both 2D slices and 3D volumes.
    """
    spatial = image.ndim - 1
    angle = math.radians(random.uniform(-max_rotate, max_rotate))
    s = random.uniform(*scale)
    cos, sin = math.cos(angle) / s, math.sin(angle) / s

    # affine_grid uses (x, y[, z]) = (W, H[, D]) ordering; rotate x-y only.
    theta = torch.eye(spatial, spatial + 1)
    theta[0, 0], theta[0, 1] = cos, -sin
    theta[1, 0], theta[1, 1] = sin, cos

    size = (1, 1) + tuple(label.shape)
    grid = F.affine_grid(theta[None], size, align_corners=False)
    image = F.grid_sample(image[None], grid, mode="bilinear",
                          padding_mode="border", align_corners=False)[0]
    label = F.grid_sample(label[None, None].float(), grid, mode="nearest",
                          padding_mode="zeros", align_corners=False)[0, 0]
    return image, label.long()


def augment(image, label):
    """Light, anatomically plausible augmentation for training samples."""
    # Left-right flip only: the pelvis is roughly symmetric left-right, but
    # flipping anterior-posterior would put the rectum in front of the prostate.
    if random.random() < 0.5:
        image, label = image.flip(-1), label.flip(-1)
    if random.random() < 0.5:
        image, label = random_affine(image, label)
    # Intensity jitter mimics scanner gain / bias differences.
    image = image * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
    if random.random() < 0.2:
        image = image + 0.05 * torch.randn_like(image)
    return image, label


# ---------------------------------------------------------------------------
# PyTorch datasets
# ---------------------------------------------------------------------------

class HipMRIVolumes(Dataset):
    """Whole 3D volumes: image (1, D, H, W) float32, label (D, H, W) int64."""

    def __init__(self, scans, names, train=False):
        self.scans = scans
        self.names = list(names)
        self.train = train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image, label = self.scans[self.names[index]]
        image, label = image.float()[None], label.long()
        if self.train:
            image, label = augment(image, label)
        return image, label


class HipMRISlices(Dataset):
    """
    Axial 2D slices of the same volumes, for the 2D U-Net baseline:
    image (1, H, W) float32, label (H, W) int64.
    """

    def __init__(self, scans, names, train=False):
        self.scans = scans
        self.train = train
        depth = scans[names[0]][0].shape[0]
        self.index = [(name, z) for name in names for z in range(depth)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        name, z = self.index[index]
        image, label = self.scans[name]
        image, label = image[z].float()[None], label[z].long()
        if self.train:
            image, label = augment(image, label)
        return image, label


# ---------------------------------------------------------------------------
# Data audit
# ---------------------------------------------------------------------------

def audit(root):
    """Print shapes, spacing, orientation and class balance of the raw data."""
    pairs = find_pairs(root)
    weeks = Counter(p["case"] for p in pairs)
    print(f"{len(pairs)} scans from {len(weeks)} patients "
          f"({min(weeks.values())}-{max(weeks.values())} scans per patient)")

    shapes, spacings, orientations = Counter(), Counter(), Counter()
    class_voxels = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    prostate_slices, bad_labels = [], []
    for p in pairs:
        img = nib.load(str(p["image"]))
        shapes[img.shape] += 1
        spacings[tuple(round(float(z), 2) for z in img.header.get_zooms()[:3])] += 1
        orientations["".join(nib.aff2axcodes(img.affine))] += 1

        label = load_nifti(p["label"]).round().long()
        if label.min() < 0 or label.max() >= NUM_CLASSES:
            bad_labels.append(key_to_name((p["case"], p["week"])))
        class_voxels += torch.bincount(label.clamp(0, NUM_CLASSES - 1).flatten(),
                                       minlength=NUM_CLASSES).double()
        # How many axial slices the gland spans decides how far z can be downsampled.
        is_prostate = label == CLASS_NAMES.index("prostate")
        prostate_slices.append(int(is_prostate.any(dim=2).any(dim=1).sum()))

    print("Shapes (x, y, z):", dict(shapes))
    print("Voxel spacing (mm):", dict(spacings))
    print("Orientation:", dict(orientations))
    print("Out-of-range labels in:", bad_labels or "none")
    total = class_voxels.sum()
    for name, count in zip(CLASS_NAMES, class_voxels):
        print(f"  {name:<10} {100 * count / total:6.2f}% of voxels")
    prostate_slices.sort()
    print(f"Prostate spans {prostate_slices[0]}-{prostate_slices[-1]} axial slices "
          f"(median {prostate_slices[len(prostate_slices) // 2]}); "
          f"{prostate_slices.count(0)} scans have no prostate")

    split = patient_split(weeks)
    for name, cases in split.items():
        print(f"{name:<5}: {len(cases)} patients, "
              f"{sum(weeks[c] for c in cases)} scans")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit or pre-process the HipMRI 3D dataset")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--prepare", metavar="CACHE_DIR",
                        help="instead of auditing, create splits.json and the "
                             "pre-processed cache in CACHE_DIR (run on a CPU node "
                             "so GPU jobs start training straight away)")
    parser.add_argument("--shape", default="x".join(map(str, TARGET_SHAPE)),
                        help="volume size D x H x W for --prepare")
    args = parser.parse_args()
    if args.prepare:
        shape = tuple(int(v) for v in args.shape.split("x"))
        scans, names = load_split_scans(args.root, args.prepare, shape)
        print({split: len(n) for split, n in names.items()}, "scans cached")
    else:
        audit(args.root)
