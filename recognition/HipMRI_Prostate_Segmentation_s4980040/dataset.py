"""
dataset.py - Data loading for HipMRI 2D prostate segmentation.

WHAT THIS FILE DOES
-------------------
1. Finds every MRI slice (image) and its matching segmentation mask (the "answer key").
2. Loads them from Nifti (.nii.gz) files, the standard medical imaging format.
3. Cleans them up so every slice looks the same to the model:
   - normalises brightness (z-score),
   - makes every slice exactly 256 x 128 pixels.
4. Wraps everything in PyTorch Dataset / DataLoader objects so train.py can
   pull batches of (image, mask) pairs during training.

THE DATA SPLIT (already done for us by the course, and already patient-level)
    train:    cases 004-035 (31 patients, 11460 slices)  -> the model learns from these
    validate: cases 036-039 (4 patients,    660 slices)  -> checked during training to tune / pick the best epoch
    test:     cases 040-042 (3 patients,    540 slices)  -> only used at the very end for final scores
No patient appears in more than one split, so there is no data leakage.

FILE NAMING
    image: case_004_week_0_slice_10.nii.gz
    mask:  seg_004_week_0_slice_10.nii.gz
    case_004 = patient ID, week_0 = which scan session, slice_10 = which slice of that scan.

MASK LABELS (each pixel in a mask holds one of these numbers)
    0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate

Run `python dataset.py` to sanity-check the data and save a preview image to outputs/.
"""

import os    # building file paths, making folders
import re    # regular expressions, used to pull the patient ID out of a filename
import glob  # finding all files that match a pattern like "*.nii.gz"

import nibabel as nib  # reads Nifti (.nii.gz) medical image files
import numpy as np     # array maths while preprocessing (fine to use here; only modules.py must avoid NumPy)
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Settings used throughout the project
# ---------------------------------------------------------------------------
DATA_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"  # where the data lives on Rangpur
SPLITS = ("train", "validate", "test")
NUM_CLASSES = 6        # labels 0-5, so the model will output 6 channels (one score per class per pixel)
PROSTATE_LABEL = 5     # the class we care most about (target Dice >= 0.75)
TARGET_SHAPE = (256, 128)  # most slices are 256x128; the few 256x144 ones get centre-cropped to this

# Pattern that matches "case_" followed by digits, e.g. "case_004_" -> captures "004"
_CASE_RE = re.compile(r"case_(\d+)_")


def patient_id(path):
    """Get the patient/case number from a filename.

    Example: '.../case_004_week_0_slice_3.nii.gz' -> '004'
    We need this to prove that no patient is in two splits at once.
    """
    filename = os.path.basename(path)            # strip the folder part, keep just the filename
    return _CASE_RE.search(filename).group(1)    # group(1) = the digits inside the brackets of the pattern


def get_file_pairs(split, root=DATA_ROOT):
    """Return a sorted list of (image_path, mask_path) pairs for 'train', 'validate' or 'test'.

    Masks are matched by exact filename (swap 'case_' for 'seg_'), NOT by list position,
    because alphabetical sorting puts slice_10 before slice_2 and positions could drift.
    """
    img_dir = os.path.join(root, f"keras_slices_{split}")      # e.g. .../keras_slices_train
    seg_dir = os.path.join(root, f"keras_slices_seg_{split}")  # e.g. .../keras_slices_seg_train

    pairs = []
    for img_path in sorted(glob.glob(os.path.join(img_dir, "*.nii.gz"))):
        # case_004_week_0_slice_10.nii.gz  ->  seg_004_week_0_slice_10.nii.gz
        seg_name = os.path.basename(img_path).replace("case_", "seg_", 1)
        seg_path = os.path.join(seg_dir, seg_name)

        # Fail loudly if a mask is missing, rather than silently training on bad data
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"No mask found for {img_path}")
        pairs.append((img_path, seg_path))

    if not pairs:  # empty list = wrong path or the data isn't there
        raise FileNotFoundError(f"No images found in {img_dir}")
    return pairs


def load_nifti_2d(path):
    """Load one 2D Nifti slice as a float32 NumPy array.

    Some HipMRI files are stored as (256, 128, 1) instead of (256, 128),
    so we drop that extra last dimension if it's there.
    """
    arr = nib.load(path).get_fdata(caching="unchanged")  # read pixel values from disk
    if arr.ndim == 3:        # shape like (256, 128, 1)
        arr = arr[:, :, 0]   # -> (256, 128)
    return arr.astype(np.float32)


def fit_to_shape(arr, shape=TARGET_SHAPE):
    """Make a 2D array exactly `shape` by centre-cropping (if too big) or zero-padding (if too small).

    The model needs every slice in a batch to be the same size. Most are already 256x128;
    for a 256x144 slice this trims 8 pixel columns off each side, keeping the centre
    (where the prostate is) untouched.
    """
    out = np.zeros(shape, dtype=arr.dtype)   # blank canvas of the target size
    h, w = arr.shape                         # current size
    th, tw = shape                           # target size

    # If the input is BIGGER than the target, start reading from these offsets (crop)
    src_h, src_w = max((h - th) // 2, 0), max((w - tw) // 2, 0)
    # If the input is SMALLER than the target, start writing at these offsets (pad)
    dst_h, dst_w = max((th - h) // 2, 0), max((tw - w) // 2, 0)
    # How many rows/columns actually get copied across
    ch, cw = min(h, th), min(w, tw)

    out[dst_h:dst_h + ch, dst_w:dst_w + cw] = arr[src_h:src_h + ch, src_w:src_w + cw]
    return out


class HipMRI2DDataset(Dataset):
    """PyTorch Dataset for the 2D HipMRI slices.

    A PyTorch Dataset only needs two things:
        __len__      -> how many samples there are
        __getitem__  -> give me sample number `idx`
    The DataLoader (below) then calls __getitem__ repeatedly to build batches.

    Each sample is (image, mask):
        image: float tensor of shape (1, 256, 128)
               the 1 is the channel dimension (greyscale = 1 channel, like RGB would be 3)
        mask:  long (integer) tensor of shape (256, 128), each pixel holds a class 0..5
               it's 'long' because PyTorch's CrossEntropyLoss expects integer class labels

    preload=False (default): read each file from disk when it's requested. Low memory, slower.
    preload=True: read every slice into RAM once at the start (~1.9 GB for train). Faster epochs.
    """

    def __init__(self, split, root=DATA_ROOT, preload=False):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.split = split
        self.pairs = get_file_pairs(split, root)                         # list of (image_path, mask_path)
        self.patients = sorted({patient_id(img) for img, _ in self.pairs})  # unique patient IDs in this split
        # If preloading, load every pair now and keep them in a list; otherwise load lazily later
        self.cache = [self._load(i) for i in range(len(self.pairs))] if preload else None

    def __len__(self):
        return len(self.pairs)

    def _load(self, idx):
        """Read one image/mask pair from disk and preprocess it (returns NumPy arrays)."""
        img_path, seg_path = self.pairs[idx]

        # --- Image ---
        img = load_nifti_2d(img_path)
        # Z-score normalisation: shift so the average pixel is 0 and the spread is 1.
        # MRI brightness varies a lot between scans; this puts every slice on the same scale.
        # The 1e-8 avoids dividing by zero on a completely blank slice.
        img = (img - img.mean()) / (img.std() + 1e-8)

        # --- Mask ---
        # Labels are stored as floats (e.g. 5.0); round and store as small integers.
        # uint8 (0-255) is plenty for labels 0-5 and uses 1/4 the memory of float32.
        mask = np.rint(load_nifti_2d(seg_path)).astype(np.uint8)

        # Make both exactly 256x128 (crop/pad the SAME way so they stay aligned pixel-for-pixel)
        return fit_to_shape(img), fit_to_shape(mask)

    def __getitem__(self, idx):
        # Use the preloaded copy if we have one, otherwise read from disk now
        img, mask = self.cache[idx] if self.cache is not None else self._load(idx)
        # NumPy -> PyTorch tensors. unsqueeze(0) adds the channel dim: (256,128) -> (1,256,128)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask).long()


def check_no_patient_leakage(*datasets):
    """Raise an error if any patient shows up in more than one split.

    Why it matters: slices from the same patient look extremely similar. If a patient were
    in both train and test, the model would basically be tested on data it has already seen,
    and the Dice score would look better than it really is.
    """
    seen = {}  # patient ID -> which split we first saw it in
    for ds in datasets:
        for pid in ds.patients:
            if pid in seen:
                raise RuntimeError(f"Patient {pid} is in both '{seen[pid]}' and '{ds.split}'")
            seen[pid] = ds.split


def get_dataloaders(batch_size=16, root=DATA_ROOT, num_workers=4, preload=False):
    """Build the train / validate / test DataLoaders (this is what train.py will call).

    A DataLoader takes a Dataset and serves it in batches:
        batch_size  -> how many slices per batch (images come out as (16, 1, 256, 128))
        shuffle     -> random order each epoch; ON for training so the model doesn't
                       memorise the order, OFF for validate/test so results are repeatable
        num_workers -> background CPU processes that load the next batch while the GPU trains
        pin_memory  -> small speed-up when copying batches from CPU RAM to the GPU
    """
    train_ds, val_ds, test_ds = (HipMRI2DDataset(s, root, preload) for s in SPLITS)
    check_no_patient_leakage(train_ds, val_ds, test_ds)  # safety check before anything trains

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Sanity check: only runs when you do `python dataset.py` directly,
# NOT when train.py does `from dataset import ...`
# ---------------------------------------------------------------------------
# ===========================================================================
# 3D VOLUMES (for the 3D Improved U-Net)
# ===========================================================================
# The full 3D MRI volumes live in two folders under the HipMRI root:
#   semantic_MRs/          -> the MRI volumes (images)
#   semantic_labels_only/  -> the matching 3D masks (same labels 0-5 as the 2D data)
# We split them by patient using the SAME patients as the 2D data, so the 2D baseline
# and the 3D model are tested on exactly the same people (a fair comparison).
DATA_ROOT_3D = "/home/groups/comp3710/HipMRI_Study_open"
VAL_CASES_3D = {"036", "037", "038", "039"}
TEST_CASES_3D = {"040", "041", "042"}
TARGET_SHAPE_3D = (128, 128, 64)  # every volume is resized to this (the spec says use downsampled 3D data)

_CASE3D_RE = re.compile(r"(\d{3})")                     # first 3-digit number = patient/case ID
_WEEK_RE = re.compile(r"week_?(\d+)", re.IGNORECASE)    # "Week0" / "week_0" -> scan session


def volume_key(path):
    """Make a (case, week) key from a 3D filename, so an image and its mask can be paired
    no matter what extra words (e.g. LFOV, SEMANTIC) each filename contains."""
    name = os.path.basename(path)
    case, week = _CASE3D_RE.search(name), _WEEK_RE.search(name)
    if case is None:
        raise ValueError(f"Can't find a 3-digit case number in {name}")
    return case.group(1), (week.group(1) if week else "0")


def get_volume_pairs(root=DATA_ROOT_3D):
    """Return a list of (image_path, mask_path, case_id) for every 3D volume."""
    img_files = sorted(glob.glob(os.path.join(root, "semantic_MRs", "*.nii*")))
    seg_files = sorted(glob.glob(os.path.join(root, "semantic_labels_only", "*.nii*")))

    # Index the masks by key, refusing duplicates (two masks for one scan = ambiguous)
    segs = {}
    for f in seg_files:
        k = volume_key(f)
        if k in segs:
            raise ValueError(f"Two masks share key {k}: {segs[k]} and {f}")
        segs[k] = f

    pairs, missing, seen = [], [], set()
    for f in img_files:
        k = volume_key(f)
        if k in seen:
            raise ValueError(f"Two images share key {k}, e.g. {f}")
        seen.add(k)
        if k in segs:
            pairs.append((f, segs[k], k[0]))
        else:
            missing.append(os.path.basename(f))
    if missing:
        raise FileNotFoundError(f"{len(missing)} volumes have no mask, e.g. {missing[:3]}")
    if not pairs:
        raise FileNotFoundError(f"No volumes found under {root}")
    return pairs


def load_nifti_3d(path):
    """Load a 3D Nifti volume as float32, dropping the extra 4th dimension some files have."""
    arr = nib.load(path).get_fdata(caching="unchanged")
    if arr.ndim == 4:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def resize_volume(arr, shape, is_mask):
    """Resize a 3D array to `shape`.
    Images use trilinear interpolation (smooth). Masks use nearest-neighbour, so labels
    stay whole numbers (blending label 3 and label 5 into "4" would be nonsense)."""
    t = torch.from_numpy(arr)[None, None]  # interpolate wants (batch, channel, X, Y, Z)
    if is_mask:
        out = torch.nn.functional.interpolate(t, size=shape, mode="nearest")
    else:
        out = torch.nn.functional.interpolate(t, size=shape, mode="trilinear", align_corners=False)
    return out[0, 0].numpy()


def split_of_case(case):
    """Which split a patient belongs to (same patients as the 2D split)."""
    if case in TEST_CASES_3D:
        return "test"
    if case in VAL_CASES_3D:
        return "validate"
    return "train"


class HipMRI3DDataset(Dataset):
    """3D HipMRI volumes, downsampled and kept in RAM.

    Each item is (image, mask):
        image: float tensor (1, 128, 128, 64)  z-score normalised per volume
        mask:  long tensor  (128, 128, 64)     class indices 0..5
    Training volumes get random mirroring along each axis (simple augmentation, as in
    nnU-Net) so the model sees more variety from only ~170 training volumes.
    """

    def __init__(self, split, root=DATA_ROOT_3D, target_shape=TARGET_SHAPE_3D, augment=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.split = split
        self.target_shape = target_shape
        self.augment = (split == "train") if augment is None else augment
        all_pairs = get_volume_pairs(root)
        self.pairs = [(img, seg) for img, seg, case in all_pairs if split_of_case(case) == split]
        self.patients = sorted({case for _, _, case in all_pairs if split_of_case(case) == split})
        print(f"Loading {len(self.pairs)} {split} volumes ({len(self.patients)} patients) into RAM...")
        self.cache = [self._load(i) for i in range(len(self.pairs))]  # small after downsampling (~5 MB each)

    def __len__(self):
        return len(self.pairs)

    def _load(self, idx):
        img_path, seg_path = self.pairs[idx]
        img = load_nifti_3d(img_path)
        mask = np.rint(load_nifti_3d(seg_path))
        if img.shape != mask.shape:  # image and mask must line up voxel-for-voxel
            raise ValueError(f"Shape mismatch {img.shape} vs {mask.shape} for {img_path}")
        img = (img - img.mean()) / (img.std() + 1e-8)  # same z-score normalisation as 2D
        img = resize_volume(img, self.target_shape, is_mask=False)
        mask = resize_volume(mask, self.target_shape, is_mask=True).astype(np.uint8)
        return img.astype(np.float32), mask

    def __getitem__(self, idx):
        img, mask = self.cache[idx]
        img, mask = torch.from_numpy(img), torch.from_numpy(mask).long()
        if self.augment:
            for axis in range(3):              # each axis flipped with 50% chance,
                if torch.rand(1).item() < 0.5:  # always the SAME way for image and mask
                    img, mask = img.flip(axis), mask.flip(axis)
        return img.unsqueeze(0), mask  # add channel dim -> (1, X, Y, Z)


def get_dataloaders_3d(batch_size=2, root=DATA_ROOT_3D, num_workers=0, target_shape=TARGET_SHAPE_3D):
    """Train / validate / test DataLoaders for the 3D volumes (patient-level split checked)."""
    train_ds, val_ds, test_ds = (HipMRI3DDataset(s, root, target_shape) for s in SPLITS)
    check_no_patient_leakage(train_ds, val_ds, test_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # Rangpur has no screen, so draw plots straight into image files
    import matplotlib.pyplot as plt

    # 1. Load all three splits and confirm no patient overlap
    datasets = [HipMRI2DDataset(s) for s in SPLITS]
    check_no_patient_leakage(*datasets)
    print("No patient leakage between splits.")
    for ds in datasets:
        print(f"{ds.split}: {len(ds)} slices from {len(ds.patients)} patients {ds.patients}")

    # 2. Find a training slice that actually contains prostate (many slices don't),
    #    checking every 25th slice so it's quick
    train_ds = datasets[0]
    for idx in range(0, len(train_ds), 25):
        img, mask = train_ds[idx]
        if (mask == PROSTATE_LABEL).any():
            break
    print(f"Sample {idx}: image {tuple(img.shape)} {img.dtype}, "
          f"mask {tuple(mask.shape)} {mask.dtype}, labels {mask.unique().tolist()}")

    # 3. Save a 3-panel picture: raw image | all labels coloured | prostate highlighted on the image.
    #    Panel 3 is how we confirm label 5 really is the prostate.
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    ax[0].imshow(img[0], cmap="gray")                 # img[0] drops the channel dim for plotting
    ax[0].set_title("Image")

    ax[1].imshow(mask, cmap="tab10", vmin=0, vmax=9)  # each label gets its own colour
    ax[1].set_title("All labels (0-5)")

    ax[2].imshow(img[0], cmap="gray")
    # Hide every pixel that ISN'T prostate, then draw the rest in orange on top of the image
    prostate_only = np.ma.masked_where(mask.numpy() != PROSTATE_LABEL, mask.numpy())
    ax[2].imshow(prostate_only, cmap="autumn", alpha=0.6)
    ax[2].set_title(f"Label {PROSTATE_LABEL} overlay (should be prostate)")

    for a in ax:
        a.axis("off")  # hide the pixel-number axes
    plt.tight_layout()
    plt.savefig("outputs/sample_preview.png", dpi=120)
    print("Saved outputs/sample_preview.png")