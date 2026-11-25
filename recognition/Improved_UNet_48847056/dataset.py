"""
dataset.py - data loading, patient-level splitting and augmentation for HipMRI segmentation.
"""

import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c : c + 1][arr == c] = 1
    return res


# load medical image functions
def load_data_2D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype=np.float32,
    getAffines=False,
    early_stop=False,
):
    """
    Load medical image data from names, cases list provided into a list for
    each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive
    memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for
    quick loading and testing scripts.
    """

    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching="unchanged")
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data
        if inImage.shape != (rows, cols):  # ADDED: a few slices are 256x144, not 256x128
            inImage = resize_2d(inImage, (rows, cols), is_label=(dtype == np.uint8))
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


def load_data_3D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype=np.float32,
    getAffines=False,
    orient=False,
    early_stop=False,
):
    """
    Load medical image data from names, cases list provided into a list for
    each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive
    memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    orient: Apply orientation and resample image? Good for images with large
    slice thickness or anisotropic resolution
    dtype: Type of the data. If dtype=np. uint8, it is assumed that the data
    is labels
    early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for
    quick loading and testing scripts.
    """

    affines = []

    interp = "linear"
    if dtype == np.uint8:  # assume labels
        interp = "nearest"

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    first_case = niftiImage.get_fdata(caching="unchanged")
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[
                i, : inImage.shape[0], : inImage.shape[1], : inImage.shape[2], :
            ] = inImage
        else:
            images[
                i, : inImage.shape[0], : inImage.shape[1], : inImage.shape[2]
            ] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images



NUM_CLASSES = 6  # labels 0..5 including background 


def resize_2d(arr, size, is_label=False):
    """Resize a 2D array to size=(rows, cols). Labels use nearest-neighbour so class
    values are never blended; images use bilinear."""
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    if is_label:
        t = torch.nn.functional.interpolate(t, size=size, mode="nearest")
    else:
        t = torch.nn.functional.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def patient_id(path):
    """'case_004_week_0_slice_3.nii.gz' / 'seg_004_...' -> '004'; 'B006_Week0_LFOV.nii.gz' -> 'B006'."""
    parts = os.path.basename(path).split("_")
    return parts[1] if parts[0].lower() in ("case", "seg") else parts[0]


def check_no_leakage(train_files, val_files, test_files):
    """Raise if any patient has files in more than one split."""
    sets = [{patient_id(f) for f in files} for files in (train_files, val_files, test_files)]
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        if sets[i] & sets[j]:
            raise AssertionError(f"Patient leakage between splits: {sorted(sets[i] & sets[j])}")
    print("No patient overlap. Patients train/val/test:", *(len(s) for s in sets))


def get_files_2d(root, split):
    """Sorted (image_files, label_files) for split in 'train' / 'validate' / 'test'.

    root is the keras_slices_data folder. The course already provides the three splits.
    """
    img_dir = os.path.join(root, f"keras_slices_{split}")
    seg_dir = os.path.join(root, f"keras_slices_seg_{split}")
    images = sorted(os.listdir(img_dir))
    images = [os.path.join(img_dir, f) for f in images]
    labels = [os.path.join(seg_dir, os.path.basename(f).replace("case_", "seg_", 1)) for f in images]
    return images, labels


def get_files_3d(root, ratios=(0.7, 0.15, 0.15)):
    """3D volumes come in a single folder, so split by patient here (all weeks stay together).

    Returns {'train': (images, labels), 'validate': ..., 'test': ...}.
    """
    img_dir = os.path.join(root, "semantic_MRs")
    seg_dir = os.path.join(root, "semantic_labels_only")
    images = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
    labels = [os.path.join(seg_dir, os.path.basename(f).replace("LFOV", "SEMANTIC")) for f in images]
    patients = sorted({patient_id(f) for f in images})
    np.random.RandomState(42).shuffle(patients)  # fixed seed -> reproducible split
    n_train, n_val = round(ratios[0] * len(patients)), round(ratios[1] * len(patients))
    owner = {}
    for k, p in enumerate(patients):
        owner[p] = "train" if k < n_train else "validate" if k < n_train + n_val else "test"
    out = {s: ([], []) for s in ("train", "validate", "test")}
    for img, lab in zip(images, labels):
        out[owner[patient_id(img)]][0].append(img)
        out[owner[patient_id(img)]][1].append(lab)
    check_no_leakage(*(out[s][0] for s in ("train", "validate", "test")))
    return out


class SegDataset(Dataset):
    """Wraps arrays from load_data_2D / load_data_3D.

    images: (N, *spatial) float, labels: (N, *spatial) integer class indices (NOT one-hot;
    load them with categorical=False and use F.one_hot where a one-hot target is needed).
    Returns image (1, *spatial) float32 and mask (*spatial) int64.

    flip_axis: which spatial axis to mirror in augmentation. It MUST be the patient's
    left-right axis; flipping the anterior-posterior or superior-inferior axis would swap
    anatomy (e.g. bladder behind the rectum). For the 2D HipMRI slices (256 x 128) the long
    axis, 0, is left-right (checked by plotting a slice). For 3D, check with a plot first.
    """

    def __init__(self, images, labels, augment=False, flip_axis=None):
        self.x = torch.from_numpy(images).unsqueeze(1)
        self.y = torch.from_numpy(labels)
        self.augment = augment
        self.flip_axis = flip_axis

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, y = self.x[i], self.y[i].long()
        if self.augment:
            if self.flip_axis is not None and torch.rand(1) < 0.5:  # mirror image+mask left-right
                x, y = x.flip(self.flip_axis + 1), y.flip(self.flip_axis)  # +1: x has a channel dim
            if torch.rand(1) < 0.5:  # intensity jitter on the image only
                x = x * (0.9 + 0.2 * torch.rand(1)) + 0.2 * (torch.rand(1) - 0.5)
        return x, y


def get_dataloaders_2d(root, batch_size=16, num_workers=0, early_stop=False):
    """Train/val/test loaders for the 2D slices, using the course's provided splits."""
    files = {s: get_files_2d(root, s) for s in ("train", "validate", "test")}
    check_no_leakage(*(files[s][0] for s in ("train", "validate", "test")))
    loaders = []
    for split in ("train", "validate", "test"):
        imgs, labs = files[split]
        x = load_data_2D(imgs, normImage=True, early_stop=early_stop)
        y = load_data_2D(labs, dtype=np.uint8, early_stop=early_stop)
        loaders.append(DataLoader(SegDataset(x, y, augment=(split == "train"), flip_axis=0),
                                  batch_size=batch_size, shuffle=(split == "train"),
                                  num_workers=num_workers))
    return loaders


def load_volumes(files, size, is_label, early_stop=False):
    """Load 3D Nifti volumes and resample each to a fixed size=(H, W, D).

    Volumes differ slightly in shape, so every one is resampled (nearest for labels so
    class values are never blended, trilinear for images) instead of padded/clipped.
    Images are z-score normalised per volume. Returns (N, H, W, D) float32 / uint8.
    """
    if early_stop:
        files = files[:4]
    out = np.zeros((len(files),) + tuple(size), dtype=np.uint8 if is_label else np.float32)
    for i, f in enumerate(tqdm(files, desc="labels" if is_label else "images")):
        v = nib.load(f).get_fdata(caching="unchanged")
        if v.ndim == 4:
            v = v[..., 0]  # sometimes an extra dim in HipMRI data
        t = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))[None, None]
        if is_label:
            t = torch.nn.functional.interpolate(t, size=size, mode="nearest")
        else:
            t = torch.nn.functional.interpolate(t, size=size, mode="trilinear", align_corners=False)
            t = (t - t.mean()) / (t.std() + 1e-8)
        out[i] = t[0, 0].numpy()
    return out


def volumes_to_slices(vols):
    """(N, H, W, D) -> (N*D, H, W): every axial slice (last axis) becomes a 2D sample."""
    return np.ascontiguousarray(np.moveaxis(vols, -1, 1).reshape(-1, *vols.shape[1:3]))


def get_data_3d(root, dims, size=(128, 128, 64), batch_size=2, flip_axis=None,
                num_workers=0, early_stop=False):
    """Data for the 2D-vs-3D comparison, all from ONE patient-level split of the 3D volumes.

    root: HipMRI_Study_open folder (contains semantic_MRs / semantic_labels_only).
    dims=3: training samples are whole volumes. dims=2: training samples are the axial
    slices of the same training volumes. Validation/test are always returned as whole
    volumes (x, y arrays) so both models are scored on identical test patients.

    Returns train_loader, (val_x, val_y), (test_x, test_y), test_files.
    """
    split = get_files_3d(root)
    data = {}
    for s in ("train", "validate", "test"):
        imgs, labs = split[s]
        data[s] = (load_volumes(imgs, size, False, early_stop), load_volumes(labs, size, True, early_stop))
    tx, ty = data["train"]
    if dims == 2:
        tx, ty = volumes_to_slices(tx), volumes_to_slices(ty)
    train_loader = DataLoader(SegDataset(tx, ty, augment=True, flip_axis=flip_axis),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, data["validate"], data["test"], split["test"][0]


if __name__ == "__main__":
    import sys

    root, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "2d_slices")
    if mode == "2d_slices":  # course-provided keras slices
        train_loader, val_loader, test_loader = get_dataloaders_2d(root, early_stop=True)
    else:  # '2d' or '3d' from the 3D volumes
        train_loader, *_ = get_data_3d(root, dims=int(mode[0]), early_stop=True)
    x, y = next(iter(train_loader))
    print("image", tuple(x.shape), x.dtype, "| mask", tuple(y.shape), y.dtype, "| labels", y.unique().tolist())