import os
import shutil

import numpy as np
import nibabel as nib
import glob
import torch
import random
from torch.utils.data import DataLoader, dataset
from torchvision import transforms
from tqdm import tqdm
from collections import Counter

SEED = 67
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)

# ======================================================================================================================
# =                                               Configurations & Paths                                               =
# ======================================================================================================================
DATASET_PATH_2D = r"./home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
DATASET_PATH_3D = r"./home/groups/comp3710/HipMRI_Study_open"
if not os.path.exists(DATASET_PATH_2D):
    DATASET_PATH_2D = r"/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    DATASET_PATH_3D = r"/home/groups/comp3710/HipMRI_Study_open"
PROCESSED_PATH = r"./processed_npy"

# Dictionary of tuples of raw, mask pairs for 2D
DATASET_2D_TYPES = {
    "train": (DATASET_PATH_2D + r"/keras_slices_train", DATASET_PATH_2D + r"/keras_slices_seg_train"),
    "test": (DATASET_PATH_2D + r"/keras_slices_test", DATASET_PATH_2D + r"/keras_slices_seg_test"),
    "validate": (DATASET_PATH_2D + r"/keras_slices_validate", DATASET_PATH_2D + r"/keras_slices_seg_validate"),
}

# Only provided raw, mask pairs; need to do train-test-val split ourselves
DATASET_3D_TYPES = {
    "raw": DATASET_PATH_3D + r"/semantic_MRs/",
    "mask": DATASET_PATH_3D + r"/semantic_labels_only/"
}
LABELS = ("Background", "Body Outline", "Bone", "Bladder", "Rectum", "Prostate")
IMAGE_SIZE = (256, 128)
BATCH_SIZE_2D = {
    "train":    75,
    "validate": 37,
    "test":     42,
}
BATCH_SIZE_3D = {
    "train":    4,
    "validate": 1,
    "test":     1,
}
TR_ID, VA_ID, TE_ID = range(3)
SPLIT_PROPS = ( # For 3D scans; this one is a tuple because it's easier to work with.
    0.9052,     # Train
    0.0521,     # Validate
    0.0427,     # Test
)
assert sum(SPLIT_PROPS) == 1.0, "Split ratios not correct!"
IMAGE_COUNT_2D = {
    "train":    11460,
    "validate": 660,
    "test":     540,
}
IMAGE_COUNT_3D = {
    "train":    211 * SPLIT_PROPS[TR_ID],
    "validate": 211 * SPLIT_PROPS[VA_ID],
    "test":     211 * SPLIT_PROPS[TE_ID],
}
COUNT_3D_EPS = 0.05 # Splitting 3D scans is a subset-sum problem.
                    # We might not be able to get exact numbers, so we need some leniency when checking if splits
                    # are precise and accurate.

# ======================================================================================================================
# =                                         Helper methods to load Nifti files                                         =
# ======================================================================================================================
def fit_to_shape(img, target_shape: tuple[int, int]):
    """
    Pad/Crop the 2D image array supposedly because the image sizes aren't consistent.
    :param img: The image array to be cropped/padded
    :param target_shape: The target dimensions of the fitted image
    :return: The fitted image
    """
    height, width = img.shape
    target_height, target_width = target_shape

    if target_height <= height and target_width <= width:
        dy = height // 2 - target_height // 2
        dx = width // 2 - target_width // 2
        return img[dy : height - dy, dx : width - dx]
    if (target_height <= height and target_width > width) or (target_height >= height and target_width < width):
        return None     # Or change to raise an error

    # Using centre padding
    dx, dy = target_width - width, target_height - height
    sx, sy = dx // 2, dy // 2    # Scaled pad sizes
    cx = cy = 0                 # Correction factors if dx|dy is odd
    if dx % 2 != 0:
        cx = 1
    if dy % 2 != 0:
        cy = 1

    return np.pad(img, ((sy, sy + cy), (sx, sx + cx)))


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res


# load medical image functions
def load_data_3D(imageNames, normImage=False, categorical=False,
                 dtype=np.float32, getAffines=False, orient=False,
                 early_stop=False):
    """
    Load medical image data from names, cases list provided into a list for
    each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive
    memory usage.

    normImage : bool (normalise the image 0.0-1.0)
    orient : Apply orientation and resample image? Good for images with large
    slice thickness or anisotropic resolution
    dtype : Type of the data. If dtype = np.uint8, it is assumed that the data
    is labels
    early_stop : Stop loading pre-maturely? Leaves arrays mostly empty, for
    quick loading and testing scripts.
    """
    affines = []

    interp = 'linear'
    if dtype == np.uint8:  # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    first_case = niftiImage.get_fdata(caching='unchanged')
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
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :] = inImage
        else:
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


def load_data_2D(imageNames, target_shape=(256, 128), normImage=False,
                 categorical=False, num_classes=None, dtype=np.float32, getAffines=False, early_stop=False):
    """
    Load medical image data from names, cases list provided into a list for
    each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive
    memory usage.

    normImage : bool (normalise the image 0.0-1.0)
    early_stop : Stop loading pre-maturely, leaves arrays mostly empty, for
    quick loading and testing scripts.

    :param imageNames: List of all .nii images in the target directory
    :param target_shape: Target dimensions of the images once loaded in
    :param normImage: idk
    :param categorical: Whether to split into categories using one-hot encoding
    :param num_classes: Number of classes/channels if we were to split into categories
    :param dtype: Data type
    :param getAffines: idk
    :param early_stop: Early stop
    :return: numpy array of loaded Nifti images
    """

    affines = []

    # get fixed size
    num = len(imageNames)
    if categorical:
        assert num_classes is not None, "pass num_classes for masks"
        images = np.zeros((num, num_classes, *target_shape), dtype=dtype)
    else:
        images = np.zeros((num, *target_shape), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        nifti_image = nib.load(inName)
        in_image = nifti_image.get_fdata(caching='unchanged')
        if in_image.ndim == 3:
            in_image = in_image[:, :, 0]

        in_image = fit_to_shape(in_image, target_shape)

        if categorical:
            # fixed class count: no missing-label shape errors
            one_hot = np.eye(num_classes, dtype=dtype)[in_image.astype(int)]
            images[i] = np.moveaxis(one_hot, -1, 0) # Reorders shape from [N,H,W,K] to [N,K,H,W]
        else:
            in_image = in_image.astype(dtype)
            if normImage:
                in_image = (in_image - in_image.mean()) / (in_image.std() + 1e-8)
            images[i] = in_image

        affines.append(nifti_image.affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


# ======================================================================================================================
# =                                             Patient Splitting Helpers                                              =
# ======================================================================================================================
def split3D(image_names: list, mask_names: list, seed: int | str = SEED) -> tuple[set[str]]:
    """
    Perform patient-level split for the 3D HipMRI scans by employing the subset-sum-problem
    :param image_names: The file names (and dir) of the raw scans
    :param mask_names: The file names (and dir) of the masks
    :param seed: Seed for Random()
    :return: 2D tuple representing the patients (file paths) delegated to train, validation, and test sets
    """
    assert len(image_names) == len(mask_names), "Inconsistent number of images and masks"
    patient_scans_count = list()  # Useful for data audit
    split_raw = dict()
    split_mask = dict()

    for index in range(len(image_names)):
        raw_split = image_names[index].replace("\\", r"/").split(r"/")[-1].split("_")
        mask_split = mask_names[index].replace("\\", r"/").split(r"/")[-1].split("_")
        patient_id_raw = raw_split[0]
        patient_id_mask = mask_split[0]
        assert patient_id_raw == patient_id_mask, "Raw/mask mismatch in dataset."

        patient_scans_count.append(patient_id_raw)
        split_raw[patient_id_raw] = raw_split[1]
        split_mask[patient_id_mask] = mask_split[1]

    # Subset-sum problem.
    # Code below generated by Claude Sonnet 5.5, 28 Oct 2026.
    rng = random.Random(seed)
    counts = Counter(patient_scans_count)   # patient -> number of images
    patients = list(counts)
    rng.shuffle(patients)                   # otherwise the DP always favours the same patients
    total = sum(counts.values())

    splits, remaining = [], patients
    for frac in SPLIT_PROPS[:-1]:           # last split takes whatever is left
        sizes = [counts[p] for p in remaining]
        idx = set(subset_sum_pick(sizes, round(frac * total)))
        splits.append({remaining[i] for i in idx})
        remaining = [p for i, p in enumerate(remaining) if i not in idx]
    splits.append(set(remaining))

    # sanity check: no patient appears in two splits
    assert sum(len(s) for s in splits) == len(set().union(*splits))

    return tuple(splits)


def subset_sum_pick(sizes, target):
    """
    Courtesy of Claude Sonnet 5.5, 28 Oct 2026 [3]
    Indices into `sizes` whose sum is closest to `target` (0/1 DP).
    :param sizes: List of numeric values representing item sizes.
    :param target: The target sum to approximate.
    :return: List of indices of `sizes` whose sum is closest to `target`.
    """
    cap = 2 * target
    reach = [False] * (cap + 1)
    parent = [None] * (cap + 1)
    reach[0] = True
    for i, a in enumerate(sizes):
        for s in range(cap, a - 1, -1):          # backwards => each patient used once
            if not reach[s] and reach[s - a]:
                reach[s] = True
                parent[s] = i
    best = min((s for s in range(cap + 1) if reach[s]), key=lambda s: abs(s - target))
    chosen, s = [], best
    while s > 0:
        chosen.append(parent[s])
        s -= sizes[parent[s]]
    return chosen


# Splits via patient id
images = sorted(glob.glob(os.path.join(DATASET_3D_TYPES["raw"], "*.nii*")))
masks = sorted(glob.glob(os.path.join(DATASET_3D_TYPES["mask"], "*.nii*")))
PATIENT_SPLIT = split3D(images, masks)
SPLIT = {
    "train":    sorted(PATIENT_SPLIT[TR_ID]),
    "validate": sorted(PATIENT_SPLIT[VA_ID]),
    "test":     sorted(PATIENT_SPLIT[TE_ID]),
}


# ======================================================================================================================
# =                                           Preprocessing .nii.gz to .npy                                            =
# ======================================================================================================================
def nii_to_npy(data_split: str):
    """
    Preprocess nii.gz scans into npy files with the numpy arrays.
    Should be called if we want to change split ratios.
    """
    dataset_type, dataset_split = data_split.split("_")
    if dataset_split not in SPLIT.keys():
        raise ValueError("Dataset split not recognised.")
    if dataset_type not in {"2D", "3D"}:
        raise ValueError("Dataset type not recognised.")

    save_path = os.path.join(PROCESSED_PATH, dataset_type, dataset_split)
    shutil.rmtree(save_path, ignore_errors=True); os.makedirs(save_path, exist_ok=True)
    print("Starting NIfTI to NumPy conversion...")
    if dataset_type == "2D":
        print(f"Processing 2D {dataset_split} split...")
        img_path, mask_path = DATASET_2D_TYPES[dataset_split]
        img_files = sorted(glob.glob(os.path.join(img_path, "*.nii*")))
        mask_files = sorted(glob.glob(os.path.join(mask_path, "*.nii*")))

        raw_scans = load_data_2D(img_files, target_shape=IMAGE_SIZE, normImage=True)
        mask_scans = load_data_2D(mask_files, target_shape=IMAGE_SIZE, categorical=True, num_classes=len(LABELS), dtype=np.uint8)
        assert len(raw_scans) == len(mask_scans), "Data Mismatch in 2D scans!"
        save_npy(raw_scans, mask_scans, save_path)
    elif dataset_type == "3D":
        patient_ids = SPLIT[dataset_split]
        print(f"Processing 3D {dataset_split} split...")
        split_raw_files, split_mask_files = [], []
        for patient_id in patient_ids:
            split_raw_files.extend(sorted(glob.glob(DATASET_3D_TYPES["raw"] + rf"{patient_id}_*.nii*")))
            split_mask_files.extend(sorted(glob.glob(DATASET_3D_TYPES["mask"] + rf"{patient_id}_*.nii*")))

        raw_scans = load_data_3D(split_raw_files, normImage=True)
        mask_scans = load_data_3D(split_mask_files, categorical=True, dtype=np.uint8)
        assert len(raw_scans) == len(mask_scans), "Data Mismatch in 3D scans!"
        save_npy(raw_scans, mask_scans, save_path)
    print("Preprocessing & conversion complete!")


def save_npy(raws, masks, save_path):
    for index in range(len(raws)):
        raw = raws[index]
        mask = masks[index]
        np.save(os.path.join(save_path, f"raw_{index}.npy"), raw)
        np.save(os.path.join(save_path, f"mask_{index}.npy"), mask)


def verify_splits(datatype: str) -> bool:
    """
    Verify the splits have been conducted and are correct.
    Must be run whenever we want to load data.
    :param datatype: 2D or 3D
    :return: Are the splits correct?
    """
    if datatype not in {"2D", "3D"}:
        raise ValueError("Datatype not recognised!")

    datasets = ("train", "validate", "test")
    for data in datasets:
        path = os.path.join(PROCESSED_PATH, datatype, data)
        if not os.path.exists(path):
            print(f"{datatype} {data} split does not exist!\nProcessing splits...")
            perform_split(datatype, data)
        raw_files = sorted(glob.glob(os.path.join(path, "raw_*.npy")))
        mask_files = sorted(glob.glob(os.path.join(path, "mask_*.npy")))
        assert len(raw_files) == len(mask_files), f"Dataset mismatch in {datatype} split."

        # Check if current split proportions matches what is defined in the code.
        # Should negate the need to manually run the nii_to_npy script whenever we want to change split ratios.
        if datatype == "3D":
            condition = abs(1.0 - (len(raw_files) / IMAGE_COUNT_3D[data])) > COUNT_3D_EPS
        else:
            condition = False   # 2D splits have been done for us; No need to check.
        if condition:
            print(f"{datatype} Splits not correct!\nRe-processing splits...")
            perform_split(datatype)
            return False
    return True


def perform_split(datatype: str, data_set: str | None = None) -> None:
    """
    Converts the dataset into .npy files or update the files and splits.
    :param datatype: 2D or 3D
    :param data_set: Train, Validation, or Test
    """
    print("Splitting to .npy files...")
    if data_set is None:
        for d_set in SPLIT.keys():
            nii_to_npy(f"{datatype}_{d_set}")
    elif data_set is not None and data_set in {"train", "validate", "test"}:
        nii_to_npy(f"{datatype}_{data_set}")
    else:
        raise ValueError("Dataset not recognised!")


# ======================================================================================================================
# =                                         Dataset Classes (Reads from .npy)                                          =
# ======================================================================================================================
class HipMRI3DDataset(torch.utils.data.Dataset):
    """
    HipMRI 3D dataset class.
    Patient-level splits to prevent data leakage.
    """
    PATCH = (   # Each must be divisible by 4 because of the two pooling steps in the model
        80,     # D
        152,    # H
        152,    # W
    )

    def __init__(self, dataset: str, transform=None):
        """
        Performs our own patient split
        :param dataset: train, validate, or test
        :param transform: transforms
        """
        self.random_crop = dataset == "train"
        self.patient_ids = SPLIT[dataset]

        path = os.path.join(PROCESSED_PATH, "3D", dataset)
        self.raw_files = sorted(glob.glob(os.path.join(path, "raw_*.npy")))
        self.mask_files = sorted(glob.glob(os.path.join(path, "mask_*.npy")))

        self.label_values = range(6)    # 0 - Background,
                                        # 1 - Body outline,
                                        # 2 - Bone,
                                        # 3 - Bladder,
                                        # 4 - Rectum,
                                        # 5 - Prostate
        self.transform = transform

    def __len__(self):
        return len(self.raw_files)

    def __getitem__(self, index: int):
        raw_scan = np.load(self.raw_files[index], mmap_mode="r")
        mask_scan = np.load(self.mask_files[index], mmap_mode="r")

        # Better to use from_numpy() here since the masks are assigned an integer in sequential order.
        # for Conv3d, shapes per batch should be (B, 1, D, H, W) / (B, D, H, W) or similar
        raw = torch.from_numpy(raw_scan.copy()).permute(2, 0, 1).unsqueeze(0)   # Becomes [1, D, H, W]
        mask = torch.from_numpy(mask_scan.copy()).permute(3, 2, 0, 1).float()   # Becomes [K, D, H, W]

        if self.random_crop:  # True for train only
            _, D, H, W = raw.shape
            d = random.randint(0, D - self.PATCH[0])
            h = random.randint(0, H - self.PATCH[1])
            w = random.randint(0, W - self.PATCH[2])
            sl = (
                slice(None),
                slice(d, d + self.PATCH[0]),
                slice(h, h + self.PATCH[1]),
                slice(w, w + self.PATCH[2])
            )
            raw, mask = raw[sl], mask[sl]
        return raw, mask


class HipMRIDataset(torch.utils.data.Dataset):
    """
    2D HipMRI dataset class.
    Patient-level splits to prevent data leakage have already been done for us on rangpur.
    """
    def __init__(self, dataset: str, img_dim: tuple[int, int], transform=None):
        # Need to load train, test, and validation sets for 2D
        # For 3D, we need to split it ourselves
        path = os.path.join(PROCESSED_PATH, "2D", dataset)
        self.raw_files = sorted(glob.glob(os.path.join(path, "raw_*.npy")))
        self.mask_files = sorted(glob.glob(os.path.join(path, "mask_*.npy")))

        self.label_values = range(6)    # 0 - Background,
                                        # 1 - Body outline,
                                        # 2 - Bone,
                                        # 3 - Bladder,
                                        # 4 - Rectum,
                                        # 5 - Prostate

        # Separate resize for masks — NEAREST is critical, do not blend labels
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_dim, interpolation=transforms.InterpolationMode.NEAREST),
        ]) if transform is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.raw_files)

    def __getitem__(self, index: int):
        raw_scan = np.load(self.raw_files[index], mmap_mode="r")
        mask_scan = np.load(self.mask_files[index], mmap_mode="r")

        # Better to use from_numpy() here since the masks are assigned an integer in sequential order.
        raw_image = torch.from_numpy(raw_scan.copy())
        mask_image = torch.from_numpy(mask_scan.copy())

        raw_image = raw_image.unsqueeze(0)
        mask_image = mask_image.float()     # Can leave as float32 for now. Consider mixed precision later.

        return raw_image, mask_image


# ======================================================================================================================
# =                                                 Dataloader Helpers                                                 =
# ======================================================================================================================
def get_dataloader_2d(dataset: str) -> DataLoader:
    mri_dataset = HipMRIDataset(dataset, IMAGE_SIZE)
    shuffle = dataset in ["train", "test"]
    return DataLoader(mri_dataset, batch_size=BATCH_SIZE_2D[dataset], shuffle=shuffle)


def get_dataloader_3d(dataset: str) -> DataLoader:
    mri_dataset = HipMRI3DDataset(dataset)
    shuffle = dataset in ["train", "test"]
    is_train = (dataset == "train")
    return DataLoader(
        mri_dataset,
        batch_size          = BATCH_SIZE_3D[dataset],
        shuffle             = shuffle,
        num_workers         = 1 if is_train else 0,
        persistent_workers  = is_train,
    )


# ======================================================================================================================
# =                                                Main (Sanity checks)                                                =
# ======================================================================================================================
if __name__ == "__main__":
    # Sanity checks
    for k, v in SPLIT.items():
        print(f"Patients in {k} split: {sum(Counter(v).values())}")

    verify_splits("3D")
    train_dataset = HipMRI3DDataset("validate")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    print(f"Total batches in validation loader: {len(train_dataloader)}")
    for i, x in enumerate(train_dataloader):
        print(f"Batch {i} - Raw shape: {x[0].shape} | Mask shape: {x[1].shape}")
        break