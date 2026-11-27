import math
import random
from collections import Counter
from pathlib import Path
import nibabel as nib
import numpy as np
from typing import NamedTuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class Sample(NamedTuple):
    patient_id: int
    week: int
    slice: int
    image_path: Path
    mask_path: Path

def _get_pairs(image_dir: Path, mask_dir: Path):
    """
    Store a pair of image, mask and other variables as a Sample(Named Tuple)
    :param image_dir: Path to the directory containing images
    :param mask_dir: Path to the directory containing segmentation masks
    :return: A sorted list of Sample objects
    """
    samples = []
    for image_path in image_dir.glob("*.nii.gz"):
        _, patient_id, _, week, _, slice_index = image_path.name.split(".")[0].split("_")
        mask_path = mask_dir / image_path.name.replace("case_", "seg_", 1)

        if not mask_path.exists():
            raise FileNotFoundError(f"No mask for {image_path.name}")
        samples.append(Sample(int(patient_id), int(week),
                              int(slice_index), image_path, mask_path))

    sorted_samples = sorted(samples, key=lambda s: (s.patient_id, s.week, s.slice))
    return sorted_samples

def check_leakage(train_samples:list[NamedTuple], test_samples:list[NamedTuple], val_samples:list[NamedTuple]):
    train_ids = {s.patient_id for s in train_samples}
    val_ids = {s.patient_id for s in val_samples}
    test_ids = {s.patient_id for s in test_samples}

    overlaps = {
        "train/val": train_ids & val_ids,
        "train/test": train_ids & test_ids,
        "val/test": val_ids & test_ids,
    }
    leaked = False
    for name, ids in overlaps.items():
        if ids:
            leaked = True
            print(f"LEAKAGE {name}: {len(ids)} shared patients, e.g. {sorted(ids)[:10]}")

    if not leaked:
        print("No patient overlap between train, val and test.")
    return leaked

def categorise_prostate(sorted_samples:list[NamedTuple]):

    count_labels = {}
    prostate_labeled = set()
    prostate_unlabeled = set()
    for sample in sorted_samples:
        mask_path = sample.mask_path
        nifti_mask = nib.load(mask_path).get_fdata(caching='unchanged')
        count_labels[f"{sample.patient_id}-{sample.week}-{sample.slice}"] = len(np.unique(nifti_mask))
        if 5 in np.unique(nifti_mask):
            prostate_labeled.add(sample)
        else:
            prostate_unlabeled.add(sample)

    print(f"There are {len(prostate_labeled)} prostate images")
    print(f"There are {len(prostate_unlabeled)} non-prostate images")

    return None

def get_label_ids(sorted_samples):

    all_label_ids = set()


    for sample in sorted_samples:
        mask_path = sample.mask_path
        nifti_mask = nib.load(mask_path).get_fdata(caching = 'unchanged')
        if nifti_mask.ndim == 3:
            nifti_mask = nifti_mask[:, :, 0]
        #nifti_mask = nifti_mask.astype(np.uint8)

        for id in np.unique(nifti_mask):
            all_label_ids.add(int(id))

    label_ids = sorted(list(all_label_ids))
    id_to_channels =  {}
    for channel, id in enumerate(label_ids):
        id_to_channels[id] = channel
    num_classes = len(label_ids)
    print("Labels: ", label_ids)

    return id_to_channels, num_classes

class LoadDataset(Dataset):
    """
    Loads 2D Nifti image slices and segmentation masks from HipMRI 2D Dataset


    """
    def __init__(self, samples, num_classes, id_channels, size=(256, 128), normalized=True):

        self.samples = samples
        self.num_classes = num_classes
        self.id_channels = id_channels
        self.size = size
        self.normalized = normalized

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.samples[idx].image_path
        mask_path = self.samples[idx].mask_path

        #loading image and mask slices
        loaded_np_image = nib.load(image_path).get_fdata(caching='unchanged')
        loaded_np_mask = nib.load(mask_path).get_fdata(caching='unchanged')

        if loaded_np_image.ndim == 3:
            loaded_np_image = loaded_np_image[:, :, 0]

        if loaded_np_mask.ndim == 3:
            loaded_np_mask = loaded_np_mask[:, :, 0]

        loaded_np_image = loaded_np_image.astype("float32")
        loaded_np_mask = loaded_np_mask.astype("uint8")

        #z-score normalisation
        if self.normalized:
            mean = loaded_np_image.mean()
            std = loaded_np_image.std()
            loaded_np_image = (loaded_np_image - mean) / (std + 1e-8)

        # resize and fix dimensions
        loaded_np_image = fit_to_size(loaded_np_image, self.size)
        loaded_np_mask = fit_to_size(loaded_np_mask, self.size)

        h, w = loaded_np_mask.shape

        #One hot encoding conversion
        mask_encoded = np.zeros((self.num_classes, h, w), dtype="float32")
        for label_id, channel in self.id_channels.items():
            mask_encoded[channel] = (loaded_np_mask == label_id).astype("float32")

        loaded_torch_image = torch.from_numpy(loaded_np_image).unsqueeze(0)
        loaded_torch_mask = torch.from_numpy(mask_encoded)


        return loaded_torch_image, loaded_torch_mask

def fit_to_size(nib_arr, size):
    out_h, out_w = size
    nib_h, nib_w = nib_arr.shape

    top, left = max((nib_h - out_h) // 2, 0), max((nib_w - out_w) // 2, 0)
    nib_arr = nib_arr[top:top + min(nib_h, out_h), left:left + min(nib_w, out_w)]

    return nib_arr

def get_dataloaders(root_path = Path("./Data/keras_slices_data"), num_workers = 1, batch_size = 8, out_size = (256, 128), full=True):

    root_path = Path(root_path)
    train_img_path = root_path / "keras_slices_train"
    val_img_path = root_path / "keras_slices_validate"
    test_img_path = root_path / "keras_slices_test"
    train_mask_path = root_path / "keras_slices_seg_train"
    val_mask_path = root_path / "keras_slices_seg_validate"
    test_mask_path = root_path / "keras_slices_seg_test"

    train_samples = _get_pairs(train_img_path, train_mask_path)
    val_samples = _get_pairs(val_img_path, val_mask_path)
    test_samples = _get_pairs(test_img_path, test_mask_path)

    print("Train/Val/Test Splits: ", len(train_samples), len(val_samples), len(test_samples))

    check_leakage(train_samples, val_samples, test_samples)

    categorise_prostate(train_samples)
    categorise_prostate(val_samples)
    categorise_prostate(test_samples)

    print ("Training dataset size: ", len(train_samples))

    id_channels, num_classes = get_label_ids(train_samples)
    print("Number of labels: ", num_classes)

    train_dataset = LoadDataset(train_samples, num_classes, id_channels, size=(256, 128))
    val_dataset = LoadDataset(val_samples, num_classes, id_channels, size=(256, 128))
    test_dataset = LoadDataset(test_samples, num_classes, id_channels, size=(256, 128))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader, id_channels, num_classes

#get_dataloaders()

