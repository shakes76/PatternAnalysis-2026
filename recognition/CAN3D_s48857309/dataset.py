"""This module allows loading of the HipMRI database.

References:
    load_data_3D: taken from the assignment sheet
"""
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


# load medical image functions
def load_data_3D(
    imageNames,
    normImage=False,
    dtype=np.float32,
    dimension=None
):
    """
    Load medical image data from names, cases list provided into a list for
    each .

    This function pre - allocates 5D arrays for conv3d to avoid excessive
    memory usage .

    normImage : bool ( normalise the image 0.0-1.0)
    dtype : Type of the data . If dtype =np.uint8 , it is assumed that the data
    is labels
    dimension : use None to infer, else specify (rows, cols, depth)
    """

    # get fixed size
    num = len(imageNames)
    if dimension is None:
        niftiImage = nib.load(imageNames[0])
        first_case = niftiImage.get_fdata(caching="unchanged")
        if len(first_case.shape) == 4:
            first_case = first_case[:, :, :, 0]  # sometimes extra dims , remove
        rows, cols, depth = first_case.shape
    else:
        rows, cols, depth = dimension
    images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")  # read disk only

        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        
        images[i, : inImage.shape[0], : inImage.shape[1], : inImage.shape[2]] = (
                inImage
            )

    return images

def load_data_2D(imageNames,  normImage = False, dtype=np.float32):
    """
    Load medical image data from names, cases list provided into a list for
    each .

    This function pre - allocates 4D arrays for conv2d to avoid excessive
    memory usage .

    normImage : bool ( normalise the image 0.0 -1.0)
    """
    num = len( imageNames )
    niftiImage = nib.load(imageNames[0])
    first_case = niftiImage.get_fdata(caching="unchanged")
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims , remove
    rows, cols = first_case.shape
    images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")  # read disk only

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage[:rows, :cols] 
        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        
        images[i, : inImage.shape[0], : inImage.shape[1]] = inImage

    return images

class HipMRI3D(Dataset):
    def __init__(self, images_path='/home/groups/comp3710/HipMRI_Study_open/semantic_MRs', labels_path='/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only', early_stop=0):
        self.image_dir = Path(images_path)
        self.labels_dir = Path(labels_path)
        self.image_paths = list(self.image_dir.glob('*.nii.gz'))
        self.labels_paths = [(self.labels_dir / "{}_SEMANTIC".format(image_path.name[0:10])).with_suffix(".nii.gz") for image_path in self.image_paths]
        self.patient_labels = [image.name[0:4] for image in self.image_paths]
        if early_stop > 0:
            self.image_paths = self.image_paths[:early_stop]
            self.labels_paths = self.labels_paths[:early_stop]
            self.patient_labels = self.patient_labels[:early_stop]

        self.images = load_data_3D(self.image_paths, True, np.float32)
        self.labels = load_data_3D(self.labels_paths, False, np.uint8)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __getitems__(self, idxs):
        return self.images[idxs], self.labels[idxs]

def load_single_hipmri_image(
        sample='B006_Week0', 
        images_path='/home/groups/comp3710/HipMRI_Study_open/semantic_MRs', 
        labels_path='/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only', 
        dimension=(256, 256, 128)):
    image_dir = Path(images_path)
    labels_dir = Path(labels_path)
    image_paths = [(image_dir / "{}_LFOV".format(sample)).with_suffix(".nii.gz")]
    label_paths = [(labels_dir / "{}_SEMANTIC".format(sample)).with_suffix(".nii.gz")]

    images = load_data_3D(image_paths, True, np.float32, dimension=dimension)
    labels = load_data_3D(label_paths, False, np.uint8, dimension=dimension)
    return torch.tensor(images), torch.tensor(labels)


def get_training_test_split(dataset: HipMRI3D, test_size=0.2, val_size=0.1):
    # Split into training/validation and test set
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_val_idx, test_idx = next(gss.split(dataset.images, groups=dataset.patient_labels))

    val_ratio = val_size / (1 - test_size)
    groups_train_val = [dataset.patient_labels[train_val_idx[i]] for i in range(len(train_val_idx))]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(gss.split(train_val_idx, groups=groups_train_val))


    train_set = Subset(dataset, train_val_idx[train_idx])
    val_set = Subset(dataset, train_val_idx[val_idx])
    test_set = Subset(dataset, test_idx)

    return train_set, val_set, test_set

class HipMRI2D(Dataset):
    def __init__(self, images_path='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train', labels_path='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train', early_stop=200):
        self.image_dir = Path(images_path)
        self.labels_dir = Path(labels_path)
        self.image_paths = list(self.image_dir.glob('*.nii.gz'))
        self.labels_paths = [(self.labels_dir / "seg{}".format(image_path.name[4:]))for image_path in self.image_paths]
        if early_stop > 0:
            self.image_paths = self.image_paths[:early_stop]
            self.labels_paths = self.labels_paths[:early_stop]

        self.images = load_data_2D(self.image_paths, True, np.float32)
        self.labels = load_data_2D(self.labels_paths, False, np.uint8)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __getitems__(self, idxs):
        return self.images[idxs], self.labels[idxs]
