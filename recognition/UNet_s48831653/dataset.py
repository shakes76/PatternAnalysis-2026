import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors

import os
import glob

target_shape = (256, 128)

###############################################################################
# 2D data loading #############################################################
###############################################################################

rangpur_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
slices_train_paths = sorted(glob.glob(os.path.join(rangpur_path, "keras_slices_train/*.nii.gz")))
seg_train_paths = sorted(glob.glob(os.path.join(rangpur_path, "keras_slices_seg_train/*.nii.gz")))
assert len(slices_train_paths) == len(seg_train_paths) and len(seg_train_paths) > 0

slices_validate_paths = sorted(glob.glob(os.path.join(rangpur_path, "keras_slices_validate/*.nii.gz")))
seg_validate_paths = sorted(glob.glob(os.path.join(rangpur_path, "keras_slices_seg_validate/*.nii.gz")))
assert len(slices_validate_paths) == len(seg_validate_paths) and len(seg_validate_paths) > 0

slices_test_paths = sorted(glob.glob(os.path.join(rangpur_path, "keras_slices_test/*.nii.gz")))
seg_test_paths = sorted(glob.glob(os.path.join(rangpur_path, "keras_slices_seg_test/*.nii.gz")))
assert len(slices_test_paths) == len(seg_test_paths) and len(seg_test_paths) > 0

SAVE_DIR = "./preprocessed_data"
train_npz_path = os.path.join(SAVE_DIR, "train.npz")
val_npz_path = os.path.join(SAVE_DIR, "val.npz")
test_npz_path = os.path.join(SAVE_DIR, "test.npz")

# Returns ndarray storing one-hot encoded layers for each segmentation category
# load_data_2D(categorical=False) produces [N, H, W]
# load_data_2D(categorical=True) produces [N, H, W, C]
def to_channels(arr: np.ndarray, dtype=np.uint8, num_classes=5) -> np.ndarray:
    # retrieve the sorted unique elements of flattened arr
    # corresponds to the number of segmented categories
    channels = np.unique(arr)

    # res stores (height, width, channels)
    res = np.zeros(arr.shape + (num_classes,), dtype=dtype)

    # store one-hot encoded layers for each category in res
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

def crop_2D(arr: np.ndarray):
    # Crops images to target_shape dimensions
    target_h, target_w = target_shape
    h, w = arr.shape[:2] # images current dimensions

    # Determine cropping indices
    start_h = max(0, (h - target_h) // 2)
    end_h = start_h + target_h

    start_w = max(0, (w - target_w) // 2)
    end_w = start_w + target_w

    if arr.ndim == 2:
        return arr[start_h:end_h, start_w:end_w]
    elif arr.ndim == 3:
        return arr[start_h:end_h, start_w:end_w, :]
    else:
        raise ValueError(f"Unexpected array dimension: {arr.ndim}")

# Load 2D medical images
def load_data_2D(imageNames, normImage=False, categorical=False,
        dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for
    each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory
    usage.

    imageNames: list of nifti file paths
    normImage: bool (normalise the image 0.0-1.0)
    categorical: if true, provided images are segmented and will be passed
    through to_channels
    getAffines: if true, also return a list of nifti affine matrices
    early_stop: Stop loading pre-maturely, leaving arrays mostly empty, for
    quick loading and testing scripts.
    '''
    affines = []
    num = len(imageNames)
    rows, cols = target_shape
    num_classes = 5

    # determine image heigth x width (x channel) dimensions and construct
    # images darray to store
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0] #remove extra dimensions if present
    if categorical:
        images = np.zeros((num, rows, cols, num_classes), dtype=dtype)
    else:
        images = np.zeros((num, rows, cols), dtype=dtype)

    # load each image and store in images darray
    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] #remove extra dimensions if present
        inImage = inImage.astype(dtype)

        if normImage: # normalise
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)

        inImage = crop_2D(inImage)
                                 
        # store in images array
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:,:,:] = inImage
        else:
            images[i,:,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    return images

class HipMRI2D_Dataset(Dataset):
    """Custom PyTorch dataset for 2D HipMRI slices"""
    def __init__(self, images, masks, transform=None):
        """
        images (np.ndarray): [N, H, W] array containing the images
        masks (np.ndarray): [N, H, W, C] array containing the corresponding masks
        transform (callable, optional): transforms image_tensor and mask_tensor
        """
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self): # retrieve size of the dataset
        return len(self.images)
    
    def __getitem__(self, idx): # for data access with data[i]
        image = self.images[idx]
        mask = self.masks[idx]

        # Resize:
        # Image [H, W] -> [1, H, W]
        image = np.expand_dims(image, axis=0)
        #Mask [H, W, C] -> [C, H, W]
        mask = mask.transpose((2, 0, 1))

        image_tensor = tv_tensors.Image(torch.from_numpy(image).to(torch.float32))
        mask_tensor = tv_tensors.Image(torch.from_numpy(mask).to(torch.float32))

        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

def data_preprocess(early_stop=False):
    """
    Load .nii.gz data into ndarrays

    early_stop (Boolean): If True, preprocessing will terminate before
                          completing the dataset
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Load .nii.gz data
    print("Preprocessing training data...")
    image_train = load_data_2D(slices_train_paths, early_stop=early_stop, normImage=True)
    mask_train = load_data_2D(seg_train_paths, early_stop=early_stop, categorical=True)
    np.savez_compressed(train_npz_path, images=image_train, masks=mask_train)

    print("Preprocessing validation data...")
    image_val = load_data_2D(slices_validate_paths, early_stop=early_stop, normImage=True)
    mask_val = load_data_2D(seg_validate_paths, early_stop=early_stop, categorical=True)
    np.savez_compressed(val_npz_path, images=image_val, masks=mask_val)

    print("Preprocessing testing data...")
    image_test = load_data_2D(slices_test_paths, early_stop=early_stop, normImage=True)
    mask_test = load_data_2D(seg_test_paths, early_stop=early_stop, categorical=True)
    np.savez_compressed(test_npz_path, images=image_test, masks=mask_test)
    print("Data preprocessing complete.")

def get_dataloaders(batch_size=16, early_stop=False):
    """
    Load training, validation and testing data, returns a DataLoader for each set

    batch_size (Integer): Specifies the number of images loaded from a return
                          DataLoader's batch
    early_stop (Boolean): If True, .nii.gz data conversion to ndarray is
                          stopped prematurely. Default value is False.
    """
    # Store preprocessed data
    if (not os.path.exists(train_npz_path) or not os.path.exists(val_npz_path)
                    or not os.path.exists(test_npz_path) or early_stop==True):
        data_preprocess(early_stop=early_stop)
    
    # access preprocessed data
    train_data = np.load(train_npz_path)
    val_data   = np.load(val_npz_path)
    test_data  = np.load(test_npz_path)

    image_train, mask_train = train_data["images"], train_data["masks"]
    image_val, mask_val = val_data["images"], val_data["masks"]
    image_test, mask_test = test_data["images"], test_data["masks"]

    mri_transform = v2.Compose([
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=(-10, 10)),
    v2.RandomAffine(degrees=0, scale=(0.95, 1.05)),
    ])
    
    # Create datasets
    train_dataset = HipMRI2D_Dataset(image_train, mask_train, mri_transform)
    val_dataset = HipMRI2D_Dataset(image_val, mask_val)
    test_dataset = HipMRI2D_Dataset(image_test, mask_test)

    # Create dataloaders
    num_workers = 4 
    #todo drop_last=True if number of images is not divisible by batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
