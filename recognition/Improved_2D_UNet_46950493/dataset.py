from pathlib import Path
import glob, os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


"""
    Load dataset from hard coded file (either RANGPUR HPC cluster files or my own local files)
    Sort data into test_loader, val_loader, test_loader with labels matched to them within a HipMRIDataset - a class of inheritance torch Dataset
    The sorted data is of type DataLoaders -  includes batching, shuffling, and multi-process data loading. 
    batch_size = 16, only training data is shuffled

    Last Confirmed data size and num of clases in labels:
    Label values: [0, 1, 2, 3, 4, 5]
    imgs -> torch.Size([16, 1, 256, 128]), labels -> torch.Size([16, 6, 256, 128])
    (B, C, H, W)   -images have 1 x channel since they are of greyscale. 

"""


class HipMRIDataset(Dataset):
    """
    Store the image paths and corresponding label paths along with the (max) number of classes within the given labels 
    """
    def __init__(self, image_paths, label_paths, num_classes):
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.num_classes = num_classes

    """Out: how many slices are in this dataset. The DataLoader uses this to know when an epoch ends."""
    def __len__(self):
        return len(self.image_paths)

    """
    In: Given an index:
    Return: the pair of (image, label) for slice at index
            image is shape (1, H, W)
            label is shape (C, H, W) -> C = num_classes, after one-hot encoding (every segmentation class has its own channel)
    """
    def __getitem__(self, idx):
        img = load_slice(self.image_paths[idx])
        #normalize slice
        img = (img - img.mean()) / (img.std() + 1e-8)   # z-score per slice
        #add third dimension to match label shape
        img = torch.from_numpy(img).unsqueeze(0)        # (1, H, W)

        #read label as 2D float array, round values to nearest whole numbers then convert numbers to integers
        lab = np.rint(load_slice(self.label_paths[idx])).astype(np.int64)  
        lab = torch.from_numpy(lab)    #convert to tensor -> (H, W)
        #convert labels to one-hot encoded form
        onehot = torch.nn.functional.one_hot(lab, self.num_classes)  # (H, W, C)
        onehot = onehot.permute(2, 0, 1).float()  #changes order from (H,W,C) to (C, H, W) for pytorch compatability 
        
        return img, onehot


def load_slice(path):
    """Load one NIfTI file as a 2D float32 array (H, W).
       Return: The single image as an array
    """
    arr = nib.load(str(path)).get_fdata(caching="unchanged")
    #if 3D only keep first slice
    if arr.ndim == 3:
        arr = arr[:, :, 0]       
    return arr.astype(np.float32)


def get_path_pairs(img_folder, lab_folder):
    imgs = sorted(Path(img_folder).glob("*.nii.gz"))
    labs = sorted(Path(lab_folder).glob("*.nii.gz"))

    assert len(imgs) == len(labs), f"{len(imgs)} images but {len(labs)} labels"

    # Check every image name matches its label name: case_035_... vs seg_035_...
    for i, l in zip(imgs, labs):
        if i.name.replace("case_", "", 1) != l.name.replace("seg_", "", 1):
            raise ValueError(f"Mismatch: {i} vs {l}")

    return imgs, labs

NUM_CLASSES = 6          # label ids 0-5, confirmed earlier
PROSTATE_CLASS = 5       #predicted from personal tests

def pad_collate(batch, multiple=16):
    """Stack slices into a batch, zero-padding the smaller ones to the largest size in the batch."""

    #find max size - whether a H = 256 and W = 144 slice exists in the batch
    H_max = max(img.shape[1] for img, _ in batch)
    W_max = max(img.shape[2] for img, _ in batch)
    #rounding H&W so that it can sustain being halved four times in stride 2 Conv
    H_max, W_max = -(-H_max // multiple) * multiple, -(-W_max // multiple) * multiple    

    #find padding needed for each slice
    imgs, labs = [], []
    for img, lab in batch:
        extra_h, extra_w = H_max - img.shape[1], W_max - img.shape[2]
        pad = (0, extra_w, 0, extra_h)

        #assign padding to class 0 (as background) for labels
        outside = F.pad(torch.zeros_like(lab[:1]), pad, value=1.0) #distinguish padding region (as 1's)
        lab = F.pad(lab, pad) #Add padding to all labels
        lab[:1] += outside  #add 1's to background class for the padding region 
        #(0's everywhere else will keep original values the same)

        #add padding to labels and images
        imgs.append(F.pad(img, pad)) 
        labs.append(lab)
        
    return torch.stack(imgs), torch.stack(labs)


def get_dataloaders(root, batch_size=16, num_workers=0, verbose=True):
    """Build the train / validation / test DataLoaders.
    root: the folder that contains the six data folders."""
    root = Path(root)

    train_imgs, train_labs = get_path_pairs(root / "keras_slices_train", root / "keras_slices_seg_train")
    val_imgs,   val_labs   = get_path_pairs(root / "keras_slices_validate", root / "keras_slices_seg_validate")
    test_imgs,  test_labs  = get_path_pairs(root / "keras_slices_test", root / "keras_slices_seg_test")

    if verbose:
        print(f"slices: train {len(train_imgs)}, val {len(val_imgs)}, test {len(test_imgs)}")
        for i, l in list(zip(train_imgs, train_labs))[:3]:
            print(i.name, "<->", l.name)

    train_ds = HipMRIDataset(train_imgs, train_labs, NUM_CLASSES)
    val_ds   = HipMRIDataset(val_imgs,   val_labs,   NUM_CLASSES)
    test_ds  = HipMRIDataset(test_imgs,  test_labs,  NUM_CLASSES)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, collate_fn=pad_collate)
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=pad_collate)
    test_loader  = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=pad_collate)
    
    return train_loader, val_loader, test_loader



# root = Path("/Users/alyxgoller/data/HipMRI_data")   

# train_imgs, train_labs = get_path_pairs(root / "train_imgs", root / "train_seg_labels")
# val_imgs,   val_labs   = get_path_pairs(root / "validate_imgs",   root / "validate_seg_labels")
# test_imgs,  test_labs  = get_path_pairs(root / "test_imgs",  root / "test_seg_labels")

# # Sanity check: images and labels must pair up
# for imgs, labs in [(train_imgs, train_labs), (val_imgs, val_labs), (test_imgs, test_labs)]:
#     assert len(imgs) == len(labs), (len(imgs), len(labs))
#     for i, l in list(zip(imgs, labs))[:3]:
#         print(i.name, "<->", l.name)


# # --- Configuring data loaders ---

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if not torch.cuda.is_available():
#     print("Warning CUDA not Found. Using CPU")


# # Obtain number of classes form labels: scan every label file once, across all three sets
# NUM_CLASSES = 6        # label ids 0-5, confirmed earlier
# PROSTATE_CLASS = 1     # PLACEHOLDER: set to the real prostate id


# # set up class objects for each model section
# train_ds = HipMRIDataset(train_imgs, train_labs, NUM_CLASSES)
# val_ds   = HipMRIDataset(val_imgs,   val_labs,   NUM_CLASSES)
# test_ds  = HipMRIDataset(test_imgs,  test_labs,  NUM_CLASSES)

# # create dataloaders for each model section
# train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
# val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)
# test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False)






