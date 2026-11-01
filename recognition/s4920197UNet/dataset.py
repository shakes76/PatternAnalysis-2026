
import numpy as np
import nibabel as nib
from collections import defaultdict 
from pathlib import Path
from torch.utils.data import Dataset
import torch
NUM_CLASSES = 6

class TestData(Dataset):
    def __init__(self, folder):
        self.test = load(folder, "test")

    def __len__(self):
        return len(self.test)
    
    def __getitem__(self, idx):
        return get_data(self.test[idx])

class TrainData(Dataset):
    def __init__(self, folder):
        self.train = load(folder, "train")

    def __len__(self):
        return len(self.train)
    
    def __getitem__(self, idx):
        return get_data(self.train[idx])

class ValidationData(Dataset):
    def __init__(self, folder):
        self.validate = load(folder, "validate")

    def __len__(self):
        return len(self.validate)
    
    def __getitem__(self, idx):
        return get_data(self.validate[idx])

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros((NUM_CLASSES,) + arr.shape, dtype=dtype)
    for c in channels:
        c = int(c)
        res[c, arr == c] = 1
    return res

def load(folder, dir):
    data = []
    seg = Path(f"{folder}/keras_slices_seg_{dir}")
    img = Path(f"{folder}/keras_slices_{dir}")
    masks = sorted([f for f in seg.iterdir() if f.is_file()])
    images = sorted([f for f in img.iterdir() if f.is_file()])
    for mask, image in zip(masks, images):
        mparts = list(filter(lambda x: x.isdigit(), mask.name.split("_")))
        imparts = list(filter(lambda x: x.isdigit(), image.name.split("_")))
        if not mparts == imparts:
            print(f"{mparts}, {imparts}")
            exit("INVALID DATASET, FILES ARENT IN SEQUENCE")
        if not ("019" in mparts and "1" in mparts):
            data.append((mask, image))
    return data

def get_data(pair):
    mpath, impath = pair
    mask = nib.load(mpath).get_fdata()
    image = nib.load(impath).get_fdata()
    if len(image.shape) == 3:
        image = image[:,:,0]
    if len(mask.shape) == 3:
        mask = mask[:,:,0]

    if image.shape != mask.shape:
        exit("DIMENSIONS DONT MATCH")

    if image.shape != (256, 128):
        print(f"DELETE {mpath} or {impath}")

    mask = to_channels(mask)
    image = (image - image.mean()) / (image.std() + 1e-8)
    mask = torch.from_numpy(mask).long()
    image = torch.from_numpy(image).float().unsqueeze(0)

    return mask, image