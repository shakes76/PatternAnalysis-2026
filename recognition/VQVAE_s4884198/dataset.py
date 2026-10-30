import glob, numpy as np, nibabel as nib, torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import os 
import re
import matplotlib.pyplot as plt 


IMG_DIRS = {
    "train": "keras_slices_train",
    "validate": "keras_slices_validate",
    "test": "keras_slices_test"
}

SEG_DIRS = {
    "train": "keras_slices_seg_train",
    "validate": "keras_slices_seg_validate",
    "test": "keras_slices_seg_test"
}

TARGET_SHAPE = (256, 128)


def load_slices(paths, is_mask=False, target_shape=TARGET_SHAPE):
    arrs = []
    for p in tqdm(paths):
        a = nib.load(p).get_fdata(caching="unchanged") # loops through files reading them into np array of pixel values
        if a.ndim == 3:
            a = a[:, :, 0]                      # drop extra dimemsions in case images have a third dimension
        a = a.astype(np.float32) #pytorch default tensor type
        if a.shape != target_shape:
            t = torch.from_numpy(a)[None, None]
            if is_mask:
                t = F.interpolate(t, size=target_shape, mode="nearest")
            else:
                t = F.interpolate(t, size=target_shape, mode="bilinear", align_corners=False)
            a = t[0, 0].numpy()        
        
        if is_mask:
            a = (a > 0).astype(np.float32)      # binary prostate maskc (prostate or not)
        else:
            a = (a - a.min()) / (a.max() - a.min() + 1e-8)   # [0, 1] (small value stops a divide by 0 err)
        arrs.append(a)
    return np.stack(arrs)[:, None]              # (N, 1, H, W)

def pad_multiple(x, multiple=8): #I dont think this is needed since the image sizes already divide by 8 but good for safety 
    height, width = x.shape[-2:] # get current dimensions 
    pad_height, pad_width = (-height) % multiple, (-width) % multiple  #padding calculation 
    return F.pad(x, (0, pad_width, 0, pad_height)) if (pad_height or pad_width) else x 

class DatasetSlice(Dataset): #needed by the DataLoader to know how to build batches 
    def __init__(self, images, masks=None):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, key):
        if self.masks is None:
            return self.images[key]
        return self.images[key], self.masks[key]

def _find(root, folder):
    paths = sorted(glob.glob(os.path.join(root, folder, "*nii*"))) #finds the nifti files in folder and sorts them so images and masks still in matching order
    if not paths:
        raise FileNotFoundError(f"no nifti files in folder {folder}")
    return paths 

def check_no_overlap(root, pattern=r"(?:case|seg)_(\d+)"):
    ids = {}
    for split, folder in IMG_DIRS.items():
        names = [os.path.basename(k) for k in _find(root, folder)] #all names in split 
        ids[split] = {m.group(1) for n in names if (m := re.search(pattern, n))} #ids from the names 
        print(f"{split} {len(names)} slices, {len(ids[split])} patient ids") 
    for a, b in [("train", "validate"), ("train", "test"), ("validate", "test")]: #overlap calculation between each different split
        overlap = ids[a] & ids[b]
        print(f"overlap {a}/{b}: {len(overlap)}")
    return ids 

def get_dataloaders(root, batch_size=32, with_masks=False, subset=None, multiple=8, num_workers=0):
    loaders = []
    for split in ["train", "validate", "test"]:
        #produce suitable images 
        img_paths = _find(root, IMG_DIRS[split])[:subset]
        images = pad_multiple(torch.from_numpy(load_slices(img_paths)), multiple)

        #also ptoduce suitable masks if with_masks is set true
        masks = None 
        if with_masks:
            seg_paths = _find(root, SEG_DIRS[split])[:subset]
            assert len(seg_paths) == len(img_paths), f"{split} mismatch between images and masks" #ensure same num images and masks 
            masks = pad_multiple(torch.from_numpy(load_slices(seg_paths, is_mask=True)), multiple)

        print(f"{split}: {tuple(images.shape)}, range [{images.min():.2f}, {images.max():.2f}]")

        #create the data loader for each split 
        loaders.append(
            DataLoader(
                DatasetSlice(images, masks),
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers
            )
        )
    return tuple(loaders)

def show_image(image, title=None):
    image = image.squeeze() # turn into just h,w
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.show()
    return

if __name__ == "__main__":
    print("testing")
    root ="C:/Users/danfu/PatternAnalysis-2026/data/"
    check_no_overlap(root)
    train_loader, _, _ = get_dataloaders(root, subset=64)
    print("batch shape:", next(iter(train_loader)).shape)
    x= next(iter(train_loader))
    show_image(x[0].numpy())