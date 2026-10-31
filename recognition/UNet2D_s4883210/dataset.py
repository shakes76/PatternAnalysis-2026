import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize as tv_resize
from torchvision.transforms import InterpolationMode

BASE = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

SPLITS = {
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "val": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}

NUM_CLASSES = 6
IMG_SIZE = (256, 128)

CLASS_COLORS = np.array([
    [0.0, 0.0, 0.0], # 0 background, black
    [0.5, 0.5, 0.5], # 1 body, grey
    [1.0, 1.0, 1.0], # 2 bone, white
    [1.0, 0.0, 0.0], # 3 bladder, red
    [0.0, 1.0, 0.0], # 4 rectum, green
    [0.0, 0.0, 1.0],  # 5 prostate, blue
])

def mask_to_rgb(mask):
    return CLASS_COLORS[mask]

def to_channels(arr: np.ndarray, num_classes: int = NUM_CLASSES,
                dtype=np.uint8) -> np.ndarray:
    """Converts label image into a one-hot tensor"""
    arr = np.round(arr).astype(np.int64)
    res = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    for c in range(num_classes):
        res[..., c][arr == c] = 1
    return res

# load medical image functions
def load_data_2D (imageNames, normImage=False, categorical=False, dtype =np.float32, getAffines=False, early_stop=False):
    """
    Load medical image data from names , cases list provided into a list for
    each. This function pre - allocates 4D arrays for conv2d to avoid excessive
    memory usage.
    normImage : bool ( normalise the image 0.0 -1.0)
    early_stop : Stop loading pre - maturely , leaves arrays mostly empty , for
    quick loading and testing scripts .
    """
    affines = []
    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape ) == 3:
        first_case = first_case [: ,: ,0] # sometimes extra dims , remove
    if categorical:
        first_case = to_channels(first_case , dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np. zeros (( num, rows, cols, channels), dtype=dtype)
    else :
        rows, cols = first_case.shape
        images = np. zeros ((num, rows, cols), dtype=dtype )

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage . affine
        if len(inImage.shape) == 3:
            inImage = inImage[: ,: ,0] # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage :
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical :
            inImage = to_channels(inImage, dtype=dtype)
            images[i ,: ,: ,:] = inImage
        else:
            images[i ,: ,:] = inImage
            affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines :
        return images , affines
    else :
        return images

def _suffix(fname: str) -> str:
    i = next(k for k, ch in enumerate(fname) if ch.isdigit())
    return fname[i:]

def pair_paths(image_dir, label_dir):
    """Match each MRI image with its segmentation label"""
    labels = {_suffix(os.path.basename(p)): p 
              for p in glob.glob(os.path.join(label_dir, "*.nii.gz"))}
    pairs = []
    for ip in sorted(glob.glob(os.path.join(image_dir, "*.nii.gz"))):
        key = _suffix(os.path.basename(ip))
        if key in labels:
            pairs.append((ip, labels[key]))
    
    return pairs

class HipMRIDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, lp = self.pairs[idx]

        # load 1 image and 1 label at a time
        img = load_data_2D([ip], normImage=True, categorical=False)[0]
        lbl = load_data_2D([lp], normImage=False, categorical=True,
                           dtype=np.uint8)[0] 

        img = torch.from_numpy(img).unsqueeze(0)
        lbl = torch.from_numpy(lbl).permute(2, 0, 1).float()

        # resize every slice to (256, 128) so the dataloader can stack them
        if img.shape[-2:] != (256, 128):
            img = tv_resize(img, [256, 128], antialias=True)
            lbl = tv_resize(lbl, [256, 128], interpolation=InterpolationMode.NEAREST)

        return img, lbl


def build_dataloaders(batch_size=8, num_workers=4):
    loaders = {}
    for name, (img_d, lbl_d) in SPLITS.items():
        pairs = pair_paths(os.path.join(BASE, img_d), os.path.join(BASE, lbl_d))

        loaders[name] = DataLoader(
            HipMRIDataset(pairs),
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

        print(f"{name}: {len(pairs)} paired slices")

    return loaders


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    loaders = build_dataloaders(batch_size=4, num_workers=2)
    imgs, lbls = next(iter(loaders["train"]))
    print("image batch:", imgs.shape, imgs.dtype)
    print("label batch:", lbls.shape, lbls.dtype,
          "sum per sample:", lbls[0].sum(dim=(1,2)))

    img = imgs[0, 0].numpy()
    lbl = lbls[0].argmax(0).numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].imshow(img, cmap="gray"); ax[0].set_title("image (normalised)")
    ax[1].imshow(mask_to_rgb(lbl))
    ax[1].set_title("label")

    plt.tight_layout()
    out = os.path.expanduser("~/test_overlay.png")
    plt.savefig(out, dpi=100)
    print(f"saved {out}")