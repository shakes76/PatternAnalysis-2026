"""
Containing the data loader for loading and preprocessing your data, including leakage-
free train/validation/test splitting functions and data augmentations.
"""
"""
70/15/15 split
256 * 256 images (might go to 512*512 later)
LAB colourspace

might need to run: 
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
for WSL
"""
import os
import random
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from skimage.color import rgb2lab
from skimage.color import lab2rgb

from sklearn.model_selection import train_test_split

# Use BSD500 dataset 

# 1. Download dataset into dataset/ if not already there
def download_bsd500(destination: str = "recognition/MambaIRv2_s48448284/dataset") -> Path:
    import tarfile
    import urllib.request

    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)

    images_dir = dest / "BSR" / "BSDS500" / "data" / "images"
    if images_dir.exists():
        print(f"BSD500 already downloaded")
        return images_dir

    url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
    tgz_path = dest / "BSR_bsds500.tgz"

    print("Downloading BSD500 ...")
    urllib.request.urlretrieve(url, tgz_path)

    print("Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(dest)

    # remove archive after extraction
    tgz_path.unlink()  
    print(f"Done. Images at {images_dir}")
    return images_dir

# put train/val/test back together
def collect_image_paths(images_dir: Path) -> list[Path]:
    all_paths = []
    for subdir in ["train", "val", "test"]:
        folder = images_dir / subdir
        if folder.exists():
            all_paths.extend(sorted(folder.glob("*.jpg")))
    print(f"Found {len(all_paths)} images total")
    return all_paths

def download_coco_subset(dest_dir, num_images=2000, seed=42):
    import urllib.request
    import json
    import random

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Download the annotations file to get image URLs
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_path = dest / "annotations.zip"

    if not (dest / "instances_val2017.json").exists():
        print("Downloading COCO annotations...")
        urllib.request.urlretrieve(ann_url, ann_path)
        import zipfile
        with zipfile.ZipFile(ann_path) as z:
            z.extract("annotations/instances_val2017.json", dest)
        ann_path.unlink()  # delete zip
        (dest / "annotations" / "instances_val2017.json").rename(dest / "instances_val2017.json")

    with open(dest / "instances_val2017.json") as f:
        coco = json.load(f)

    random.seed(seed)
    images = random.sample(coco["images"], num_images)

    print(f"Downloading {num_images} COCO images to {dest}...")
    for i, img in enumerate(images):
        url = f"http://images.cocodataset.org/val2017/{img['file_name']}"
        urllib.request.urlretrieve(url, dest / img["file_name"])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{num_images}")

    print(f"Done. {num_images} images saved to {dest}")

# 2. Split into training / validation / testing
def split(paths: list[Path], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> tuple[list[Path], list[Path, list[Path]]]:
    train_val, test = train_test_split(paths, test_size = 1 - train_ratio - val_ratio, random_state=seed)
    val_size = val_ratio/ (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=val_size, random_state=seed)

    return train, val, test

# 3. Working with LAB colours
# ref: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_rgb_to_gray.html
# the con
def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an HxWx3 uint8 RGB numpy array to Lab.
    Returns float32 array with:
        L in [0, 100]
        a in [-128, 127]
        b in [-128, 127]
    """
    return rgb2lab(img_rgb).astype(np.float32)


def lab_to_rgb(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """
    Reconstruct an RGB image from L (HxW) and ab (HxWx2) arrays.
    L should be in [0,100], ab in [-128, 127].
    Returns uint8 RGB array clipped to [0, 255].
    """
    lab = np.concatenate([L[..., np.newaxis], ab], axis=-1)
    rgb = lab2rgb(lab)  # returns float [0,1]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def normalise_L(L: np.ndarray) -> np.ndarray:
    """Normalise L channel from [0, 100] → [-1, 1] for model input."""
    return (L / 50.0) - 1.0


def normalise_ab(ab: np.ndarray) -> np.ndarray:
    """Normalise ab channels from [-128, 127] → [-1, 1] for model output."""
    return ab / 128.0


def denormalise_L(L_norm: np.ndarray) -> np.ndarray:
    """Reverse normalise_L: [-1, 1] → [0, 100]."""
    return (L_norm + 1.0) * 50.0


def denormalise_ab(ab_norm: np.ndarray) -> np.ndarray:
    """Reverse normalise_ab: [-1, 1] → [-128, 127]."""
    return ab_norm * 128.0

# 3. Data
class ColouriseDataset(Dataset):

    IMG_SIZE = 256

    def __init__(self, paths: list[Path], augment: bool = False):
        self.paths = paths
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def _load_and_resize(self, path: Path) -> np.ndarray:
        """Load image, ensure RGB, resize to IMG_SIZE×IMG_SIZE."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.IMG_SIZE, self.IMG_SIZE))
        return np.array(img, dtype=np.uint8)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """
        Training augmentations applied in RGB space before Lab conversion.
        Only spatial augmentations are safe here — colour jitter would corrupt
        the Lab target.
        """
        img_pil = Image.fromarray(img)
        # Random horizontal flip
        if random.random() > 0.5:
            img_pil = TF.hflip(img_pil)
        # Random crop → resize back to IMG_SIZE (adds scale variation)
        crop_size = random.randint(int(self.IMG_SIZE * 0.85), self.IMG_SIZE)
        i, j, h, w = transforms.RandomCrop.get_params(
            img_pil, output_size=(crop_size, crop_size)
        )
        img_pil = TF.crop(img_pil, i, j, h, w)
        img_pil = img_pil.resize((self.IMG_SIZE, self.IMG_SIZE), Image.BICUBIC)
        return np.array(img_pil, dtype=np.uint8)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_rgb = self._load_and_resize(self.paths[idx])

        if self.augment:
            img_rgb = self._augment(img_rgb)

        lab = rgb_to_lab(img_rgb)            # HxWx3, float32
        L = lab[:, :, 0]                     # HxW
        ab = lab[:, :, 1:]                   # HxWx2

        L_norm = normalise_L(L)              # [-1, 1]
        ab_norm = normalise_ab(ab)           # [-1, 1]

        # Convert to tensors: (C, H, W)
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0)   # (1, 256, 256)
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1)  # (2, 256, 256)

        return L_tensor, ab_tensor

# 4. Make Data loaders
def get_dataloaders(
    dataset_dir: str = "recognition/MambaIRv2_s48448284/dataset",
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Downloads BSD500 (if needed), splits it, and returns
    (train_loader, val_loader, test_loader).
    """
    # images_dir = download_bsd500(dataset_dir)
    # all_paths = collect_image_paths(images_dir)
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    all_paths = sorted(
        str(p) for p in Path(dataset_dir).rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    
    train_paths, val_paths, test_paths = split(all_paths, seed=seed)

    train_ds = ColouriseDataset(train_paths, augment=True)
    val_ds = ColouriseDataset(val_paths, augment=False)
    test_ds = ColouriseDataset(test_paths, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # check if it works
    download_coco_subset(dest_dir="recognition/MambaIRv2_s48448284/dataset/coco")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4, num_workers=0)

    L_batch, ab_batch = next(iter(train_loader))
    print(f"L  shape: {L_batch.shape}  range: [{L_batch.min():.2f}, {L_batch.max():.2f}]")
    print(f"ab shape: {ab_batch.shape}  range: [{ab_batch.min():.2f}, {ab_batch.max():.2f}]")