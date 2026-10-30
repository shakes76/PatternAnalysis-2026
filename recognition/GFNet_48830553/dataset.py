import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

DATA_TEST_PATH = "/home/groups/comp3710/ADNI/AD_NC/test/"
DATA_TRAIN_PATH = "/home/groups/comp3710/ADNI/AD_NC/train/"

# Hyperparameters and Configuration
IMAGE_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = 1
SPLIT_RATIO = 0.8
SEED = 42
NORM_MEAN = [0.5]
NORM_STD = [0.5]


class ADNIDataset(Dataset):
    """Custom Dataset for ADNI images."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label


def data_preprocess():
    """
    Defines image transformations and normalisation pipelines.
    Calibrated for 2D brain MRI scans and GFNet Fourier domain requirements:
    - Moderate brightness/contrast jitter (0.15) to preserve tissue contrast boundaries.
    - Removed RandomErasing: rectangular cutouts inject sharp step discontinuities
      that create high-frequency sinc harmonics across the 2D FFT spectrum, causing
      GFNet to overfit in the frequency domain and trigger massive validation loss spikes.
    - Mild random affine and Gaussian blur for spatial regularisation.
    
    Returns:
        tuple: (train_transform, eval_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.04, 0.04),
            scale=(0.96, 1.04),
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    return train_transform, eval_transform


def data_split(train_dir=DATA_TRAIN_PATH, split_ratio=0.8, seed=42):
    """
    Splits images in the training directory into training and validation sets
    at the patient level to prevent data leakage.
    Splits filenames from the right to safely handle patient IDs containing underscores.

    Label mapping:
        AD -> 1
        NC -> 0

    Args:
        train_dir (str): Root directory of training data containing 'AD' and 'NC'.
        split_ratio (float): Ratio of patients to use for training (default: 0.8).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_samples, val_samples) where each is a list of (image_path, label).
    """
    train_samples = []
    val_samples = []

    classes = [("NC", 0), ("AD", 1)]

    for class_name, label in classes:
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        patient_to_images = {}
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpeg", ".jpg")):
                # File naming convention: <patient_id>_<slice_number>.jpeg
                name_without_ext = os.path.splitext(fname)[0]
                patient_id = name_without_ext.rsplit("_", 1)[0]
                img_path = os.path.join(class_dir, fname)
                patient_to_images.setdefault(patient_id, []).append((img_path, label))

        patient_ids = sorted(list(patient_to_images.keys()))
        rng = random.Random(seed)
        rng.shuffle(patient_ids)

        split_idx = int(len(patient_ids) * split_ratio)
        train_patients = set(patient_ids[:split_idx])
        val_patients = set(patient_ids[split_idx:])

        for pid in train_patients:
            train_samples.extend(patient_to_images[pid])

        for pid in val_patients:
            val_samples.extend(patient_to_images[pid])

    return train_samples, val_samples


def _load_test_samples(test_dir=DATA_TEST_PATH):
    """
    Loads all test samples from test directory.
    
    Returns:
        list: List of (image_path, label) tuples.
    """
    test_samples = []
    classes = [("NC", 0), ("AD", 1)]

    for class_name, label in classes:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpeg", ".jpg")):
                img_path = os.path.join(class_dir, fname)
                test_samples.append((img_path, label))

    return test_samples


def data_loaders(train_dir=DATA_TRAIN_PATH, test_dir=DATA_TEST_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, split_ratio=SPLIT_RATIO, seed=SEED, pin_memory=True):
    """
    Creates and returns PyTorch DataLoaders for training, validation, and testing.
    Configured with pin_memory, persistent_workers, and prefetch_factor for accelerated CUDA data transfer.

    Args:
        train_dir (str): Directory containing training 'AD' and 'NC' subdirectories.
        test_dir (str): Directory containing test 'AD' and 'NC' subdirectories.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker threads for DataLoader.
        split_ratio (float): Patient split ratio for training data (default: 0.8).
        seed (int): Seed for patient splitting.
        pin_memory (bool): Whether to pin memory for fast CUDA host-to-device transfers.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_transform, eval_transform = data_preprocess()
    train_samples, val_samples = data_split(train_dir=train_dir, split_ratio=split_ratio, seed=seed)
    test_samples = _load_test_samples(test_dir=test_dir)

    train_dataset = ADNIDataset(train_samples, transform=train_transform)
    val_dataset = ADNIDataset(val_samples, transform=eval_transform)
    test_dataset = ADNIDataset(test_samples, transform=eval_transform)

    use_pin_memory = pin_memory and torch.cuda.is_available()
    use_persistent = num_workers > 0

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
        "persistent_workers": use_persistent,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
