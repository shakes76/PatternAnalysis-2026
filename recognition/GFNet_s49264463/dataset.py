"""Data loading for the preprocessed ADNI brain MRI slices (AD vs NC).

Expected layout under `root`:
    root/{train,test}/{AD,NC}/<patientID>_<sliceNumber>.jpeg

The provided data has no validation set, so one is carved out of the training
patients here. Splitting is always by patient (never by slice), because each
patient contributes about 20 near-identical slices and a slice-level split
would let the model be validated on brains it has already seen.
"""
import os
import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Class folder name -> label
CLASS_TO_LABEL = {"NC": 0, "AD": 1}


def group_by_patient(root, split):
    """Group image files by patient ID for one split ("train" or "test").

    Returns {patient_id: [(file_path, label), ...]}.
    """
    patients = defaultdict(list)
    for class_name, label in CLASS_TO_LABEL.items():
        folder = os.path.join(root, split, class_name)
        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            patient_id = filename.split("_")[0]   # "218391_78.jpeg" -> "218391"
            patients[patient_id].append((os.path.join(folder, filename), label))
    return dict(patients)


def split_train_val(patients, val_frac=0.15, seed=42, gap=50):
    """Split by groups of nearby patient IDs (likely repeat scans of one person)."""
    ids = sorted(patients, key=int)
    groups, cur = [], [ids[0]]
    for x, y in zip(ids, ids[1:]):
        if int(y) - int(x) <= gap:
            cur.append(y)
        else:
            groups.append(cur)
            cur = [y]
    groups.append(cur)

    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = round(val_frac * len(ids))
    val_ids = set()
    for g in groups:
        if len(val_ids) >= n_val:
            break
        val_ids.update(g)

    train_ids = set(patients) - val_ids
    assert train_ids.isdisjoint(val_ids), "Patient appears in both train and val!"
    train_items = [it for pid in sorted(train_ids) for it in patients[pid]]
    val_items = [it for pid in sorted(val_ids) for it in patients[pid]]
    return train_items, val_items


def get_transforms(train):
    """Resize + normalise for all sets; light augmentation for training only."""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),   # placeholder: compute from train later
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


class ADNIDataset(Dataset):
    """Serves (image_tensor, label) pairs from a list of (path, label)."""

    def __init__(self, items, transform=None):
        self.items = items            # list of (file_path, label)
        self.transform = transform    # train or eval transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        image = Image.open(path)      # stays greyscale (1 channel)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_dataloaders(root, batch_size=32, num_workers=2):
    """Build train, validation and test DataLoaders (shuffle on for train only)."""
    train_patients = group_by_patient(root, "train")
    train_items, val_items = split_train_val(train_patients)
    test_items = [it for items in group_by_patient(root, "test").values() for it in items]

    train_ds = ADNIDataset(train_items, get_transforms(train=True))
    val_ds = ADNIDataset(val_items, get_transforms(train=False))
    test_ds = ADNIDataset(test_items, get_transforms(train=False))

    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers),
    )
