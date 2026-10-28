"""
Data loading and preprocessing for ADNI AD vs NC classification

 Expected layout:
      <data_root>/train/AD/*.jpeg
      <data_root>/train/NC/*.jpeg
      <data_root>/test/AD/*.jpeg
      <data_root>/test/NC/*.jpeg

Filenames are <scanID>_<sliceNumber>.jpeg, with 20 slices per scan.

Some subjects have several scans (different visits), so splitting is done by
subject using <data_root>/../meta_data_with_label.json when it is available.
"""

import os
import re
import json
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASSES = {"NC": 0, "AD": 1}
IMG_EXTS = (".jpeg", ".jpg", ".png")


def get_patient_id(filename):
    """Return the scan ID from a filename of the above format"""
    return os.path.basename(filename).split("_")[0]


def load_subject_map(data_root):
    """
    Map each scan ID to its ADNI subject ID (e.g: "068_S_0473") using the
    metadata JSON in the AD_NC folder. Returns {} if the file is missing.
    """
    meta_path = os.path.join(
        os.path.dirname(os.path.normpath(data_root)), "meta_data_with_label.json"
    )
    if not os.path.exists(meta_path):
        print(f"No metadata at {meta_path}, splitting by scan ID instead")
        return {}

    with open(meta_path) as f:
        meta = json.load(f)

    subject_map = {}
    for scan_id, entry in meta.items():
        match = re.search(r"ADNI_(\d{3}_S_\d{4})", entry["raw"])
        if match:
            subject_map[scan_id] = match.group(1)
    return subject_map


def gather_samples(data_root, split, subject_map=None):
    """
    List every (image_path, label, subject_id) in data_root/split/{AD,NC}.
    Falls back to the scan ID if the subject is unknown.
    """
    subject_map = subject_map or {}
    samples = []
    for class_name, label in CLASSES.items():
        class_dir = os.path.join(data_root, split, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(IMG_EXTS):
                scan_id = get_patient_id(fname)
                subject_id = subject_map.get(scan_id, scan_id)
                samples.append((os.path.join(class_dir, fname), label, subject_id))
    return samples


def split_by_patient(samples, val_fraction=0.15, seed=42):
    """
    Split samples into train/val by patient to esnure that each
    patient is only in one split. Uses stratified sampling on the label so the AD/NC ratio
    is preserved in both splits.
    """
    patient_label = {pid: lbl for _, lbl, pid in samples}  # one label per patient
    patients = sorted(patient_label)
    labels = [patient_label[p] for p in patients]

    train_pids, val_pids = train_test_split(
        patients, test_size=val_fraction, stratify=labels, random_state=seed
    )

    val_pids = set(val_pids)
    train = [s for s in samples if s[2] not in val_pids]
    val = [s for s in samples if s[2] in val_pids]
    return train, val


class ADNIDataset(Dataset):
    """Returns (image_tensor, label) for each MRI slice."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        image = Image.open(path).convert("L")  # force grey scale
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class PerImageStandardise:
    """Scale each image to zero mean and variance to 1 to ensure same scale of intensity
    from scan to scan.
    """

    def __call__(self, x):
        return (x - x.mean()) / (x.std() + 1e-8)


def get_transforms(train, strong_aug=False):
    """Transforms the training images with flips, rotations, scaling, etc, while
    both the training and test gets turned into tensors and standardised.
    strong_aug makes the training changes bigger so the model can't rely on
    exact head shape, position or brightness to recognise individual subjects.
    """
    steps = []
    if train and not strong_aug:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
        ]
    if train and strong_aug:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    steps += [transforms.ToTensor()]
    if train and strong_aug:
        # blank out a small random patch (works on tensors, so after ToTensor)
        steps += [transforms.RandomErasing(p=0.25, scale=(0.02, 0.1))]
    steps += [PerImageStandardise()]
    return transforms.Compose(steps)


def get_dataloaders(
    data_root,
    batch_size=32,
    val_fraction=0.15,
    num_workers=4,
    seed=42,
    strong_aug=False,
):
    """Build train/val/test DataLoaders based on patient split."""
    subject_map = load_subject_map(data_root)
    train_samples, val_samples = split_by_patient(
        gather_samples(data_root, "train", subject_map), val_fraction, seed
    )
    test_samples = gather_samples(data_root, "test", subject_map)

    # train and val must never share a patient
    train_pids = {pid for _, _, pid in train_samples}
    val_pids = {pid for _, _, pid in val_samples}
    test_pids = {pid for _, _, pid in test_samples}
    assert not (train_pids & val_pids), "Patient leakage between train and val"

    # the official test split shares some subjects with train:
    print(
        f"Subjects: train {len(train_pids)}, val {len(val_pids)}, test {len(test_pids)}"
        f" | test subjects also in train/val: {len(test_pids & (train_pids | val_pids))}"
    )

    train_ds = ADNIDataset(
        train_samples, get_transforms(train=True, strong_aug=strong_aug)
    )
    val_ds = ADNIDataset(val_samples, get_transforms(train=False))
    test_ds = ADNIDataset(test_samples, get_transforms(train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
