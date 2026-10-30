import random
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = ["NC", "AD"]  # index 0 = NC, index 1 = AD (AD is the "positive" class)
MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
ROOT = Path("data/ADNI/AD_NC")


def class_to_idx(cls):
    if cls == "NC":
        return 0
    return 1

def scan_split(split):
    """Transforms raw data files to a list of (filepath, class, id) tuples"""
    records = [] 
    for cls in CLASS_NAMES:
        folder = Path(ROOT) / split / cls          # e.g. AD_NC/test/AD
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")
        for p in sorted((ROOT/split/cls).glob("*.jpeg")): # Gets files from location like AD_NC/test/AD
            patient_id, slice_num = p.stem.split('_')

            records.append((str(p), class_to_idx(cls), patient_id))
    if not records:
        raise FileNotFoundError(f"No images found under {ROOT/split}")
    return records


def split_by_patient(records, val_frac=0.15, seed=123):
    """Patient-level, class-stratified split into (val, test)."""
    patients_by_label = {0: [], 1: []}
    for path, label, pid in records:
        if pid not in patients_by_label[label]:
            patients_by_label[label].append(pid)

    rng = random.Random(seed)
    val_patients = []
    train_patients = []
    for label, patients in patients_by_label.items():
        rng.shuffle(patients)
        n = int(len(patients) * val_frac)
        val_patients = val_patients + patients[:n]
        train_patients = train_patients + patients[n:]
    val = []
    train = []
    for r in records:
        if r[2] in val_patients:
            val.append(r)
        if r[2] in train_patients:
            train.append(r)
    assert not ({r[2] for r in train} & {r[2] for r in val}), "Patient leakage: train/val"
    return train, val


class ADNIDataset(Dataset):
    """Transforms greyscale slices, resized, augmented (train only), 3-channel normalised tensor."""

    def __init__(self, records, img_size=224, train=False, return_path=False):
        self.records, self.return_path = records, return_path
        self.patient_ids = [r[2] for r in records]   
        steps = [T.Resize((img_size, img_size))]
        if train:
            steps += [T.RandomHorizontalFlip(), T.RandomAffine(degrees=5, translate=(0.03, 0.03))]
        steps += [T.ToTensor(), T.Normalize(MEAN, STD)]
        self.tf = T.Compose(steps)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        path, label, _ = self.records[i]
        img = self.tf(Image.open(path).convert("RGB"))  # 1 channel replicated to 3
        return (img, label, path) if self.return_path else (img, label)


def _load_splits(val_frac=0.15, seed=123):
    train_all = scan_split("train")
    test = scan_split("test")
    train, val = split_by_patient(train_all, val_frac, seed)
    seen = {r[2] for r in train_all}
    assert not (seen & {r[2] for r in test}), "Patient leakage: train/test"
    return train, val, test


def get_dataloaders(root=None, batch_size=64, img_size=224, num_workers=3, val_frac=0.15, seed=123):
    """Returns (train_loader, val_loader, test_loader)."""
    train, val, test = _load_splits(val_frac, seed)
    for name, recs in (("train", train), ("val", val), ("test", test)):
        num_ad = sum(r[1] for r in recs)
        # Print a summary of data loaded
        print(f"{name}: {len(recs)} images, {len({r[2] for r in recs})} patients, "
              f"AD={num_ad} NC={len(recs) - num_ad}")
    return (
        DataLoader(ADNIDataset(train, img_size, train=True), batch_size, shuffle=True, drop_last=True, 
                   num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        DataLoader(ADNIDataset(val, img_size), batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        DataLoader(ADNIDataset(test, img_size), batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
    )