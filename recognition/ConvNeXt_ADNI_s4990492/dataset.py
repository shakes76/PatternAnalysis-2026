"""Load ADNI slices from a patient-level split manifest."""

import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def split_patient_rows(rows, seed=42):
    """Return the original stratified 70/15/remainder patient split in memory.

    This function never writes manifests. Reuse frozen manifests for evaluation.
    Sorted patient IDs and a local RNG preserve the historical assignment rule.
    """
    import random
    patient_class = {}
    for row in rows:
        patient, label = row['patient_id'], row['class_name']
        if label not in ('AD', 'NC'):
            raise ValueError('Unknown class: ' + label)
        if patient in patient_class and patient_class[patient] != label:
            raise ValueError('Patient has conflicting classes')
        patient_class[patient] = label
    rng = random.Random(seed)
    assignment = {}
    for label in ('AD', 'NC'):
        patients = sorted(p for p, c in patient_class.items() if c == label)
        rng.shuffle(patients)
        n_train, n_val = int(len(patients)*0.70), int(len(patients)*0.15)
        groups = {'train': patients[:n_train],
                  'val': patients[n_train:n_train+n_val],
                  'test': patients[n_train+n_val:]}
        for name, members in groups.items():
            assignment.update({patient: name for patient in members})
    result = {name: [] for name in ('train', 'val', 'test')}
    for row in rows:
        result[assignment[row['patient_id']]].append(row)
    return result


class ADNISliceDataset(Dataset):
    """Return a grayscale image tensor and its binary class label."""

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)

        with self.manifest_path.open(newline="") as file:
            reader = csv.DictReader(file)
            required = {"path", "target", "patient_id", "scan_id"}

            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"Manifest must contain these columns: {required}"
                )

            self.rows = list(reader)

        if not self.rows:
            raise ValueError("The manifest contains no images.")

        for row in self.rows:
            if row["target"] not in {"0", "1"}:
                raise ValueError(f"Invalid target: {row['target']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        with Image.open(row["path"]) as image:
            image = image.convert("L")
            image_tensor = TF.to_tensor(image)

        label = torch.tensor(int(row["target"]), dtype=torch.long)

        return image_tensor, label


class RGB224Dataset(ADNISliceDataset):
    def __init__(self, manifest_path, training=False):
        super().__init__(manifest_path)
        self.training = training
        self.affine = transforms.RandomAffine(5, translate=(0.03, 0.03),
                                             interpolation=InterpolationMode.BILINEAR, fill=0)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['path']) as source:
            image = ImageOps.pad(source.convert('L'), (224, 224),
                                 method=Image.Resampling.BILINEAR, color=0)
            if self.training:
                image = self.affine(image)
            image = TF.to_tensor(image.convert('RGB'))
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.tensor(int(row['target']), dtype=torch.long)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check one image from an ADNI split manifest."
    )
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    dataset = ADNISliceDataset(args.manifest)
    image, label = dataset[0]

    print("Images in manifest:", len(dataset))
    print("Image shape [channels, height, width]:", tuple(image.shape))
    print("Image dtype:", image.dtype)
    print("Pixel range:", image.min().item(), image.max().item())
    print("Label:", label.item())
    print("All pixel values finite:", torch.isfinite(image).all().item())