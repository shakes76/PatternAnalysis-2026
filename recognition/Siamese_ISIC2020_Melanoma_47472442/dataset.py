"""Load ISIC images, split patients and sample training batches.

Images and ``ISIC_2020_Training_GroundTruth_v2.csv`` live under repository
``data/``. Run this module to write the fixed patient split and print the
image, patient and melanoma counts for each role.

Author: Alex Anderson (47472442), COMP3710 Pattern Recognition, 2026.
"""

import csv
import json
import random
from pathlib import Path
from typing import Any, NamedTuple, cast

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
METADATA_CSV = DATA_DIR / "ISIC_2020_Training_GroundTruth_v2.csv"
SPLIT_FILE = DATA_DIR / "splits" / "isic_protocol.json"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet RGB statistics


class Sample(NamedTuple):
    """Store the identity, class and file path of one labelled image.

    Parameters
    ----------
    image_name : str
        ISIC image identifier without the file extension.
    patient_id : str
        Patient identifier used to keep images in the same split.
    label : int
        Class label: 0 for benign and 1 for melanoma.
    path : pathlib.Path
        Location of the JPEG image.
    """

    image_name: str
    patient_id: str
    label: int  # 0 benign, 1 melanoma
    path: Path


def read_samples(metadata_csv: Path = METADATA_CSV) -> list[Sample]:
    """Match metadata rows to JPEG images under the data directory.

    Parameters
    ----------
    metadata_csv : pathlib.Path, optional
        ISIC metadata CSV. Defaults to the file under repository ``data/``.
        Every listed image must have a matching JPEG there.

    Returns
    -------
    list of Sample
        Labelled image records in CSV row order.
    """
    paths = {path.stem: path for path in DATA_DIR.rglob("*.jpg")}
    with metadata_csv.open(newline="") as stream:
        return [
            Sample(row["image_name"], row["patient_id"], int(row["target"]), paths[row["image_name"]])
            for row in csv.DictReader(stream)
        ]


def _split_patients(labels: dict[str, int], fractions: tuple[float, ...], seed: int) -> list[list[str]]:
    """Split patients, stratified by whether they have melanoma.

    Parameters
    ----------
    labels : dict of str to int
        Patient identifiers mapped to 1 if any image is melanoma, else 0.
    fractions : tuple of float
        Fractions of each label group assigned to successive partitions.
        The last partition receives the remaining patients.
    seed : int
        Seed for shuffling patients within each label group.

    Returns
    -------
    list of list of str
        Sorted patient identifiers for each fraction, then the remainder.
    """
    rng = random.Random(seed)
    groups: list[list[str]] = [[] for _ in range(len(fractions) + 1)]
    for label in (0, 1):
        patients = sorted(patient for patient, target in labels.items() if target == label)
        rng.shuffle(patients)
        start = 0
        for group, fraction in zip(groups, fractions, strict=False):  # the last group takes the remainder
            count = round(len(patients) * fraction)
            group.extend(patients[start : start + count])
            start += count
        groups[-1].extend(patients[start:])
    return [sorted(group) for group in groups]


def create_splits(samples: list[Sample], output: Path = SPLIT_FILE) -> None:
    """Write fixed patient groups stratified by melanoma presence.

    Parameters
    ----------
    samples : list of Sample
        Labelled images whose patients will be assigned to groups.
    output : pathlib.Path, optional
        Destination JSON file, defaulting to
        ``data/splits/isic_protocol.json``. Parent directories are created;
        an existing file is replaced.

    Notes
    -----
    Save fit, development, selection, calibration and final patient groups.
    `load_splits` combines fit and development patients for training.
    """
    labels: dict[str, int] = {}
    for sample in samples:
        labels[sample.patient_id] = max(labels.get(sample.patient_id, 0), sample.label)
    tuning, development, pool = _split_patients(labels, (0.15, 0.15), 42)
    final, fit = _split_patients({p: labels[p] for p in pool}, (0.2,), 20261007)
    calibration, selection = _split_patients({p: labels[p] for p in tuning}, (0.5,), 20261008)
    partitions = dict(fit=fit, selection=selection, calibration=calibration, final=final, development=development)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"partitions": partitions}, indent=2) + "\n")


def load_splits(samples: list[Sample], split_file: Path = SPLIT_FILE) -> dict[str, list[Sample]]:
    """Group images using the saved patient assignments.

    Parameters
    ----------
    samples : list of Sample
        Labelled image records.
    split_file : pathlib.Path, optional
        Patient groups, defaulting to ``data/splits/isic_protocol.json``.

    Returns
    -------
    dict of str to list of Sample
        Image records for train, selection, calibration and final roles.
        Training combines the saved fit and development groups. Image order
        within each role follows the input records. An assertion stops the
        run if any patient appears in more than one role.
    """
    partitions = json.loads(split_file.read_text())["partitions"]
    roles = {
        "train": set(partitions["fit"] + partitions["development"]),
        "selection": set(partitions["selection"]),
        "calibration": set(partitions["calibration"]),
        "final": set(partitions["final"]),
    }
    assert sum(map(len, roles.values())) == len(set().union(*roles.values())), "patient groups overlap"
    return {role: [s for s in samples if s.patient_id in patients] for role, patients in roles.items()}


class ISICDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load RGB lesion images as normalised square tensors.

    Parameters
    ----------
    samples : list of Sample
        Labelled image records in dataset order.
    image_size : int
        Output height and width in pixels.
    train : bool, optional
        Apply random crops, flips, rotation and colour jitter if True.
        Default is False, which uses resizing and a centre crop.

    Attributes
    ----------
    samples : list of Sample
        Image records supplied to the dataset.
    image_size : int
        Output side length in pixels.
    transform : torchvision.transforms.Compose
        Image preparation followed by ImageNet RGB normalisation.
    """

    def __init__(self, samples: list[Sample], image_size: int, train: bool = False) -> None:
        """Store the image records and build the image transformations."""
        self.samples = samples
        self.image_size = image_size
        geometry = (
            [
                # Random crops of 35-100% of the image vary scale and position of the lesion
                transforms.RandomResizedCrop(image_size, scale=(0.35, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # RGB fills are supported; Torchvision infers only the scalar default's type.
                transforms.RandomRotation(45, fill=cast(Any, [round(255 * m) for m in MEAN])),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            ]
            if train
            else [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
        )
        self.transform = transforms.Compose([*geometry, transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self) -> int:
        """Return the number of labelled images."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load and transform one image and return its label.

        Parameters
        ----------
        index : int
            Position of the image record in the dataset.

        Returns
        -------
        image : torch.Tensor of shape (3, image_size, image_size)
            RGB image after transforms and ImageNet normalisation.
        label : int
            Class label: 0 for benign and 1 for melanoma.
        """
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            # Draft decoding reduces JPEG decoding work before resizing.
            image.draft("RGB", (self.image_size, self.image_size))
            image = ImageOps.exif_transpose(image).convert("RGB")
        return cast(torch.Tensor, self.transform(image)), sample.label


def melanoma_sampler(samples: list[Sample], melanoma_fraction: float, seed: int) -> WeightedRandomSampler:
    """Build a sampler with a target melanoma share per epoch.

    Parameters
    ----------
    samples : list of Sample
        Training records containing both classes.
    melanoma_fraction : float
        Expected melanoma fraction across draws, between 0 and 1.
    seed : int
        Seed for the sampler's random generator.

    Returns
    -------
    torch.utils.data.WeightedRandomSampler
        Draws ``len(samples)`` indices with replacement per epoch.
        Individual batches and epochs can differ from the target fraction.
    """
    melanomas = sum(s.label for s in samples)
    benign_weight = (1 - melanoma_fraction) / (len(samples) - melanomas)
    weights = [melanoma_fraction / melanomas if s.label else benign_weight for s in samples]
    return WeightedRandomSampler(weights, len(samples), generator=torch.Generator().manual_seed(seed))


if __name__ == "__main__":
    all_samples = read_samples()
    create_splits(all_samples)
    for role, group in load_splits(all_samples).items():
        patients = len({s.patient_id for s in group})
        print(f"{role}: {len(group)} images, {patients} patients, {sum(s.label for s in group)} melanomas")
