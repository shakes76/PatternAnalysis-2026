"""Data preparation and loading for ADNI MRI slice classification."""

from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


LABEL_MAP = {"NC": 0, "AD": 1}
DUPLICATE_FILENAME = "976382_96 (1).jpeg"


# Validate the integrity of all splits before training or evaluation.
def validate_splits(train_df, val_df, test_df):
    """Check labels, paths, and patient independence."""
    frames = [train_df, val_df, test_df]
    names = ["train", "val", "test"]
    required = {"patient_id", "image_id", "label", "path"}

    # Check required columns, non-empty splits, labels, and unique paths.
    for name, frame in zip(names, frames):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")

        if frame.empty:
            raise ValueError(f"{name}: dataset is empty.")

        if frame[list(required)].isna().any().any():
            raise ValueError(f"{name}: required fields contain missing values.")

        if not frame["label"].isin(LABEL_MAP).all():
            raise ValueError(f"{name}: unexpected class labels.")

        if not frame["path"].is_unique:
            raise ValueError(f"{name}: duplicate image paths.")

    # Ensure each patient has one consistent diagnosis across all records.
    combined = pd.concat(frames, ignore_index=True)
    if combined.groupby("patient_id")["label"].nunique().gt(1).any():
        raise ValueError("Conflicting labels found within patients.")

    # Prevent patient-level or image-level leakage between splits.
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            patient_overlap = (
                set(frames[i]["patient_id"])
                & set(frames[j]["patient_id"])
            )
            path_overlap = (
                set(frames[i]["path"])
                & set(frames[j]["path"])
            )

            if patient_overlap or path_overlap:
                raise ValueError(
                    f"Overlap detected between {names[i]} and {names[j]}."
                )


# Create patient-independent, class-stratified train/validation/test splits.
def create_patient_splits(records, seed=42):
    """Reproduce the final stratified 70/15/15 patient split.

    `records` should contain the full image inventory produced by
    data exploration. Original image paths and row order are preserved.
    This function returns DataFrames and does not write files.
    """
    required = {"patient_id", "image_id", "label", "path", "filename"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    candidates = records.copy()

    if candidates[list(required)].isna().any().any():
        raise ValueError("Required fields contain missing values.")

    candidates["patient_id"] = candidates["patient_id"].astype(str)
    candidates["image_id"] = candidates["image_id"].astype(str)

    # Exclude the duplicate file identified during dataset exploration.
    # Remove the specific duplicate identified during data exploration.
    candidates = candidates[
        candidates["filename"] != DUPLICATE_FILENAME
    ].copy().reset_index(drop=True)

    if not candidates["label"].isin(LABEL_MAP).all():
        raise ValueError("Unexpected class labels.")

    if not candidates["path"].is_unique:
        raise ValueError("Duplicate image paths remain.")

    if candidates.groupby("patient_id")["label"].nunique().gt(1).any():
        raise ValueError("Conflicting labels found within patients.")

    # Split patients rather than individual MRI slices to avoid leakage.
    patients = (
        candidates[["patient_id", "label"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    # Reserve 30% of patients for validation and testing.
    train_patients, holdout_patients = train_test_split(
        patients,
        test_size=0.30,
        random_state=seed,
        stratify=patients["label"],
    )

    # Divide the held-out patients equally into validation and test sets.
    val_patients, test_patients = train_test_split(
        holdout_patients,
        test_size=0.50,
        random_state=seed,
        stratify=holdout_patients["label"],
    )

    # Recover all MRI slices belonging to the selected patients.
    def collect(patient_frame, split_name):
        frame = candidates[
            candidates["patient_id"].isin(patient_frame["patient_id"])
        ].copy()
        frame["split"] = split_name
        return frame.reset_index(drop=True)

    splits = (
        collect(train_patients, "train"),
        collect(val_patients, "val"),
        collect(test_patients, "test"),
    )

    validate_splits(*splits)

    if sum(len(frame) for frame in splits) != len(candidates):
        raise ValueError("Some candidate images were lost during splitting.")

    return splits


# Read the saved CSV splits and verify their consistency.
def load_split_csvs(split_folder):
    """Load existing final splits without modifying them."""
    split_folder = Path(split_folder)

    frames = tuple(
        pd.read_csv(
            split_folder / f"{name}.csv",
            dtype={"image_id": str, "patient_id": str},
        )
        for name in ("train", "val", "test")
    )

    validate_splits(*frames)
    return frames


# Translate original CSV image paths to the current extracted image root.
def resolve_image_paths(frame, source_root, image_root):
    """Map original CSV paths to an existing image directory.

    `source_root` is the original root recorded in the CSV.
    `image_root` contains the same relative folder structure, such as
    the extracted Colab cache. Images are not copied or extracted here.
    """
    source_root = Path(source_root)
    image_root = Path(image_root)
    result = frame.copy()

    # Preserve relative paths while replacing the original dataset root.
    def map_path(value):
        original = Path(value)
        try:
            relative = original.relative_to(source_root)
        except ValueError as error:
            raise ValueError(
                f"Image path is outside source_root: {original}"
            ) from error

        if ".." in relative.parts:
            raise ValueError(f"Invalid relative image path: {relative}")

        return str(image_root / relative)

    result["local_path"] = result["path"].map(map_path)

    # Fail early if any image referenced by a split is unavailable.
    missing = result.loc[
        ~result["local_path"].map(lambda p: Path(p).is_file()),
        "local_path",
    ]

    if not missing.empty:
        raise FileNotFoundError(
            f"{len(missing)} images are missing. Examples:\n"
            + "\n".join(missing.head().tolist())
        )

    return result


# Construct model-specific training augmentation and shared evaluation preprocessing.
def build_transforms(model_name, image_size=256):
    """Return the final experiment's training and evaluation transforms."""
    model_name = model_name.lower()
    if model_name not in {"cnn", "gfnet"}:
        raise ValueError("model_name must be 'cnn' or 'gfnet'.")

    # Evaluation uses deterministic resizing, tensor conversion, and normalization.
    evaluation_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # CNN training uses affine, erasing, and optional elastic augmentation.
    if model_name == "cnn":
        training_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.98, 1.02),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.4,
                scale=(0.01, 0.10),
                ratio=(0.5, 2.0),
                value=0,
            ),
            transforms.RandomApply([
                transforms.ElasticTransform(
                    alpha=10.0,
                    sigma=3.0,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0,
                )
            ], p=0.3),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        # GFNet training uses horizontal flips and affine augmentation.
    else:
        training_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=10,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    return training_transform, evaluation_transform


# Wrap MRI slice records as a PyTorch dataset.
class MRIDataset(Dataset):
    """Load grayscale MRI slices with NC=0 and AD=1."""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.label_map = LABEL_MAP.copy()

    def __len__(self):
        return len(self.dataframe)

    # Load one grayscale MRI slice and its binary class label.
    def __getitem__(self, index):
        record = self.dataframe.iloc[index]

        with Image.open(record["local_path"]) as source:
            image = source.convert("L")

        if self.transform is not None:
            image = self.transform(image)

        label = self.label_map[record["label"]]
        return image, label


# Assemble DataLoaders from the validated patient-independent splits.
def build_dataloaders(
    split_folder,
    source_root,
    image_root,
    model_name,
    batch_size=32,
    num_workers=8,
    pin_memory=True,
):
    """Build train, validation, and test loaders from saved split CSVs."""
    # Load the saved splits and resolve their paths to actual image files.
    frames = load_split_csvs(split_folder)
    frames = tuple(
        resolve_image_paths(frame, source_root, image_root)
        for frame in frames
    )

    # Apply stochastic augmentation only to the training split.
    train_transform, evaluation_transform = build_transforms(model_name)

    datasets = [
        MRIDataset(frames[0], train_transform),
        MRIDataset(frames[1], evaluation_transform),
        MRIDataset(frames[2], evaluation_transform),
    ]

    # Shuffle training samples; preserve validation and test order.
    return tuple(
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(index == 0),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for index, dataset in enumerate(datasets)
    )