"""Data preparation and loading for ADNI MRI slice classification."""

import argparse
import json
import re
import zipfile
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


LABEL_MAP = {"NC": 0, "AD": 1}
DUPLICATE_FILENAME = "976382_96 (1).jpeg"


def build_image_inventory(image_root, metadata_path):
    """Match the original notebook's folder order and metadata ID mapping.

    Inputs are already extracted course images and meta_data_with_label.json.
    Folder labels are retained; metadata supplies patient identifiers only.
    """
    image_root = Path(image_root).resolve()
    with Path(metadata_path).open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary keyed by image ID.")

    mapping = {}
    for image_id, info in metadata.items():
        raw = info.get("raw", "") if isinstance(info, dict) else ""
        subjects = set(re.findall(r"\d{3}_S_\d{4}", str(raw)))
        if len(subjects) > 1:
            raise ValueError(f"Ambiguous patient mapping for image {image_id}.")
        if subjects:
            mapping[str(image_id)] = subjects.pop()

    records = []
    for split in ("train", "test"):
        for label in ("AD", "NC"):
            folder = image_root / split / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing image folder: {folder}")
            images = sorted(
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if not images:
                raise ValueError(f"No images found in {folder}.")
            for path in images:
                image_id = path.stem.split("_")[0]
                records.append({
                    "split": split, "label": label, "filename": path.name,
                    "image_id": image_id, "path": str(path),
                    "patient_id": mapping.get(image_id),
                })

    frame = pd.DataFrame(records)
    missing = frame.loc[frame["patient_id"].isna(), "filename"]
    if not missing.empty:
        raise ValueError(
            f"{len(missing)} images have no patient mapping: {missing.head().tolist()}"
        )
    return frame


def validate_splits(train_df, val_df, test_df):
    """Check labels, paths, and patient independence."""
    frames = [train_df, val_df, test_df]
    names = ["train", "val", "test"]
    required = {"patient_id", "image_id", "label", "path"}

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

    combined = pd.concat(frames, ignore_index=True)
    if combined.groupby("patient_id")["label"].nunique().gt(1).any():
        raise ValueError("Conflicting labels found within patients.")

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

    patients = (
        candidates[["patient_id", "label"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    train_patients, holdout_patients = train_test_split(
        patients,
        test_size=0.30,
        random_state=seed,
        stratify=patients["label"],
    )

    val_patients, test_patients = train_test_split(
        holdout_patients,
        test_size=0.50,
        random_state=seed,
        stratify=holdout_patients["label"],
    )

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

    if set().union(*(set(f["patient_id"]) for f in splits)) != set(candidates["patient_id"]):
        raise ValueError("Some candidate patients were lost during splitting.")
    return splits


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


def resolve_image_paths(frame, source_root, image_root):
    """Map original CSV paths to an existing image directory.

    `source_root` is the original root recorded in the CSV.
    `image_root` contains the same relative folder structure, such as
    the extracted Colab cache. Images are not copied or extracted here.
    """
    source_root = Path(source_root)
    image_root = Path(image_root)
    result = frame.copy()

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


def build_transforms(model_name, image_size=256):
    """Return (training_transform, evaluation_transform) for the named model.

    Both pipelines resize to 256x256 and map [0,1] intensities to [-1,1].
    Erasing uses zero before normalization (black pixels after display).
    The GFNet notebook deliberately disables erasing, translation and elastic
    deformation. Re-enabling them would define a different experiment.
    """
    model_name = model_name.lower()
    if model_name not in {"cnn", "gfnet"}:
        raise ValueError("model_name must be 'cnn' or 'gfnet'.")

    evaluation_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Preserve the final notebooks' different training augmentation pipelines.
    # Neither validation nor test data receive random augmentation.
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


class MRIDataset(Dataset):
    """Load grayscale MRI slices with NC=0 and AD=1."""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.label_map = LABEL_MAP.copy()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        """Return one transformed grayscale tensor and integer NC/AD label."""
        record = self.dataframe.iloc[index]

        with Image.open(record["local_path"]) as source:
            image = source.convert("L")

        if self.transform is not None:
            image = self.transform(image)

        label = self.label_map[record["label"]]
        return image, label


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
    frames = load_split_csvs(split_folder)
    frames = tuple(
        resolve_image_paths(frame, source_root, image_root)
        for frame in frames
    )

    train_transform, evaluation_transform = build_transforms(model_name)

    datasets = [
        MRIDataset(frames[0], train_transform),
        MRIDataset(frames[1], evaluation_transform),
        MRIDataset(frames[2], evaluation_transform),
    ]

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


def compare_split_membership(new_frames, reference_dir):
    """Compare every image assignment, ignoring machine-specific root paths."""
    reference_frames = load_split_csvs(reference_dir)

    def keys(frame):
        return sorted(
            (str(row.patient_id), str(row.image_id), str(row.label),
             '/'.join(Path(row.path).parts[-3:]))
            for row in frame.itertuples(index=False)
        )

    for name, new, old in zip(('train', 'val', 'test'), new_frames, reference_frames):
        if keys(new) != keys(old):
            raise ValueError(f'{name}: assignments differ from reference splits.')



def extract_image_archives(archive_root, image_root):
    """Extract the notebook's four ZIPs into a fresh cache directory.

    Archives are train/AD.zip, train/NC.zip, test/AD.zip and test/NC.zip.
    Flat image entries are expected, matching the supplied notebooks.
    Existing nonempty caches are never overwritten.
    """
    root = Path(image_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Extraction requires a new or empty image root.")
    plans = []
    for split in ("train", "test"):
        for label in ("AD", "NC"):
            archive = Path(archive_root) / split / f"{label}.zip"
            with zipfile.ZipFile(archive) as zf:
                names = set()
                for item in zf.infolist():
                    if item.is_dir():
                        continue
                    # Reject traversal and unexpected nesting before extraction.
                    if (Path(item.filename).name != item.filename
                            or "\\" in item.filename):
                        raise ValueError(f"Expected flat ZIP image names: {item.filename}")
                    if item.filename in names:
                        raise ValueError(f"Duplicate ZIP member: {item.filename}")
                    names.add(item.filename)
            plans.append((archive, root / split / label))
    for archive, destination in plans:
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for item in zf.infolist():
                if not item.is_dir():
                    with zf.open(item) as src, (destination / item.filename).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        print("Extracted:", archive, flush=True)


def save_data_figures(records, frames, output):
    """Save inventory plots, raw input examples and split counts.

    Example labels follow course annotations; this is not a diagnostic audit.
    """
    counts = records.groupby(["split", "label"], sort=False).size()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["/".join(k) for k in counts.index], counts.to_numpy())
    ax.set(xlabel="Original split / class", ylabel="Number of images",
           title="ADNI Image Class Distribution")
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    properties = []
    for ax, label in zip(axes, ("AD", "NC")):
        row = records.loc[records.label == label].iloc[0]
        with Image.open(row.path) as image:
            array = np.asarray(image)
            properties.append({"label": label, "filename": row.filename,
                               "size": list(image.size), "mode": image.mode,
                               "pixel_min": float(array.min()),
                               "pixel_max": float(array.max()),
                               "pixel_mean": float(array.mean())})
            ax.imshow(image.convert("L"), cmap="gray")
        ax.set_title("Alzheimer's Disease (AD)" if label == "AD" else "Normal Control (NC)")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "adni_examples.png", dpi=200)
    plt.close(fig)
    with (output / "example_image_properties.json").open("w") as file:
        json.dump(properties, file, indent=2)
    rows = []
    for name, frame in zip(("train", "val", "test"), frames):
        for label in LABEL_MAP:
            subset = frame.loc[frame.label == label]
            rows.append({"split": name, "label": label, "images": len(subset),
                         "patients": subset.patient_id.nunique()})
    pd.DataFrame(rows).to_csv(output / "split_summary.csv", index=False)

def main():
    """Prepare the raw inventory, patient-independent splits and audit figures."""
    parser = argparse.ArgumentParser(description='Prepare ADNI patient-level split CSVs.')
    parser.add_argument('--image-root', required=True,
                        help='Extracted AD_NC root containing train/ and test/.')
    parser.add_argument('--metadata', required=True,
                        help='Path to meta_data_with_label.json.')
    parser.add_argument('--output-dir', required=True,
                        help='New or empty directory for generated CSVs.')
    parser.add_argument('--reference-split-dir',
                        help='Optional existing splits to verify exact image membership.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--archive-root', help='Optional root containing four ZIP archives; extracts into image-root.')
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Choose a new or empty output directory.')

    if args.archive_root:
        extract_image_archives(args.archive_root, args.image_root)
    records = build_image_inventory(args.image_root, args.metadata)
    frames = create_patient_splits(records, seed=args.seed)
    if args.reference_split_dir:
        compare_split_membership(frames, args.reference_split_dir)
        print('Reference comparison passed for every image assignment.')

    original_train = set(records.loc[records['split'] == 'train', 'patient_id'])
    original_test = set(records.loc[records['split'] == 'test', 'patient_id'])
    summary = {
        'image_root': str(Path(args.image_root).resolve()),
        'metadata': str(Path(args.metadata).resolve()),
        'seed': args.seed,
        'original_images': len(records),
        'original_patients': int(records.patient_id.nunique()),
        'unmapped_images': 0,
        'conflicting_patient_labels': 0,
        'final_patient_overlap': 0,
        'original_patient_overlap': len(original_train & original_test),
        'removed_duplicate_records': int((records['filename'] == DUPLICATE_FILENAME).sum()),
        'retained_images': sum(len(frame) for frame in frames),
        'splits': {},
    }
    output.mkdir(parents=True, exist_ok=True)
    records.to_csv(output / 'image_records.csv', index=False)
    save_data_figures(records, frames, output)
    for name, frame in zip(('train', 'val', 'test'), frames):
        frame.to_csv(output / f'{name}.csv', index=False)
        patients = frame[['patient_id', 'label']].drop_duplicates()
        summary['splits'][name] = {
            'images': len(frame), 'patients': len(patients),
            'images_per_class': {label: int((frame['label'] == label).sum())
                                 for label in LABEL_MAP},
            'patients_per_class': {label: int((patients['label'] == label).sum())
                                   for label in LABEL_MAP},
        }
    with (output / 'preparation_summary.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
    print('Saved CSVs to:', output)
    print('Use this image_root as --source-root when loading these CSVs.')


if __name__ == '__main__':
    main()
