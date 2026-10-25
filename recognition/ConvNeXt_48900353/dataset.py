"""
File for dataset loading, loading images, connecting images to subjects, creating subject splits,
and applying preprocessing steps ready to be inputted to models.

Images are collected from each dataset split and class folder, then each subject ID is
extracted from the filename before its slice index. The resulting image table is split
into training, validation, and test sets by subject, ensuring that slices from the same
subject remain in only one split.

"""

from pathlib import Path
from collections import Counter

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

IMAGE_SIZE = (224, 224)

#random transforms vary training images before scaling and normalization
TRAIN_TRANSFORM = v2.Compose([
    v2.Resize((256, 256)),
    v2.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomAffine(
        degrees=15, translate=(0.1, 0.1), scale=(0.9, 1)
    ),
    v2.ColorJitter(
        brightness=0.1, contrast=0.15,
    ),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.1160], std=[0.2230]),

])

# evaluation uses fixed transforms so repeated predictions are comparable
EVAL_TRANSFORM = v2.Compose([
    v2.Resize(IMAGE_SIZE),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.1160], std=[0.2230]),
])


def subject_id_from_filename(path):
    """Extract the subject identifier before the first underscore in a filename

    Args:
        path: Image path whose filename begins with a subject ID

    Returns:
        str: Subject ID extracted from file name
    """
    parts = Path(path).stem.split("_")
    return parts[0]


def build_image_table(dataset_root, class_names, dataset_split_names):
    """Collect JPEG image paths, class names, and subject IDs from dataset folders

    Args:
        dataset_root: root containing the original split folders
        class_names: class subfolders to scan within each split
        dataset_split_names: original split folders to scan

    Returns:
        list[dict]: one record per .jpeg file with image_path, class_name and subject_id keys
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    rows = []

    # iterate through all folders in dataset
    for split in dataset_split_names:
        for class_name in class_names:
            folder = dataset_root / split / class_name

            # sort JPEG paths
            paths = sorted(
                path for path in folder.rglob("*")
                if path.suffix.lower() == ".jpeg" and path.is_file()
            )

            # append (image path, class, subject id)
            for path in paths:
                subject_id = subject_id_from_filename(path)
                rows.append({
                    "image_path": str(path),
                    "class_name": class_name,
                    "subject_id": subject_id,
                })
    return rows


def create_splits(dataset_root, class_names, dataset_split_names, seed):
    """Create stratified subject splits and print image and class counts

    Args:
        dataset_root: root containing split and class subfolders
        class_names: class folders to include
        dataset_split_names: original split folders to pool together
        seed: random state used for both subject-level splits

    Returns:
        tuple: Training, validation, and test image splits
    """
    # create table where each row is exactly one image with its subject ID
    image_records = build_image_table(
        dataset_root,
        class_names,
        dataset_split_names,
    )

    # split subjects into 70% training and 30% for validation/testing
    subjects_by_id = {}
    for record in image_records:
        subjects_by_id.setdefault(
            record["subject_id"],
            {"subject_id": record["subject_id"], "class_name": record["class_name"]},
        )
    subjects = list(subjects_by_id.values())
    train, remaining = train_test_split(
        subjects,
        test_size=0.3,
        stratify=[subject["class_name"] for subject in subjects],
        random_state=seed,
    )
    val, test = train_test_split(
        remaining,
        test_size=1 / 3, # one third of the remaining 30% is 10% overall
        stratify=[subject["class_name"] for subject in remaining],
        random_state=seed,
    )

    # count all images
    image_count = len(image_records)
    print(f"Total: {image_count} images from {len(subjects)} subjects")
    splits = []

    # build each split from subject IDs so all slices from one subject stay together
    for name, group in (("Train", train), ("Validation", val), ("Test", test)):
        subject_ids = {subject["subject_id"] for subject in group}
        split_records = [
            record for record in image_records
            if record["subject_id"] in subject_ids
        ]
        splits.append(split_records)

        # print image and subject counts, and class distributions
        print(
            f"\n{name}: {len(split_records)} images "
            f"({len(split_records) / image_count:.2%}), "
            f"{len(group)} subjects"
        )
        class_counts = Counter(record["class_name"] for record in split_records)
        print("\n".join(
            f"{class_name}    {class_counts[class_name]}"
            for class_name in class_names
            if class_counts[class_name]
        ))

    return tuple(splits)


def preprocess_image(image, augment=False):
    """Resize, augment, and normalize an image for model input.

    Args:
        image: input image, expected to be single-channel grayscale
        augment: apply random training transforms when True, otherwise use evaluation transforms

    Returns:
        Tensor: Normalized float32 image of shape (1, 224, 224)
    """
    transform = TRAIN_TRANSFORM if augment else EVAL_TRANSFORM
    return transform(image)


class ADNIDataset(Dataset):
    """Dataset for loading and preprocessing a split of ADNI images"""

    def __init__(self, records, label_map, augment=False):
        """Store image records, label indices, and the selected preprocessing transform.

        Args:
            records: records with image_path and class_name keys.
            label_map: mapping from each class name to its target index
            augment: whether to use training-time image augmentation
        """
        self.samples = [
            (record["image_path"], record["class_name"])
            for record in records
        ]
        self.transform = TRAIN_TRANSFORM if augment else EVAL_TRANSFORM
        self.label_map = label_map

    def __len__(self):
        """Count the image samples stored in this split.

        Returns:
            int: number of image records in the dataset
        """
        return len(self.samples)

    def __getitem__(self, index):
        """Load and preprocess one image and convert its class into a target tensor.

        Args:
            index: position of the sample in the stored image records

        Returns:
            tuple: Float32 image tensor of shape (1, 224, 224) for grayscale files
        """
        image_path, class_name = self.samples[index]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image)
        # CrossEntropyLoss expects integer class indices stored as torch.long.
        label = torch.tensor(self.label_map[class_name], dtype=torch.long)
        return image_tensor, label


def create_dataloaders(dataset_root, class_names, dataset_split_names, label_map, seed, batch_size=32, num_workers=0):
    """Build subject-separated loaders with augmentation and shuffling for training.

    Args:
        dataset_root: root containing the original dataset splits
        class_names: class subfolders to include
        dataset_split_names: original split folders to pool together
        label_ma: mapping from class names to integer targets
        seed: seed for subject splitting and each loader generator
        batch_size: maximum number of images in each batch
        num_workers: number of loading processes; zero loads in the main process

    Returns:
        tuple: Training, validation, and test loaders
    """

    splits = create_splits(dataset_root,class_names,dataset_split_names, seed)

    datasets = []
    for split_index, frame in enumerate(splits):
        is_training = split_index == 0
        datasets.append(
            ADNIDataset(frame, label_map, augment=is_training)
        )

    loaders = []
    for split_index, dataset in enumerate(datasets):
        loaders.append(DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_index == 0, # only shuffle training set
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed),  # seed sampling and worker RNGs
            pin_memory=torch.cuda.is_available(),  # use pinned host memory for CUDA transfers
            persistent_workers=num_workers > 0,  # reuse loading workers across epochs
        ))
    return tuple(loaders)

