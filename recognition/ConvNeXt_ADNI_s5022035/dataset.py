import os
import random
import json
import hashlib
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


class ADNIDataset(Dataset):
    def __init__(self, file_paths, labels, patient_ids, transform=None, cache_dir=None):
        self.file_paths = file_paths
        self.labels = labels
        self.patient_ids = patient_ids
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def load_image(self, img_path):
        if self.cache_dir is None:
            with Image.open(img_path) as source:
                return source.convert("L")
        source_path = Path(img_path).resolve()
        stat = source_path.stat()
        key = hashlib.sha256(
            f"gray-v1:{source_path}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}".encode()
        ).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached = self.cache_dir / f"{key}.npy"
        if cached.exists():
            values = np.load(cached, allow_pickle=False)
            if values.dtype != np.uint8 or values.ndim != 2 or not values.size:
                raise ValueError(f"Invalid decoded-image cache: {cached}")
        else:
            with Image.open(source_path) as source:
                values = np.array(source.convert("L"), dtype=np.uint8)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    np.save(stream, values, allow_pickle=False)
                os.replace(temporary, cached)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        return Image.fromarray(values)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = self.load_image(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class DeviceAugmentation:
    """Tensor augmentation after transfer; same parameter ranges as the CPU profiles."""

    def __init__(self, profile, mean, std):
        if profile not in {"standard", "standard_flip", "mild"}:
            raise ValueError("Unknown device augmentation profile")
        self.profile = profile
        self.mean, self.std = normalization_values((mean, std))

    def __call__(self, images):
        if images.ndim != 4 or images.shape[1] != 1 or not images.shape[0]:
            raise ValueError("Device augmentation requires a nonempty [B, 1, H, W] tensor")
        augmented = []
        mild = self.profile == "mild"
        degrees, translation, scale_range = (5, 0.02, (0.97, 1.03)) if mild else (8, 0.04, (0.95, 1.05))
        for image in images:
            if self.profile == "standard_flip" and torch.rand((), device=image.device).item() < 0.5:
                image = TF.hflip(image)
            params = transforms.RandomAffine.get_params(
                [-degrees, degrees], [translation, translation], list(scale_range), None,
                [image.shape[-1], image.shape[-2]],
            )
            image = TF.affine(image, *params, interpolation=transforms.InterpolationMode.NEAREST, fill=0)
            if mild:
                order, brightness, contrast, _, _ = transforms.ColorJitter.get_params(
                    [0.92, 1.08], [0.90, 1.10], None, None
                )
                for operation in order:
                    if operation == 0:
                        image = TF.adjust_brightness(image, brightness)
                    elif operation == 1:
                        image = TF.adjust_contrast(image, contrast)
            augmented.append(image)
        return TF.normalize(torch.stack(augmented), [self.mean], [self.std])


def extract_patient_id(file_path):
    filename = os.path.basename(file_path)
    stem, _ = os.path.splitext(filename)
    parts = stem.split('_')
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(
            f"Expected an image filename in patient_slice format, got {filename!r}"
        )
    return parts[0]


def load_split_files(split_dir):
    class_map = {"NC": 0, "AD": 1}  
    patient_files = {}
    
    for class_name, label in class_map.items():
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Expected class directory does not exist: {class_dir}")
        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(class_dir, filename)
                patient_id = extract_patient_id(filename)
                if patient_id not in patient_files:
                    patient_files[patient_id] = {"label": label, "files": []}
                elif patient_files[patient_id]["label"] != label:
                    raise ValueError(
                        f"Patient ID {patient_id!r} has files in both classes under {split_dir}. "
                        "Resolve longitudinal or conflicting labels before training."
                    )
                patient_files[patient_id]["files"].append((file_path, label))
    if not patient_files:
        raise ValueError(f"No supported images found under {split_dir}")
    return patient_files


def seed_data_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def slice_positions(file_paths):
    positions = []
    for path in file_paths:
        extract_patient_id(path)
        positions.append(int(os.path.splitext(os.path.basename(path))[0].split('_')[1]))
    return np.asarray(positions)


def training_position_bins(file_paths, bins=5):
    if bins < 1:
        raise ValueError("Position bin count must be positive")
    positions = slice_positions(file_paths)
    if not len(positions):
        raise ValueError("Cannot bin an empty training dataset")
    cuts = np.unique(np.quantile(positions, np.linspace(0, 1, bins + 1)[1:-1]))
    return cuts, np.searchsorted(cuts, positions, side="right")


def position_balanced_sample_weights(dataset, bins=5):
    labels = np.asarray(dataset.labels)
    if len(labels) != len(dataset.file_paths) or set(labels) != {0, 1}:
        raise ValueError("Position balancing requires matching paths and NC/AD labels")
    cuts, assignments = training_position_bins(dataset.file_paths, bins)
    weights = np.empty(len(labels), dtype=np.float64)
    rows = []
    for index in range(len(cuts) + 1):
        selected = assignments == index
        count = int(selected.sum())
        counts = [int(np.sum(selected & (labels == label))) for label in (0, 1)]
        row = {
            "lower_inclusive": float(cuts[index - 1]) if index else None,
            "upper_exclusive": float(cuts[index]) if index < len(cuts) else None,
            "n_slices": count,
            "NC": counts[0],
            "AD": counts[1],
        }
        rows.append(row)
        if not count:
            continue
        if 0 in counts:
            raise ValueError(
                f"Position bin {row} is missing a class; cannot balance it. "
                "Use fewer bins based on training data, not validation/test scores."
            )
        for label in (0, 1):
            weights[selected & (labels == label)] = count / (2 * counts[label])
    return torch.as_tensor(weights, dtype=torch.double), {
        "training_bin_cuts": cuts.tolist(),
        "bins": rows,
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
        "policy": "Equal expected NC/AD mass within each bin; preserve bin mass",
    }


SPLIT_PROTOCOLS = ("original", "pooled_70_20_10", "subject_70_15_15")


def pool_scan_files(train_files, test_files):
    pooled = {}
    seen = set()
    for source in (train_files, test_files):
        for patient, record in source.items():
            target = pooled.setdefault(patient, {"label": record["label"], "files": []})
            if target["label"] != record["label"]:
                raise ValueError("Conflicting patient labels across original partitions")
            for path, label in record["files"]:
                key = (patient, os.path.basename(path))
                if key in seen:
                    raise ValueError("Duplicate patient/filename across original partitions")
                seen.add(key)
                target["files"].append((path, label))
    return pooled


def stratified_partitions(pooled, random_state, test_size, validation_size):
    patients = sorted(pooled)
    labels = [pooled[patient]["label"] for patient in patients]
    development, test_ids = train_test_split(
        patients, test_size=test_size, stratify=labels, random_state=random_state
    )
    train_ids, val_ids = train_test_split(
        development, test_size=validation_size,
        stratify=[pooled[patient]["label"] for patient in development],
        random_state=random_state,
    )
    partitions = []
    for ids in (train_ids, val_ids, test_ids):
        selected = {patient: pooled[patient] for patient in sorted(ids)}
        if {record["label"] for record in selected.values()} != {0, 1}:
            raise ValueError("Every pooled partition must contain both classes")
        partitions.append(selected)
    return partitions


def pooled_patient_partitions(train_files, test_files, random_state):
    return stratified_partitions(
        pool_scan_files(train_files, test_files), random_state, 0.10, 2 / 9
    )


def subject_partitions(train_files, test_files, random_state, metadata):
    from utils import resolve_subject

    subjects = {}
    for scan, record in pool_scan_files(train_files, test_files).items():
        subject = resolve_subject(metadata, scan, record["label"])
        target = subjects.setdefault(subject, {"label": record["label"], "files": []})
        if target["label"] != record["label"]:
            raise ValueError("Conflicting subject labels across visits")
        target["files"].extend(record["files"])
    held_out_count = (len(subjects) * 15 + 99) // 100
    return stratified_partitions(subjects, random_state, held_out_count, held_out_count)


def assemble_paths(patient_files, patient_ids):
    paths, labels, assembled_patient_ids = [], [], []
    for patient_id in patient_ids:
        for path, label in patient_files[patient_id]["files"]:
            paths.append(path)
            labels.append(label)
            assembled_patient_ids.append(patient_id)
    return paths, labels, assembled_patient_ids


def split_paths(base_dir, val_ratio, random_state, split_protocol, metadata_path=None):
    if split_protocol not in SPLIT_PROTOCOLS:
        raise ValueError(f"Unknown split protocol: {split_protocol}")
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    train_patient_files = load_split_files(train_dir)
    test_patient_files = load_split_files(test_dir)
    if {record["label"] for record in train_patient_files.values()} != {0, 1}:
        raise ValueError("Both NC and AD patients must be present in training")
    if {record["label"] for record in test_patient_files.values()} != {0, 1}:
        raise ValueError("Both NC and AD patients must be present in test")

    if split_protocol == "subject_70_15_15":
        if metadata_path is None:
            raise ValueError("Subject splitting requires an explicit metadata_path")
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        partitions = subject_partitions(
            train_patient_files, test_patient_files, random_state, metadata
        )
        return tuple(assemble_paths(records, list(records)) for records in partitions)
    if metadata_path is not None:
        raise ValueError("metadata_path requires the subject_70_15_15 protocol")
    if split_protocol == "pooled_70_20_10":
        partitions = pooled_patient_partitions(
            train_patient_files, test_patient_files, random_state
        )
        return tuple(assemble_paths(records, list(records)) for records in partitions)

    overlapping_patients = set(train_patient_files).intersection(test_patient_files)
    if overlapping_patients:
        raise ValueError(
            f"Patient-level leakage detected: {len(overlapping_patients)} patient IDs "
            f"occur in both train and test (examples: "
            f"{', '.join(sorted(overlapping_patients)[:5])})."
        )
    
    train_patient_ids = list(train_patient_files)
    all_train_paths, all_train_labels, all_train_patient_ids = assemble_paths(
        train_patient_files, train_patient_ids
    )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    patient_counts_by_class = {}
    for patient_id, record in train_patient_files.items():
        label = record["label"]
        patient_counts_by_class[label] = patient_counts_by_class.get(label, 0) + 1
    min_class_patients = min(patient_counts_by_class.values())
    n_splits = min(max(2, round(1.0 / val_ratio)), min_class_patients)
    if n_splits < 2:
        raise ValueError(
            "At least two training patients per class are needed for a "
            "stratified patient-group validation split"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    train_indices, val_indices = next(
        splitter.split(
            all_train_paths,
            all_train_labels,
            groups=all_train_patient_ids,
        )
    )
    train_indices = train_indices.tolist()
    val_indices = val_indices.tolist()
    tr_paths = [all_train_paths[index] for index in train_indices]
    tr_labels = [all_train_labels[index] for index in train_indices]
    tr_patient_ids = [all_train_patient_ids[index] for index in train_indices]
    val_paths = [all_train_paths[index] for index in val_indices]
    val_labels = [all_train_labels[index] for index in val_indices]
    val_patient_ids = [all_train_patient_ids[index] for index in val_indices]
    if set(tr_patient_ids).intersection(val_patient_ids):
        raise RuntimeError("Patient leakage detected between train and validation")

    test_patient_ids = list(test_patient_files.keys())
    test_paths, test_labels, test_patient_ids = assemble_paths(
        test_patient_files, test_patient_ids
    )
    return ((tr_paths, tr_labels, tr_patient_ids),
            (val_paths, val_labels, val_patient_ids),
            (test_paths, test_labels, test_patient_ids))


def split_manifest(loaders, base_dir, protocol, seed):
    root = Path(base_dir).resolve()
    splits = {}
    for name, loader in zip(("train", "validation", "test"), loaders):
        dataset = loader.dataset
        splits[name] = [
            {"path": Path(path).resolve().relative_to(root).as_posix(),
             "label": int(label), "patient_id": patient}
            for path, label, patient in zip(
                dataset.file_paths, dataset.labels, dataset.patient_ids
            )
        ]
    result = {"protocol": protocol, "seed": seed, "splits": splits}
    if protocol == "subject_70_15_15":
        result["identity_metadata_sha256"] = loaders[0].dataset.identity_metadata_sha256
    return result


def compute_channel_stats(file_paths, img_size):
    if not file_paths or img_size < 1:
        raise ValueError("Channel statistics require training images and a positive image size")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.ToTensor(),
    ])
    count, total, squared = 0, 0.0, 0.0
    for path in file_paths:
        with Image.open(path) as source:
            values = transform(source.convert("L")).to(dtype=torch.float64)
        count += values.numel()
        total += values.sum().item()
        squared += values.square().sum().item()
    mean = total / count
    std = float(np.sqrt(max(0.0, squared / count - mean * mean)))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Training images must have nonzero finite pixel standard deviation")
    return {
        "mean": mean, "std": std, "images": len(file_paths), "pixels": count,
        "img_size": img_size,
        "source": "Training split only; grayscale, deterministic resize, ToTensor; before augmentation",
        "weighting": "Population pixel mean/std over all resized training images, including background",
    }


def normalization_values(values):
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError("Normalization requires a (mean, std) pair")
    mean, std = map(float, values)
    if not np.isfinite([mean, std]).all() or not 0 <= mean <= 1 or std <= 0:
        raise ValueError("Normalization needs finite mean in [0, 1] and positive std")
    return mean, std


def checkpoint_normalization(checkpoint, img_size=224):
    directory = Path(checkpoint).parent
    config_path = directory / "training_config.json"
    mode = "fixed"
    if config_path.exists():
        mode = json.loads(config_path.read_text(encoding="utf-8")).get("normalization", "fixed")
    if mode not in {"fixed", "training"}:
        raise ValueError("Checkpoint declares an unsupported normalization mode")
    path = directory / "normalization.json"
    if mode == "training" and not path.is_file():
        raise FileNotFoundError("Training-normalized checkpoint requires normalization.json")
    if not path.exists():
        return 0.5, 0.5
    if mode != "training":
        raise ValueError("normalization.json conflicts with checkpoint normalization mode")
    report = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = directory / "split_manifest.json"
    if (report.get("img_size") != img_size or not manifest_path.is_file()
            or report.get("manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest()):
        raise ValueError("Checkpoint normalization image size or split provenance differs")
    return normalization_values((report["mean"], report["std"]))


def verify_checkpoint_split(checkpoint, loaders, base_dir, protocol, seed):
    expected = checkpoint_normalization(checkpoint)
    for loader in loaders:
        actual = getattr(loader.dataset, "normalization_values", (0.5, 0.5))
        if actual != expected:
            raise ValueError("Checkpoint normalization differs from evaluation; load checkpoint_normalization first")
    manifest_path = Path(checkpoint).parent / "split_manifest.json"
    if not manifest_path.exists():
        if protocol != "original":
            raise ValueError("Pooled evaluation requires the training split_manifest.json")
        return
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = split_manifest(loaders, base_dir, protocol, seed)
    if saved != current:
        raise ValueError("Checkpoint split manifest differs from current evaluation protocol/inventory")


def get_adni_dataloaders(
    base_dir="/home/groups/comp3710/ADNI/AD_NC",
    batch_size=32,
    img_size=224,
    val_ratio=0.15,
    random_state=42,
    num_workers=1,
    augmentation="standard",
    split_protocol="original",
    metadata_path=None,
    normalization=(0.5, 0.5),
    cache_dir=None,
    gpu_augmentation=False,
):
    if augmentation not in {"standard", "standard_flip", "mild"}:
        raise ValueError("augmentation must be 'standard', 'standard_flip' or 'mild'")
    train, validation, test = split_paths(
        base_dir, val_ratio, random_state, split_protocol, metadata_path
    )
    tr_paths, tr_labels, tr_patient_ids = train
    val_paths, val_labels, val_patient_ids = validation
    test_paths, test_labels, test_patient_ids = test
    groups = [set(partition[2]) for partition in (train, validation, test)]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Patient leakage between generated partitions")

    statistics = None
    if normalization == "training":
        statistics = compute_channel_stats(tr_paths, img_size)
        normalization = (statistics["mean"], statistics["std"])
    mean, std = normalization_values(normalization)

    if augmentation == "mild":
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.02, 0.02),
                scale=(0.97, 1.03),
            ),
            transforms.ColorJitter(brightness=0.08, contrast=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(
                degrees=8,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ])

    if augmentation == "standard_flip":
        train_transform.transforms.insert(1, transforms.RandomHorizontalFlip(p=0.5))

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])
    if gpu_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), transforms.ToTensor(),
        ])

    train_dataset = ADNIDataset(
        tr_paths, tr_labels, tr_patient_ids, transform=train_transform, cache_dir=cache_dir
    )
    val_dataset = ADNIDataset(
        val_paths, val_labels, val_patient_ids, transform=eval_transform, cache_dir=cache_dir
    )
    test_dataset = ADNIDataset(
        test_paths, test_labels, test_patient_ids, transform=eval_transform, cache_dir=cache_dir
    )
    if gpu_augmentation:
        train_dataset.device_augmentation = DeviceAugmentation(augmentation, mean, std)
    for dataset in (train_dataset, val_dataset, test_dataset):
        dataset.normalization_values = (mean, std)
        dataset.channel_statistics = statistics
    if split_protocol == "subject_70_15_15":
        digest = hashlib.sha256(Path(metadata_path).read_bytes()).hexdigest()
        for dataset in (train_dataset, val_dataset, test_dataset):
            dataset.identity_metadata_sha256 = digest

    generator = torch.Generator()
    generator.manual_seed(random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_data_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader