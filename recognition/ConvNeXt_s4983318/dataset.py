"""COMP3710 ADNI data pipeline: audit, patient-level splitting and image loading.

Stage 1 validates images and creates the three CSV manifests (already run
locally). Stage 2 reads those manifests in PyTorch without repeating the audit.
No model is trained by this script.

Split unit: patient, stratified by diagnosis, approximately 70/15/15.
Training labels: NC=0, AD=1. Original metadata labels: NC=0, AD=2.
Only images actually present in data_root are included. Source files are
read only; the original train/test directories remain unchanged.

Pillow image-decoding API reference:
https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.load

AI assistance: ChatGPT assisted with stage-1 audit and stage-2 loader
scaffolding. Independently verify local outputs and document AI use in README.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import PurePosixPath
from collections import Counter
from itertools import combinations
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image, ImageOps, __version__ as pillow_version
except ImportError:
    raise SystemExit("Pillow is required. Run: python -m pip install Pillow")


CLASS_NAMES = {0: "NC", 1: "AD"}
METADATA_LABELS = {"NC": 0, "AD": 2}
TRAINING_LABELS = {"NC": 0, "AD": 1}
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
PATIENT_PATTERN = re.compile(r"ADNI_(\d{3}_S_\d{4,})_")
FIELDS = [
    "relative_path", "patient_id", "scan_id", "slice_index", "label",
    "metadata_label", "class_name", "original_split", "width", "height", "mode",
]


def load_metadata(path):
    """Read JSON, rejecting duplicate scan keys instead of silently overwriting."""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    metadata = json.loads(path.read_text(encoding="utf-8-sig"),
                          object_pairs_hook=unique_keys)
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("Metadata must be a non-empty dictionary of scans.")
    return metadata


def build_image_index(data_root, metadata, expected_images=None):
    """Match each actual image to a scan, patient and diagnosis; fully decode it."""
    for split in ("train", "test"):
        for name in ("AD", "NC"):
            if not (data_root / split / name).is_dir():
                raise ValueError(f"Missing source directory: {data_root / split / name}")

    paths = sorted(
        (p for p in data_root.rglob("*")
         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.relative_to(data_root).as_posix(),
    )
    if not paths:
        raise ValueError("No images found. Check --data-root.")
    if expected_images is not None and len(paths) != expected_images:
        raise ValueError(f"Found {len(paths)} images; expected {expected_images}.")

    print(f"Checking metadata and decoding {len(paths)} images...", flush=True)
    rows, seen_slices, patient_labels = [], {}, {}
    for i, path in enumerate(paths, start=1):
        relative = path.relative_to(data_root)
        if (len(relative.parts) < 3 or relative.parts[0] not in ("train", "test")
                or relative.parts[1] not in METADATA_LABELS):
            raise ValueError(f"Unexpected image location: {relative}")
        original_split, class_name = relative.parts[:2]
        match = re.fullmatch(r"(\d+)_(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Expected scanID_sliceID filename: {relative}")
        scan_id, slice_index = match.group(1), int(match.group(2))
        record = metadata.get(scan_id)
        if not isinstance(record, dict):
            raise ValueError(f"No metadata record for image: {relative}")
        if record.get("label") != METADATA_LABELS[class_name]:
            raise ValueError(f"Folder/metadata label mismatch: {relative}")
        patient_match = PATIENT_PATTERN.search(str(record.get("raw", "")))
        if patient_match is None:
            raise ValueError(f"Cannot parse patient ID for scan: {scan_id}")
        patient_id = patient_match.group(1)
        label = TRAINING_LABELS[class_name]
        if patient_id in patient_labels and patient_labels[patient_id] != label:
            raise ValueError(f"Patient has conflicting diagnoses: {patient_id}")
        patient_labels[patient_id] = label

        # The same slice copied to two directories must not become two samples.
        slice_key = (scan_id, slice_index)
        if slice_key in seen_slices:
            raise ValueError(f"Duplicate slice: {relative} and {seen_slices[slice_key]}")
        seen_slices[slice_key] = relative.as_posix()

        try:
            with Image.open(path) as image:
                image.load()  # Decode pixels: opening a header alone is insufficient.
                width, height = image.size
                mode = image.mode
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot decode image {relative}: {error}") from error

        rows.append(dict(zip(FIELDS, [
            relative.as_posix(), patient_id, scan_id, slice_index, label,
            record["label"], class_name, original_split, width, height, mode,
        ])))
        if i % 2000 == 0 or i == len(paths):
            print(f"  Decoded {i}/{len(paths)}", flush=True)
    return rows


def make_patient_splits(rows, seed=42):
    """Shuffle sorted patients within each class, then assign all their images."""
    patient_labels = {}
    for row in rows:
        patient, label = row["patient_id"], row["label"]
        if patient in patient_labels and patient_labels[patient] != label:
            raise ValueError(f"Patient has conflicting diagnoses: {patient}")
        patient_labels[patient] = label

    rng = random.Random(seed)
    assignment = {}
    for label in (0, 1):
        patients = sorted(p for p, y in patient_labels.items() if y == label)
        n_val = n_test = round(len(patients) * 0.15)
        if min(n_val, n_test, len(patients) - n_val - n_test) < 1:
            raise ValueError(f"Too few {CLASS_NAMES[label]} patients for three splits.")
        rng.shuffle(patients)
        for i, patient in enumerate(patients):
            if i < n_test:
                assignment[patient] = "test"
            elif i < n_test + n_val:
                assignment[patient] = "val"
            else:
                assignment[patient] = "train"

    splits = {name: [] for name in ("train", "val", "test")}
    for row in rows:
        splits[assignment[row["patient_id"]]].append(row)
    return splits


def validate_splits(rows, splits):
    """Require complete image coverage and zero patient/scan overlap."""
    assigned = [row["relative_path"] for part in splits.values() for row in part]
    original = [row["relative_path"] for row in rows]
    if len(set(assigned)) != len(assigned) or sorted(assigned) != sorted(original):
        raise ValueError("Each input image must be assigned exactly once.")
    overlaps = {}
    for name, part in splits.items():
        if {row["label"] for row in part} != {0, 1}:
            raise ValueError(f"Split {name} must contain both classes.")
    for left, right in combinations(splits, 2):
        pair = f"{left}-{right}"
        overlaps[pair] = {}
        for field in ("patient_id", "scan_id"):
            common = ({r[field] for r in splits[left]}
                      & {r[field] for r in splits[right]})
            overlaps[pair][field] = len(common)
            if common:
                raise ValueError(f"{field} leakage in {pair}: {len(common)}")
    return overlaps


def summarize(rows):
    """Count images, scans and patients separately, including per-class counts."""
    def counts(part):
        return {"images": len(part),
                "scans": len({r["scan_id"] for r in part}),
                "patients": len({r["patient_id"] for r in part})}
    result = counts(rows)
    result["by_class"] = {
        name: counts([r for r in rows if r["label"] == label])
        for label, name in CLASS_NAMES.items()
    }
    return result


# -------------------- Stage 2: PyTorch data loading --------------------
# CNN and ConvNeXt will use the SAME manifests and input preprocessing.
# The scan's original "train"/"test" folder is NOT its new split.

def load_manifest(manifest_path, data_root):
    """Read a stage-1 manifest and validate labels, paths and patient IDs.

    Relative paths point to the ORIGINAL JPEG locations; no copies are made.
    """
    manifest_path = Path(manifest_path)
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise ValueError(f"Data directory not found: {data_root}")
    if not manifest_path.is_file():
        raise ValueError(f"Manifest missing: {manifest_path}")

    required = {"relative_path", "patient_id", "scan_id", "slice_index",
                "label", "class_name"}
    records = []
    seen_paths = set()
    patient_labels = {}
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected CSV columns in {manifest_path}")
        for line_number, row in enumerate(reader, start=2):
            try:
                rel = PurePosixPath(row["relative_path"])
                patient = row["patient_id"]
                scan = row["scan_id"]
                label = int(row["label"])
                name = row["class_name"]
                slice_index = int(row["slice_index"])
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid CSV entry on line {line_number}") from exc
            if (rel.is_absolute() or ".." in rel.parts or "\\" in str(rel)
                    or len(rel.parts) < 3):
                raise ValueError(f"Unsafe relative path on line {line_number}: {rel}")
            if not patient or not scan or label not in CLASS_NAMES:
                raise ValueError(f"Invalid patient, scan or label on line {line_number}")
            if CLASS_NAMES[label] != name:
                raise ValueError(f"Label/class mismatch on line {line_number}")
            if rel.as_posix() in seen_paths:
                raise ValueError(f"Repeated image path on line {line_number}: {rel}")
            seen_paths.add(rel.as_posix())
            if patient in patient_labels and patient_labels[patient] != label:
                raise ValueError(f"Conflicting diagnoses for {patient}")
            patient_labels[patient] = label
            records.append({
                "relative_path": rel.as_posix(), "patient_id": patient,
                "scan_id": scan, "slice_index": slice_index,
                "label": label, "class_name": name,
            })
    if not records:
        raise ValueError(f"Empty manifest: {manifest_path}")
    return records


def validate_manifest_separation(manifests):
    """Fail if patients, scans OR image paths occur in multiple splits."""
    for left, right in combinations(manifests, 2):
        for field in ("patient_id", "scan_id", "relative_path"):
            shared = ({record[field] for record in manifests[left]}
                      & {record[field] for record in manifests[right]})
            if shared:
                raise ValueError(
                    f"Data leakage between {left} and {right}: "
                    f"{len(shared)} overlapping {field} values"
                )


def create_transforms(image_size=224, train=False, augment=False):
    """MRI: preserve aspect ratio, pad to square, then ImageNet normalization.

    Grayscale is converted to three identical RGB channels because standard
    (pretrained) ConvNeXt expects three input channels. The baseline will use
    the same input so comparisons are fair. Random rotation is opt-in and
    restricted to training; no left-right flip or random validation transforms.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    try:
        from torchvision import transforms
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "A torchvision installation compatible with PyTorch is required. "
            "Check versions using python -c \"import torch, torchvision; "
            "print(torch.__version__, torchvision.__version__)\""
        ) from exc
    steps = []
    if train and augment:
        # A conservative augmentation for later experiments, disabled by default.
        steps.append(transforms.RandomRotation(degrees=5, fill=0))
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(steps)


class ADNISliceDataset(Dataset):
    """Read MRI slices on demand from an existing stage-1 CSV manifest.

    Each sample contains image [3, H, W], label 0=NC/1=AD, patient ID,
    scan ID and source-relative path (used for later error analysis).
    """

    def __init__(self, data_root, manifest_path, image_size=224,
                 train=False, augment=False, records=None):
        self.data_root = Path(data_root).expanduser().resolve()
        self.image_size = image_size
        self.records = (load_manifest(manifest_path, self.data_root)
                        if records is None else records)
        self.transform = create_transforms(image_size, train, augment)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.data_root / Path(record["relative_path"])
        try:
            with Image.open(path) as image:
                # Black padding preserves the entire brain field of view.
                # Unlike direct square resizing, it does not distort anatomy.
                image = ImageOps.pad(image.convert("L"),
                                     (self.image_size, self.image_size),
                                     method=Image.Resampling.BILINEAR, color=0)
                image = image.convert("RGB")  # ConvNeXt expects three channels.
                tensor = self.transform(image)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read MRI slice {path}: {exc}") from exc
        return {"image": tensor,
                "label": record["label"],
                "patient_id": record["patient_id"],
                "scan_id": record["scan_id"],
                "relative_path": record["relative_path"]}


def make_dataloaders(data_root, manifest_dir, batch_size=8,
                     image_size=224, num_workers=0, seed=42, augment_train=False):
    """Create loaders from the EXISTING CSVs; do NOT regenerate splits.

    Returns ({'train','val','test'} -> DataLoader). With num_workers=0 it is
    Windows-friendly and makes initial debugging simpler.
    """
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive; num_workers non-negative")
    root = Path(data_root).expanduser().resolve()
    folder = Path(manifest_dir).expanduser().resolve()
    manifests = {part: load_manifest(folder / f"{part}.csv", root)
                 for part in ("train", "val", "test")}
    validate_manifest_separation(manifests)
    generator = torch.Generator().manual_seed(seed)
    return {
        part: DataLoader(
            ADNISliceDataset(root, folder / f"{part}.csv", image_size=image_size,
                         train=(part == "train"), augment=augment_train,
                         records=manifests[part]),
            batch_size=batch_size,
            shuffle=(part == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if part == "train" else None,
        )
        for part in ("train", "val", "test")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-images", type=int, default=None)
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.metadata = args.metadata.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is not empty. Existing splits are preserved; "
                         "use a new empty directory if you intend to prepare a new version.")

    metadata = load_metadata(args.metadata)
    rows = build_image_index(args.data_root, metadata, args.expected_images)
    splits = make_patient_splits(rows, args.seed)
    overlaps = validate_splits(rows, splits)

    original = {name: [r for r in rows if r["original_split"] == name]
                for name in ("train", "test")}
    old_patient_overlap = len({r["patient_id"] for r in original["train"]}
                              & {r["patient_id"] for r in original["test"]})
    old_scan_overlap = len({r["scan_id"] for r in original["train"]}
                           & {r["scan_id"] for r in original["test"]})
    used_scans = {r["scan_id"] for r in rows}
    unused = {name: sum(record.get("label") == label and key not in used_scans
                        for key, record in metadata.items())
              for name, label in METADATA_LABELS.items()}
    shapes = Counter(f'{r["width"]}x{r["height"]} {r["mode"]}' for r in rows)
    report = {
        "stage": "data_audit_and_patient_split", "seed": args.seed,
        "target_patient_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "training_label_map": TRAINING_LABELS, "metadata_label_map": METADATA_LABELS,
        "python_version": sys.version.split()[0], "pillow_version": pillow_version,
        "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "all_images_decoded": True, "total": summarize(rows),
        "original_folders": {
            f"{split}/{name}": summarize([r for r in original[split] if r["class_name"] == name])
            for split in ("train", "test") for name in ("AD", "NC")
        },
        "original_patient_overlap": old_patient_overlap,
        "original_scan_overlap": old_scan_overlap,
        "unused_binary_metadata_scans": unused,
        "image_sizes_and_modes": dict(sorted(shapes.items())),
        "slices_per_scan_histogram": dict(sorted(Counter(Counter(r["scan_id"] for r in rows).values()).items())),
        "new_splits": {name: summarize(part) for name, part in splits.items()},
        "new_split_overlap": overlaps,
        "note": "Unused metadata records do not by themselves indicate an incomplete download. "
                "Image fractions may differ from patient fractions. "
                "Image decoding is a readability check, not a checksum comparison with the server.",
    }

    # Write manifests only after every audit and split validation has passed.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        with (args.output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(part)
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nSOURCE FOLDERS")
    for name, info in report["original_folders"].items():
        print(f'{name}: images={info["images"]}, scans={info["scans"]}, patients={info["patients"]}')
    print(f"Original train/test overlap: patients={old_patient_overlap}, scans={old_scan_overlap}")
    print(f"Unused binary metadata scans: {unused}")
    print(f"Image sizes and modes: {dict(shapes)}")
    print(f"\nNEW PATIENT SPLITS (seed={args.seed})")
    for name, info in report["new_splits"].items():
        patients = {key: value["patients"] for key, value in info["by_class"].items()}
        print(f'{name}: images={info["images"]}, scans={info["scans"]}, '
              f'patients={info["patients"]}, patients_by_class={patients}')
    for pair, info in overlaps.items():
        print(f'Overlap {pair}: patients={info["patient_id"]}, scans={info["scan_id"]}')
    print(f"\nPASS: all {len(rows)} images decoded and assigned exactly once.")
    print(f"Manifests and audit_summary.json saved to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
