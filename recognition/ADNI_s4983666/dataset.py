"""Read audited ADNI fold manifests and deterministically prepare MRI slices.

The baseline uses fixed intensity scaling rather than statistics estimated
from the dataset. It does not augment validation images, and the fold loader
never returns calibration or final-test images to the training pipeline.
"""

import argparse
from collections import Counter
import io
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

import adni_splits


def manifest_sha256(splits_dir):
    """Fingerprint the frozen manifest set for checkpoints and run records."""
    return adni_splits.digest((Path(splits_dir) / "COMPLETED.json").read_bytes())


def load_fold(data_root, splits_dir, fold):
    """Fully audit the sources before exposing one development fold.

    Verification is mandatory on every invocation. The source folders' old
    train/test names are provenance only; role membership comes exclusively
    from the newly audited patient manifests.
    """
    data_root, splits_dir = Path(data_root).resolve(), Path(splits_dir).resolve()
    adni_splits.verify(argparse.Namespace(data_root=data_root, output=splits_dir))
    seal_bytes = (splits_dir / "COMPLETED.json").read_bytes()
    seal = adni_splits.read_json(splits_dir / "COMPLETED.json")

    def read_verified(path, reader):
        relative = path.relative_to(splits_dir).as_posix()
        expected = seal["sha256"][relative]
        adni_splits.require(adni_splits.digest(path.read_bytes()) == expected,
                            f"Manifest changed after verification: {relative}")
        value = reader(path)
        adni_splits.require(adni_splits.digest(path.read_bytes()) == expected,
                            f"Manifest changed while being read: {relative}")
        return value

    report = read_verified(splits_dir / "report.json", adni_splits.read_json)
    folds = report["config"]["folds"]
    adni_splits.require(type(fold) is int and 1 <= fold <= folds,
                        f"fold must be an integer between 1 and {folds}.")
    expected_slices = report["config"]["expected_slices"]
    result = {"report": report, "manifest_sha256": adni_splits.digest(seal_bytes)}
    for role in ("train", "early_stop", "val"):
        path = splits_dir / f"fold_{fold:02d}" / f"{role}.csv"
        rows = read_verified(path, lambda value: adni_splits.read_csv(value, adni_splits.FIELDS))
        adni_splits.require(rows and all(row["partition"] == "development" for row in rows),
                            f"Only nonempty development manifests are allowed for {role}.")
        scans = Counter(row["image_id"] for row in rows)
        adni_splits.require(all(count == expected_slices for count in scans.values()),
                            f"The {role} manifest contains an incomplete scan.")
        result[role] = rows
    adni_splits.require((splits_dir / "COMPLETED.json").read_bytes() == seal_bytes,
                        "The completion marker changed while loading the fold.")
    return result


class ADNISliceDataset(Dataset):
    """Load one slice per item with fixed, deterministic grayscale processing.

    ``image_size`` is (height, width). Pixel values are mapped from [0, 255]
    to [-1, 1] with fixed constants; no training, validation, calibration, or
    test population statistics are estimated by this transform.
    """

    def __init__(self, rows, data_root, image_size=(240, 256)):
        self.data_root = Path(data_root).resolve()
        adni_splits.require(self.data_root.is_dir(), f"Missing data directory: {self.data_root}")
        adni_splits.require(len(image_size) == 2 and all(type(n) is int and n > 0 for n in image_size),
                            "image_size must contain positive integer height and width.")
        self.image_size = tuple(image_size)
        self.rows = [dict(row) for row in rows]
        adni_splits.require(self.rows, "A slice dataset cannot be empty.")
        self.paths = []
        seen_paths, seen_resolved_paths, seen_slices = set(), set(), set()
        scan_owners = {}
        for row in self.rows:
            relative = row["relative_path"]
            adni_splits.require(isinstance(relative, str) and bool(relative),
                                "Every slice must have a nonempty relative path.")
            path = Path(relative)
            adni_splits.require(not path.is_absolute() and ".." not in path.parts,
                                f"Slice paths must stay inside the data directory: {relative}")
            resolved = (self.data_root / path).resolve()
            adni_splits.require(resolved.is_relative_to(self.data_root) and resolved.is_file(),
                                f"Slice path is missing or outside the data directory: {relative}")
            adni_splits.require(relative not in seen_paths and resolved not in seen_resolved_paths,
                                f"Duplicate slice path: {relative}")
            patient_id, image_id = row["patient_id"], row["image_id"]
            adni_splits.require(isinstance(patient_id, str) and bool(patient_id)
                                and isinstance(image_id, str) and bool(image_id),
                                "Patient and scan identifiers must be nonempty strings.")
            adni_splits.require(str(row["label"]) in ("0", "1"),
                                f"Binary labels must be NC=0 or AD=1: {relative}")
            label = int(row["label"])
            slice_index = int(row["slice_index"])
            adni_splits.require(str(slice_index) == str(row["slice_index"]) and slice_index >= 0,
                                f"Invalid nonnegative slice index: {relative}")
            scan_key = (image_id, slice_index)
            adni_splits.require(scan_key not in seen_slices,
                                f"Duplicate scan/slice identifier: {scan_key}")
            owner = (patient_id, label)
            adni_splits.require(image_id not in scan_owners or scan_owners[image_id] == owner,
                                f"One scan has inconsistent patients or labels: {image_id}")
            scan_owners[image_id] = owner
            seen_paths.add(relative)
            seen_resolved_paths.add(resolved)
            seen_slices.add(scan_key)
            self.paths.append(resolved)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        content = self.paths[index].read_bytes()
        # The startup audit validates all sources. Check this slice again to
        # detect source changes after that audit without rereading other data.
        if "file_sha256" in row:
            adni_splits.require(adni_splits.digest(content) == row["file_sha256"],
                                f"Source image changed after verification: {row['relative_path']}")
        with Image.open(io.BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("L")
            target_size = (self.image_size[1], self.image_size[0])
            if image.size != target_size:
                image = image.resize(target_size, resample=Image.Resampling.BILINEAR)
            # bytearray supplies writable storage for frombuffer, and to()
            # creates an independent float tensor before the buffer expires.
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            tensor = pixels.to(dtype=torch.float32).reshape(1, *self.image_size)
            tensor = tensor.div(255.0).sub(0.5).div(0.5)
        return {
            "image": tensor,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "patient_id": row["patient_id"],
            "image_id": row["image_id"],
            "slice_index": int(row["slice_index"]),
            "relative_path": row["relative_path"],
        }
