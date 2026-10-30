
from pathlib import Path
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DATA_ROOT = Path("/home/groups/comp3710/HipMRI_Study_open")


def get_data_pairs(root=DATA_ROOT):
    """Find matching MRI and segmentation mask files."""
    mri_dir = root / "semantic_MRs"
    mask_dir = root / "semantic_labels_only"

    pairs = []

    for mri_path in sorted(mri_dir.glob("*_LFOV.nii.gz")):
        prefix = mri_path.name.replace("_LFOV.nii.gz", "")
        mask_path = mask_dir / f"{prefix}_SEMANTIC.nii.gz"

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        patient_id = prefix.split("_")[0]
        pairs.append((patient_id, mri_path, mask_path))

    return pairs
'''
[
    (
        "B006",
        "B006_Week0_LFOV.nii.gz",
        "B006_Week0_SEMANTIC.nii.gz"
    ),
    (
        "B006",
        "B006_Week1_LFOV.nii.gz",
        "B006_Week1_SEMANTIC.nii.gz"
    )
]
'''

# 211 data 38 patients 211 scans were grouped.
# independent from each other to prevent information from the same patient from leaking.
# seed=42 ensures the patient grouping is consistent across repeated runs.
def split_by_patient(pairs, seed=42):
    """Split patients without leaking scans across datasets."""
    patients = sorted({p[0] for p in pairs})

    rng = random.Random(seed)
    rng.shuffle(patients)

    train_ids = set(patients[:26])
    val_ids = set(patients[26:32])
    test_ids = set(patients[32:])

    train = [p for p in pairs if p[0] in train_ids]
    val = [p for p in pairs if p[0] in val_ids]
    test = [p for p in pairs if p[0] in test_ids]

    return train, val, test


def audit_data(pairs):
    """Audit all MRI-mask pairs before training."""
    from collections import Counter

    shape_counts = Counter()
    all_labels = set()
    label_voxel_counts = Counter()
    affine_mismatches = []
    shape_mismatches = []
    unusual_shapes = []
    invalid_labels = []
    unreadable_files = []

    expected_shape = (256, 256, 128)
    allowed_labels = set(range(6))

    for index, (patient_id, mri_path, mask_path) in enumerate(pairs, 1):
        try:
            mri_img = nib.load(str(mri_path))
            mask_img = nib.load(str(mask_path))

            shape_counts[mri_img.shape] += 1

            if mri_img.shape != expected_shape:
                unusual_shapes.append(mri_path.name)

            if mri_img.shape != mask_img.shape:
                shape_mismatches.append(mri_path.name)

            if not np.allclose(mri_img.affine, mask_img.affine):
                affine_mismatches.append(mri_path.name)

            # Inspect actual mask values
            mask_data = np.asanyarray(mask_img.dataobj)
            labels, counts = np.unique(mask_data, return_counts=True)

            for label, count in zip(labels, counts):
                value = float(label)
                all_labels.add(value)
                label_voxel_counts[value] += int(count)

                if value not in allowed_labels:
                    invalid_labels.append((mask_path.name, value))

            # Read the MRI data to detect unreadable files
            mri_data = np.asanyarray(mri_img.dataobj)
            if not np.isfinite(mri_data).all():
                unreadable_files.append(
                    (mri_path.name, "Non-finite MRI values")
                )

        except Exception as error:
            unreadable_files.append((mri_path.name, str(error)))

        if index % 25 == 0 or index == len(pairs):
            print(f"Checked {index}/{len(pairs)} scans", flush=True)

    print("\n===== DATA AUDIT RESULTS =====")
    print("Total MRI-mask pairs:", len(pairs))
    print("Unique patients:", len({p[0] for p in pairs}))
    print("MRI shapes:", dict(shape_counts))
    print("All mask labels:", sorted(all_labels))
    print("Voxel counts per label:", dict(sorted(label_voxel_counts.items())))
    print("Unusual shapes:", unusual_shapes)
    print("Shape mismatches:", shape_mismatches)
    print("Affine mismatches:", affine_mismatches)
    print("Invalid labels:", invalid_labels)
    print("Unreadable/invalid MRI files:", unreadable_files)

    passed = not (
        shape_mismatches
        or affine_mismatches
        or invalid_labels
        or unreadable_files
    )

    print("\nAudit status:", "PASS" if passed else "REVIEW REQUIRED")


# 归一化 MRI	  mean()、std()	        调整灰度值范围，让训练更稳定
# 转换 Tensor  torch.from_numpy()	让 PyTorch 能够处理数据
# 降采样	      F.interpolate()	    统一尺寸，降低 GPU 内存需求
# MRI 是灰度值，需要归一化；Mask 是类别标签，不需要归一化。
# 降采样时，MRI 用 trilinear，Mask 用 nearest，防止破坏标签编号。
class HipMRIDataset(Dataset):
    def __init__(self, pairs, target_shape=(64, 128, 128)):
        self.pairs = pairs
        self.target_shape = target_shape

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        patient_id, mri_path, mask_path = self.pairs[index]

        # Load 3D MRI and mask
        mri_img = nib.load(str(mri_path))
        mask_img = nib.load(str(mask_path))

        if mri_img.shape != mask_img.shape:
            raise ValueError(f"Shape mismatch: {mri_path.name}")

        if not np.allclose(mri_img.affine, mask_img.affine):
            raise ValueError(f"Affine mismatch: {mri_path.name}")

        mri = np.asarray(mri_img.dataobj, dtype=np.float32)
        mask = np.asarray(mask_img.dataobj).astype(np.int64)

        # Normalize MRI using nonzero voxels 归一化
        foreground = mri != 0

        if foreground.any():
            mean = mri[foreground].mean()
            std = mri[foreground].std()

            mri[foreground] = (
                    (mri[foreground] - mean) / max(float(std), 1e-6)
            )

        # NIfTI (X, Y, Z) -> PyTorch (D, H, W)
        mri = np.transpose(mri, (2, 0, 1)).copy()
        mask = np.transpose(mask, (2, 0, 1)).copy()

        # Add batch and channel dimensions
        mri = torch.from_numpy(mri).unsqueeze(0).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

        # Downsample MRI with trilinear interpolation
        mri = F.interpolate(
            mri,
            size=self.target_shape,
            mode="trilinear",
            align_corners=False
        )

        # Downsample mask with nearest-neighbor interpolation
        mask = F.interpolate(
            mask.float(),
            size=self.target_shape,
            mode="nearest"
        )

        # MRI: [1, D, H, W]
        # Mask: [D, H, W]
        return mri.squeeze(0), mask.squeeze(0).squeeze(0).long()


if __name__ == "__main__":
    pairs = get_data_pairs()
    audit_data(pairs)
    train, val, test = split_by_patient(pairs)

    print("Total scans:", len(pairs))
    print("Train scans:", len(train))
    print("Validation scans:", len(val))
    print("Test scans:", len(test))

    train_ids = {p[0] for p in train}
    val_ids = {p[0] for p in val}
    test_ids = {p[0] for p in test}

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)

    print("Patient-level split: OK")

    dataset = HipMRIDataset(train)
    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Mask labels:", torch.unique(mask))

