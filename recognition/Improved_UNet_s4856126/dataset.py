
"""HipMRI 2D MRI/segmentation dataset for COMP3710."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ==========================================
# 1. 数据路径和目标标签
# ==========================================

DATA_ROOT = Path(
    "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
)

# 统一所有 MRI 和 Mask 的尺寸
TARGET_SIZE = (256, 128)

# 数据集共有六个标签：0, 1, 2, 3, 4, 5
NUM_CLASSES = 6

# 尚未确认哪个标签对应 Prostate
# 确认后再设置正确编号
PROSTATE_LABEL = None


# ==========================================
# 2. PyTorch Dataset
# ==========================================

class HipMRIDataset(Dataset):

    def __init__(
        self,
        split="train",
        data_root=DATA_ROOT,
        binary_segmentation=False,
        target_label=None,
        augment=False,
    ):

        # 检查数据集划分
        if split not in ("train", "validate", "test"):
            raise ValueError("Invalid dataset split")

        # 二分类必须指定目标标签
        if binary_segmentation and target_label is None:
            raise ValueError(
                "Binary segmentation requires target_label"
            )

        if target_label is not None and not binary_segmentation:
            raise ValueError(
                "Set binary_segmentation=True to use target_label"
            )

        if target_label is not None and (
            not isinstance(target_label, int)
            or not 0 <= target_label < NUM_CLASSES
        ):
            raise ValueError("Invalid target_label")

        # 数据增强只能用于训练集
        if augment and split != "train":
            raise ValueError(
                "Augmentation is only allowed for training"
            )

        self.split = split
        self.binary_segmentation = binary_segmentation
        self.target_label = target_label
        self.augment = augment

        data_root = Path(data_root)

        # MRI 图像目录
        image_dir = data_root / f"keras_slices_{split}"

        # Ground Truth Mask 目录
        mask_dir = data_root / f"keras_slices_seg_{split}"

        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(
                f"Data directories not found: {image_dir}, {mask_dir}"
            )

        # ==================================
        # 3. MRI 和 Mask 自动配对
        # ==================================

        self.pairs = []

        for image_path in sorted(
            image_dir.glob("case_*.nii.gz")
        ):

            mask_name = image_path.name.replace(
                "case_", "seg_", 1
            )

            mask_path = mask_dir / mask_name

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing mask: {mask_path}"
                )

            self.pairs.append((image_path, mask_path))

        if not self.pairs:
            raise ValueError("No MRI files found")

        # 确认没有多余的 Mask
        expected = {
            mask_path.name
            for _, mask_path in self.pairs
        }

        actual = {
            path.name
            for path in mask_dir.glob("seg_*.nii.gz")
        }

        if expected != actual:
            raise ValueError(
                "MRI and Mask file lists do not match"
            )

    # ======================================
    # 4. 返回数据集大小
    # ======================================

    def __len__(self):
        return len(self.pairs)

    # ======================================
    # 5. 读取单张 MRI 和 Mask
    # ======================================

    def __getitem__(self, index):

        image_path, mask_path = self.pairs[index]

        # 使用 nibabel 读取医学影像
        image = nib.load(image_path).get_fdata(
            dtype=np.float32
        ).squeeze()

        mask = nib.load(mask_path).get_fdata(
            dtype=np.float32
        ).squeeze()

        # 确认是二维图像
        if image.ndim != 2 or mask.ndim != 2:
            raise ValueError(
                f"Expected 2D arrays: "
                f"{image.shape}, {mask.shape}"
            )

        # MRI 和 Mask 原始尺寸必须相同
        if image.shape != mask.shape:
            raise ValueError(
                f"MRI and Mask shapes do not match: "
                f"{image_path.name}: {image.shape}, "
                f"{mask_path.name}: {mask.shape}"
            )

        # 检查无效数值
        if not np.isfinite(image).all():
            raise ValueError(
                f"Invalid MRI values: {image_path.name}"
            )

        if not np.isfinite(mask).all():
            raise ValueError(
                f"Invalid Mask values: {mask_path.name}"
            )

        # Mask 必须是整数类别
        if not np.array_equal(mask, np.rint(mask)):
            raise ValueError(
                f"Mask contains non-integer labels: "
                f"{mask_path.name}"
            )

        # 转换 Mask 为整数类别
        mask = mask.astype(np.int64)

        # 确保标签在 0~5 范围内
        if mask.min() < 0 or mask.max() >= NUM_CLASSES:
            raise ValueError(
                f"Invalid segmentation labels in "
                f"{mask_path.name}: "
                f"{np.unique(mask).tolist()}"
            )

        # ==================================
        # 6. Resize MRI and Mask
        # ==================================

        # 所有样本统一为 256 x 128
        # 解决 DataLoader 中不同尺寸无法 stack 的问题

        if image.shape != TARGET_SIZE:

            # MRI 使用 Bilinear Interpolation
            # 连续图像适合双线性插值

            image_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    image, dtype=np.float32
                )
            )

            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.unsqueeze(0)

            image_tensor = F.interpolate(
                image_tensor,
                size=TARGET_SIZE,
                mode="bilinear",
                align_corners=False
            )

            image = image_tensor[0, 0].numpy()

            # Mask 使用 Nearest Neighbor
            # 避免创建 0~5 以外的类别

            mask_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    mask, dtype=np.int64
                )
            )

            mask_tensor = mask_tensor.unsqueeze(0)
            mask_tensor = mask_tensor.unsqueeze(0)

            mask_tensor = F.interpolate(
                mask_tensor.float(),
                size=TARGET_SIZE,
                mode="nearest"
            )

            mask = mask_tensor[0, 0].long().numpy()

        # ==================================
        # 7. MRI Normalization
        # ==================================

        # Z-score 标准化
        mean = image.mean()
        std = image.std()

        image = (image - mean) / (std + 1e-6)

        # ==================================
        # 8. Data Augmentation
        # ==================================

        # 仅用于 Training
        # 随机改变 MRI 的亮度和对比度
        # 不改变 Mask，也不移动解剖结构

        if self.augment and np.random.random() < 0.5:

            scale = np.random.uniform(0.9, 1.1)
            shift = np.random.uniform(-0.1, 0.1)

            image = image * scale + shift

        # ==================================
        # 9. 转换为 PyTorch Tensor
        # ==================================

        # MRI: [H, W] -> [1, H, W]
        image = torch.from_numpy(
            np.ascontiguousarray(
                image[None, :, :],
                dtype=np.float32
            )
        )

        # ==================================
        # 10. 处理 Ground Truth Mask
        # ==================================

        if self.binary_segmentation:

            # 将指定目标标签转换为 1
            # 其他类别全部转换为 0

            binary_mask = (
                mask == self.target_label
            ).astype(np.float32)

            # Binary Mask: [1, H, W]
            mask_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    binary_mask[None, :, :]
                )
            )

        else:

            # 保留所有原始类别 0、1、2、3、4、5
            # Multi-class Mask: [H, W]

            mask_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    mask, dtype=np.int64
                )
            )

        return image, mask_tensor


# ==========================================
# 11. 检查患者数据是否重叠
# ==========================================

def check_patient_splits(*datasets):

    case_sets = []

    for dataset in datasets:

        # 从 case_004_week_0_slice_10 中提取 004
        case_ids = {
            path.name.split("_")[1]
            for path, _ in dataset.pairs
        }

        case_sets.append(case_ids)

    # 检查不同 split 是否共享同一个 case
    for i in range(len(case_sets)):
        for j in range(i + 1, len(case_sets)):

            overlap = case_sets[i] & case_sets[j]

            if overlap:
                raise ValueError(
                    f"Patient/case leakage detected: {overlap}"
                )

    return [
        len(case_ids)
        for case_ids in case_sets
    ]


# ==========================================
# 12. 测试 Dataset 和 DataLoader
# ==========================================

if __name__ == "__main__":

    # 未确认前列腺标签，先测试多类别读取
    datasets = {
        split: HipMRIDataset(
            split,
            binary_segmentation=False
        )
        for split in ("train", "validate", "test")
    }

    for split, dataset in datasets.items():

        image, mask = dataset[0]

        print(f"\n===== {split.upper()} =====")
        print("Samples:", len(dataset))

        print("MRI shape:", tuple(image.shape))
        print("MRI dtype:", image.dtype)

        print("Mask shape:", tuple(mask.shape))
        print("Mask dtype:", mask.dtype)

        print(
            "Mask labels:",
            torch.unique(mask).tolist()
        )

        print(
            "MRI mean/std:",
            round(image.mean().item(), 4),
            round(image.std(unbiased=False).item(), 4)
        )

    # ==================================
    # 13. 检查 Patient-level Split
    # ==================================

    counts = check_patient_splits(
        *datasets.values()
    )

    print(
        "\nDisjoint case counts:",
        counts
    )

    # ==================================
    # 14. 测试 DataLoader
    # ==================================

    train_loader = DataLoader(
        datasets["train"],
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    batch_images, batch_masks = next(
        iter(train_loader)
    )

    print(
        "Batch MRI:",
        tuple(batch_images.shape)
    )

    print(
        "Batch Mask:",
        tuple(batch_masks.shape)
    )

    print(
        "Batch labels:",
        torch.unique(batch_masks).tolist()
    )

    # 检查 Batch 尺寸
    assert batch_images.shape == (
        8, 1, 256, 128
    ), "Unexpected MRI batch shape"

    assert batch_masks.shape == (
        8, 256, 128
    ), "Unexpected Mask batch shape"

    assert batch_masks.dtype == torch.int64

    print("Batch loading successful!")

    # ==================================
    # 15. 二分类测试
    # ==================================

    # 只有确认 Prostate 对应的标签后才执行
    if PROSTATE_LABEL is not None:

        binary_data = HipMRIDataset(
            "train",
            binary_segmentation=True,
            target_label=PROSTATE_LABEL
        )

        binary_image, binary_mask = binary_data[0]

        print(
            "Binary MRI:",
            tuple(binary_image.shape)
        )

        print(
            "Binary Mask:",
            tuple(binary_mask.shape)
        )

        print(
            "Binary labels:",
            torch.unique(binary_mask).tolist()
        )

    else:
        print(
            "Prostate label unverified: "
            "binary conversion not run."
        )

    print("\nDataset loading test completed!")
