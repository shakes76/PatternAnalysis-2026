"""HipMRI 2D data loading and preprocessing for prostate segmentation.

The supplied HipMRI ``keras_slices_data`` folder already separates train,
validation and test slices by participant case ID. This module keeps those
splits unchanged, checks that no case ID appears in more than one split, and
returns PyTorch tensors ready for a 6-class segmentation model.

Preprocessing policy (fixed before any training):

* Shape: slices wider than 128 columns are centre-cropped to 256 x 128.
  Only the 60 training slices of Case 019 / Week 1 (256 x 144) are affected;
  the crop keeps every bladder, rectum and prostate pixel. Images are never
  resized, so labels are never interpolated.
* Intensity: per-slice z-score, ``(x - mean) / (std + 1e-8)``, as in the
  course loader (Appendix B), then clipped to [-5, 5] as in Isensee et al.
  (2018). No statistics are fitted on any split, so nothing leaks from
  validation or test data.
* Labels: stored values are kept as they are (0 Background, 1 Body, 2 Bones,
  3 Bladder, 4 Rectum, 5 Prostate; CSIRO dataset description) and are also
  returned as a one-hot multi-channel mask.
* Augmentation (training only): the same random rotation/scale and flip are
  applied to the image and mask (nearest-neighbour for the mask), followed by
  gamma augmentation on the image.
* Optional 2.5D input (``context_slices=K``, default 0 = plain 2D): the
  image also holds the K slices on each side of the target slice from the
  same series, giving 2K + 1 channels in slice order with the target slice
  in the middle. At the ends of a series the end slice is repeated. Each
  channel is normalised on its own (as a 2D slice would be), all channels
  share the same augmentation, and the mask is that of the target slice
  only. With K = 0 every sample is exactly as before.

Each sample is a dictionary with ``image`` (2K + 1, 256, 128), ``mask``
one-hot (6, 256, 128), ``label`` integer map (256, 128) and the ``case``,
``week`` and ``slice`` identifiers used to stack predictions into per-series
volumes.
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF

NUM_CLASSES = 6
CLASS_NAMES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
PROSTATE_LABEL = 5
TARGET_HEIGHT = 256
TARGET_WIDTH = 128

# Folder names inside keras_slices_data for each supplied split.
SPLIT_FOLDERS = {
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "validation": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}

# Pair counts recorded by the full dataset audit (21 October 2026).
EXPECTED_PAIRS = {"train": 11460, "validation": 660, "test": 540}
# Slices wider than 128 columns: only Case 019 / Week 1 in the training split.
EXPECTED_CROPPED = {"train": 60, "validation": 0, "test": 0}

IMAGE_PATTERN = re.compile(r"^case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz$")


def find_pairs(data_root, split):
    """Return a sorted list of (image_path, mask_path, case, week, slice).

    Image ``case_XXX_week_W_slice_S.nii.gz`` is paired with mask
    ``seg_XXX_week_W_slice_S.nii.gz`` in the matching mask folder.
    """
    image_folder, mask_folder = SPLIT_FOLDERS[split]
    image_dir = Path(data_root) / image_folder
    mask_dir = Path(data_root) / mask_folder
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing {image_dir} or {mask_dir}")

    pairs = []
    for image_path in image_dir.glob("case_*.nii.gz"):
        match = IMAGE_PATTERN.match(image_path.name)
        if match is None:
            raise ValueError(f"Unexpected image filename: {image_path.name}")
        mask_path = mask_dir / image_path.name.replace("case_", "seg_", 1)
        if not mask_path.is_file():
            raise FileNotFoundError(f"No mask for {image_path.name}")
        case, week, slice_index = match.groups()
        pairs.append((image_path, mask_path, case, int(week), int(slice_index)))

    mask_count = len(list(mask_dir.glob("seg_*.nii.gz")))
    if mask_count != len(pairs):
        raise ValueError(f"{split}: {len(pairs)} images but {mask_count} masks")

    # Sort by case, week and slice so a series is stored in slice order.
    pairs.sort(key=lambda p: (p[2], p[3], p[4]))
    return pairs


def check_split_disjoint(data_root):
    """Check that no filename case ID appears in more than one split.

    Case IDs are the only participant grouping available in the filenames.
    Returns a dictionary mapping each split to its sorted case IDs.
    """
    cases = {split: sorted({p[2] for p in find_pairs(data_root, split)})
             for split in SPLIT_FOLDERS}
    splits = list(cases)
    for i, first in enumerate(splits):
        for second in splits[i + 1:]:
            shared = set(cases[first]) & set(cases[second])
            if shared:
                raise ValueError(f"Case IDs in both {first} and {second}: {sorted(shared)}")
    return cases


def load_nifti_2d(path):
    """Load one Nifti slice as a 2D NumPy array (some HipMRI files carry an extra axis)."""
    array = np.asarray(nib.load(str(path)).dataobj)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2D slice, got shape {array.shape}")
    return array


def centre_crop_width(image, mask, width=TARGET_WIDTH):
    """Centre-crop axis 1 of an image/mask pair to ``width`` columns.

    The same column range is used for the image and the mask, so they stay
    aligned. Pairs that are already ``width`` columns wide are returned as is.
    """
    if image.shape != mask.shape:
        raise ValueError(f"Image shape {image.shape} differs from mask shape {mask.shape}")
    height, current_width = image.shape
    if height != TARGET_HEIGHT or current_width < width:
        raise ValueError(f"Unsupported slice shape {image.shape}")
    start = (current_width - width) // 2
    return image[:, start:start + width], mask[:, start:start + width]


def normalise_slice(image, clip_value=5.0, eps=1e-8):
    """Per-slice z-score over all pixels, then clip to [-clip_value, clip_value].

    ``correction=0`` gives the population standard deviation, matching
    NumPy's ``std()`` in the course loader (Appendix B).
    """
    image = (image - image.mean()) / (image.std(correction=0) + eps)
    return image.clamp(-clip_value, clip_value)


def one_hot_mask(label):
    """Convert an integer label map (H, W) into a one-hot float mask (6, H, W)."""
    return F.one_hot(label, NUM_CLASSES).permute(2, 0, 1).float()


class TrainAugmentation:
    """Paired augmentation applied to training slices only.

    Rotation, scaling and flipping use the same random parameters for the
    image and the mask. The mask uses nearest-neighbour interpolation so it
    keeps integer labels. Regions moved in from outside the slice are filled
    with 0 (zero intensity / Background), which matches the zero image
    background. Random numbers come from ``torch``: with DataLoader workers
    each worker gets its own stream seeded from the loader's generator; with
    ``num_workers=0`` they come from the global torch seed, which
    ``make_dataloaders`` sets.

    Rotation and scaling are drawn together with probability ``p_affine``;
    flipping and gamma each have their own probability.

    Axis 0 (256 pixels) is treated as left-right and is the only flipped axis.
    The Nifti affines carry no orientation, so this is inferred from the
    near-perfect mirror symmetry of the bone and bladder masks along axis 0.
    """

    def __init__(self, max_rotation=15.0, scale_range=(0.85, 1.15),
                 gamma_range=(0.7, 1.5), p_affine=0.5, p_flip=0.5, p_gamma=0.5):
        self.max_rotation = max_rotation
        self.scale_range = scale_range
        self.gamma_range = gamma_range
        self.p_affine = p_affine
        self.p_flip = p_flip
        self.p_gamma = p_gamma

    @staticmethod
    def _uniform(low, high):
        return low + (high - low) * torch.rand(()).item()

    def __call__(self, image, label):
        """``image`` is a float tensor (C, H, W); ``label`` is a long tensor (H, W).

        Every channel gets the same rotation, scale, flip and gamma.
        """
        if torch.rand(()) < self.p_affine:
            angle = self._uniform(-self.max_rotation, self.max_rotation)
            scale = self._uniform(*self.scale_range)
            image = TF.affine(image, angle=angle, translate=[0, 0], scale=scale,
                              shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR,
                              fill=0.0)
            label = TF.affine(label.unsqueeze(0), angle=angle, translate=[0, 0], scale=scale,
                              shear=[0.0, 0.0], interpolation=InterpolationMode.NEAREST,
                              fill=0).squeeze(0)

        if torch.rand(()) < self.p_flip:
            # Flip along axis 0 of the slice (dimension -2 of the tensors).
            image = torch.flip(image, dims=[-2])
            label = torch.flip(label, dims=[-2])

        if torch.rand(()) < self.p_gamma:
            # Gamma needs values in [0, 1]; this temporary min-max scaling is
            # removed again by the z-score that follows in the dataset.
            # Each channel is scaled with its own minimum and maximum.
            gamma = self._uniform(*self.gamma_range)
            low = image.amin(dim=(-2, -1), keepdim=True)
            high = image.amax(dim=(-2, -1), keepdim=True)
            image = ((image - low) / (high - low + 1e-8)).clamp(0.0, 1.0) ** gamma

        return image, label


class HipMRISliceDataset(Dataset):
    """HipMRI 2D slices for one supplied split.

    Args:
        data_root: path to the ``keras_slices_data`` folder.
        split: ``"train"``, ``"validation"`` or ``"test"``.
        augment: optional callable ``(image, label) -> (image, label)``;
            pass ``TrainAugmentation()`` for the training split only.
        cache: load every slice into memory once (about 1.1 GB for the
            training split) instead of decompressing files every epoch.
            On Windows, DataLoader workers copy the whole dataset, so use
            ``num_workers=0`` there; Linux (e.g. Colab) shares it.
        check_counts: compare the pair and cropped-slice counts with the
            full dataset audit.
        context_slices: number K of neighbouring slices on each side added
            as extra input channels (2.5D input); 0 gives plain 2D slices.
    """

    def __init__(self, data_root, split, augment=None, cache=True, check_counts=True,
                 context_slices=0):
        if split not in SPLIT_FOLDERS:
            raise ValueError(f"split must be one of {list(SPLIT_FOLDERS)}")
        if context_slices < 0:
            raise ValueError("context_slices must be 0 or more")
        self.split = split
        self.augment = augment
        self.context_slices = context_slices
        self.pairs = find_pairs(data_root, split)
        if check_counts and len(self.pairs) != EXPECTED_PAIRS[split]:
            raise ValueError(f"{split}: found {len(self.pairs)} pairs, "
                             f"expected {EXPECTED_PAIRS[split]}")
        self.neighbours = self._neighbour_indices() if context_slices else None

        if cache:
            reads = [self._read_pair(i) for i in range(len(self.pairs))]
            self.cache = [(image, mask) for image, mask, _ in reads]
            cropped_flags = [was_cropped for _, _, was_cropped in reads]
        else:
            self.cache = None
            # The Nifti header shape is enough to know which slices get cropped.
            cropped_flags = [nib.load(str(p[0])).shape[1] != TARGET_WIDTH for p in self.pairs]
        # (case, week, slice) of every slice that is centre-cropped.
        self.cropped = [p[2:] for p, flag in zip(self.pairs, cropped_flags) if flag]
        if check_counts and len(self.cropped) != EXPECTED_CROPPED[split]:
            raise ValueError(f"{split}: {len(self.cropped)} slices need cropping, "
                             f"expected {EXPECTED_CROPPED[split]}")

    def _neighbour_indices(self):
        """For each pair, the pair indices of slices s-K, ..., s+K of the same series.

        Slice numbers outside the series are clamped to its first or last
        slice. A series must hold every slice number between its first and
        last, otherwise a neighbour would be missing.
        """
        index_of = {pair[2:]: i for i, pair in enumerate(self.pairs)}
        series = defaultdict(list)
        for case, week, slice_index in index_of:
            series[(case, week)].append(slice_index)
        bounds = {}
        for key, slices in series.items():
            low, high = min(slices), max(slices)
            if len(slices) != high - low + 1:
                raise ValueError(f"{self.split}: series {key} has gaps in its slice numbers")
            bounds[key] = (low, high)

        k = self.context_slices
        neighbours = []
        for _, _, case, week, slice_index in self.pairs:
            low, high = bounds[(case, week)]
            neighbours.append([index_of[(case, week, min(max(s, low), high))]
                               for s in range(slice_index - k, slice_index + k + 1)])
        return neighbours

    def _arrays(self, index):
        """Return the (image, mask) arrays of one pair from the cache or from disk."""
        if self.cache is not None:
            return self.cache[index]
        image, mask, _ = self._read_pair(index)
        return image, mask

    def _read_pair(self, index):
        """Read, check and crop one pair.

        Returns (image as int16 or float32, mask as uint8, whether it was cropped).
        """
        image_path, mask_path, case, week, slice_index = self.pairs[index]
        image = load_nifti_2d(image_path)
        mask = load_nifti_2d(mask_path)

        if not np.isfinite(image).all():
            raise ValueError(f"{image_path.name}: non-finite intensity values")
        if not np.array_equal(mask, np.round(mask)) or mask.min() < 0 or mask.max() >= NUM_CLASSES:
            raise ValueError(f"{mask_path.name}: labels must be integers 0-{NUM_CLASSES - 1}")

        was_cropped = image.shape[1] != TARGET_WIDTH
        if was_cropped:
            image, mask = centre_crop_width(image, mask)
        elif image.shape != mask.shape or image.shape[0] != TARGET_HEIGHT:
            raise ValueError(f"{image_path.name}: unsupported shape {image.shape}")

        # HipMRI intensities are whole numbers below 32768, so int16 halves
        # the cache size without changing any value.
        if np.array_equal(image, np.round(image)) and np.abs(image).max() < 32768:
            image = image.astype(np.int16)
        else:
            image = image.astype(np.float32)
        return (np.ascontiguousarray(image), np.ascontiguousarray(mask.astype(np.uint8)),
                was_cropped)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image, mask = self._arrays(index)
        _, _, case, week, slice_index = self.pairs[index]

        if self.neighbours is not None:
            # 2.5D: stack the neighbouring slices; the target slice is in the middle.
            image = np.stack([self._arrays(j)[0] for j in self.neighbours[index]])
        else:
            image = image[np.newaxis]
        image = torch.from_numpy(image.astype(np.float32))  # (C, H, W)
        label = torch.from_numpy(mask.astype(np.int64))    # (H, W)
        if self.augment is not None:
            image, label = self.augment(image, label)
        # Per-channel z-score; with one channel this is the 2D normalisation.
        image = torch.cat([normalise_slice(channel.unsqueeze(0)) for channel in image])
        if not torch.isfinite(image).all():
            raise ValueError(f"Non-finite values after normalisation: {self.pairs[index][0].name}")

        return {
            "image": image,
            "mask": one_hot_mask(label),  # F.one_hot also rejects labels outside 0-5
            "label": label,
            "split": self.split,
            "case": case,
            "week": week,
            "slice": slice_index,
        }

    def series_keys(self):
        """Return the sorted (case, week) series identifiers in this split."""
        return sorted({(p[2], p[3]) for p in self.pairs})


def make_dataloaders(data_root, batch_size=16, num_workers=None, seed=42, cache=True,
                     context_slices=0):
    """Build train/validation/test DataLoaders using the supplied splits.

    Only the training loader shuffles and augments. Shuffling uses a seeded
    generator so the slice order is reproducible; with ``num_workers=0`` the
    global torch seed is also set so augmentations repeat. A batch size of at
    least 8 keeps the chance of a batch with no prostate slice low (about
    2.7% at 8), and the incomplete last training batch is dropped for the
    same reason. Validation and test keep every slice.

    ``num_workers=None`` picks 0 on Windows (workers there would each copy
    the cached data) and 2 elsewhere, e.g. on Colab. Use the same value for
    both models so they see the same augmentation stream. Workers are not
    persistent, so reseeding the training generator before an epoch (as
    ``train.py`` does) fixes that epoch's order and augmentation.
    ``context_slices`` is passed to every split (see ``HipMRISliceDataset``).
    """
    if batch_size < 8:
        raise ValueError("batch_size must be at least 8 (see the data policy)")
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 2
    if num_workers == 0:
        torch.manual_seed(seed)
    check_split_disjoint(data_root)

    datasets = {
        "train": HipMRISliceDataset(data_root, "train", augment=TrainAugmentation(), cache=cache,
                                    context_slices=context_slices),
        "validation": HipMRISliceDataset(data_root, "validation", cache=cache,
                                         context_slices=context_slices),
        "test": HipMRISliceDataset(data_root, "test", cache=cache,
                                   context_slices=context_slices),
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=(name == "train"),
                         drop_last=(name == "train"),
                         num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                         generator=generator if name == "train" else None)
        for name, dataset in datasets.items()
    }
    return loaders


def summarise_split(dataset):
    """Return simple counts used to check a split before training."""
    prostate_slices = 0
    label_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)
    series = defaultdict(int)
    for index in range(len(dataset)):
        _, mask = dataset._arrays(index)
        counts = np.bincount(mask.ravel(), minlength=NUM_CLASSES)
        label_pixels += counts
        prostate_slices += int(counts[PROSTATE_LABEL] > 0)
        case, week = dataset.pairs[index][2:4]
        series[(case, week)] += 1
    return {
        "pairs": len(dataset),
        "series": len(series),
        "cases": len({case for case, _ in series}),
        "slices_per_series": sorted(set(series.values())),
        "prostate_slices": prostate_slices,
        "label_pixels": dict(zip(CLASS_NAMES, label_pixels.tolist())),
        "cropped_slices": len(dataset.cropped),
        "cropped_series": sorted({(c, w) for c, w, _ in dataset.cropped}),
    }


def save_augmentation_preview(dataset, path, n=8, seed=0):
    """Save image/mask overlays of augmented samples to check their alignment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    # Prefer prostate-positive slices so the overlay shows the target class.
    positives = [i for i in range(len(dataset))
                 if (dataset._arrays(i)[1] == PROSTATE_LABEL).any()]
    step = max(1, len(positives) // n)
    indices = positives[::step][:n]

    fig, axes = plt.subplots(2, len(indices), figsize=(2.2 * len(indices), 8))
    for column, index in enumerate(indices):
        sample = dataset[index]
        # The target slice (the middle channel for 2.5D input) matches the label.
        image = sample["image"][sample["image"].shape[0] // 2].numpy()
        label = sample["label"].numpy()
        title = f"{sample['case']}/w{sample['week']}/s{sample['slice']}"
        axes[0, column].imshow(image, cmap="gray")
        axes[0, column].set_title(title, fontsize=8)
        axes[1, column].imshow(image, cmap="gray")
        axes[1, column].imshow(np.ma.masked_equal(label, 0), cmap="tab10",
                               vmin=0, vmax=9, alpha=0.45, interpolation="nearest")
        axes[1, column].contour(label == PROSTATE_LABEL, levels=[0.5], colors="red", linewidths=0.8)
        for row in range(2):
            axes[row, column].axis("off")
    axes[0, 0].set_ylabel("augmented image")
    fig.suptitle("Training augmentation check (red = prostate contour)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Check the HipMRI 2D dataset pipeline.")
    parser.add_argument("--data-root", default="keras_slices_data",
                        help="path to the keras_slices_data folder")
    parser.add_argument("--preview", default=None,
                        help="optional PNG path for a training augmentation preview")
    parser.add_argument("--context-slices", type=int, default=0,
                        help="neighbouring slices on each side for 2.5D input (0 = 2D)")
    args = parser.parse_args()

    cases = check_split_disjoint(args.data_root)
    print("Case IDs per split:", {split: len(ids) for split, ids in cases.items()})
    channels = 2 * args.context_slices + 1
    for split in SPLIT_FOLDERS:
        augment = TrainAugmentation() if split == "train" else None
        dataset = HipMRISliceDataset(args.data_root, split, augment=augment,
                                     context_slices=args.context_slices)
        print(split, summarise_split(dataset))

        # Check every item of the split (training items are freshly augmented).
        for index in range(len(dataset)):
            sample = dataset[index]
            assert sample["image"].shape == (channels, TARGET_HEIGHT, TARGET_WIDTH)
            assert sample["mask"].shape == (NUM_CLASSES, TARGET_HEIGHT, TARGET_WIDTH)
            assert sample["image"].abs().max() <= 5.0
            assert torch.all(sample["mask"].sum(dim=0) == 1)
        print(f"{split}: all {len(dataset)} items passed shape, range and one-hot checks")
        if split == "train" and args.preview:
            save_augmentation_preview(dataset, args.preview)
            print("Saved augmentation preview to", args.preview)


if __name__ == "__main__":
    main()
