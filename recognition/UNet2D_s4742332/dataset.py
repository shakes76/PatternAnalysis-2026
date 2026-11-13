"""
Containing the data loader for loading and preprocessing your data, including leakage-
free train/validation/test splitting functions and data augmentations
"""

"""
Data Notes:
1.
"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass
import re
from collections import Counter
from itertools import combinations
import numpy as np, random
import nibabel as nib
import matplotlib.pyplot as plt
from typing import Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

_NAME_RE = re.compile(r"(?:case|seg)_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz")
SPLITS = ("train", "validate", "test")
H, W = 256, 128 # most slices; 019_week_1 is 256x144 and gets cropped
N_LABELS = 6
# the data folder (root) and the slice cache folder (cache_dir, ~2 GB, keep it outside the repo) are passed in
# from the command line (--data-root, --cache-dir in train.py and predict.py), nothing here points at a drive


# dataclasses
@dataclass(frozen=True, order=True)
class SliceID:
    """one slice: patient case, weekly scan, axial slice index and the paths to both it's image and mask paths
    not a seperated ID as mask-images must always be paired
    """
    patient: int # parsed as int so 004 and 4 are the same patient
    week: int
    slice: int
    image_path: Path
    mask_path: Path

def patient_weights(ids: list[SliceID], mode: str):
    if mode not in ("inverse", "sqrt"):
        raise ValueError(f"patient_weighting must be None, 'inverse' or 'sqrt', got {mode!r}")
    n_scans = Counter(p for p, _ in {(s.patient, s.week) for s in ids})
    power = 1.0 if mode == "inverse" else 0.5
    return torch.tensor([n_scans[s.patient] ** -power for s in ids], dtype=torch.double)

def full_load_dataset(seed: int | None, batch_size: int, augment: bool, num_workers: int, n_val: int, n_test: int, root: Path, cache_dir: Path,
                      patient_weighting: str | None = None, slice_weights=None, resplit: bool = True, context: int = 0):
    """slice_weights: optional callable (train dataset) -> tensor of per-slice sampling weights, multiplied with the patient weights.
    resplit=False keeps the provided train/validate/test patients, seed then only drives shuffling.
    context: neighbouring slices on each side stacked as extra channels (2.5D), 0 = plain 2D"""
    print(f"Loading dataset with seed {seed}")
    all_ids = build_all_sliceids(root)
    if resplit:
        all_ids = resplit_patients(all_ids, seed, n_val, n_test)
    pooled = [s for ids in all_ids.values() for s in ids]
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    loaders = {}
    for split, ids in all_ids.items():
        train = split == "train"
        ds = HipMRISlices(ids, pooled, cache_dir, JointTransform2D() if train and augment else None, context)
        sampler = None
        if train and (patient_weighting or slice_weights):  # train only, validate/test stay unweighted
            weights = torch.ones(len(ds), dtype=torch.double)
            if patient_weighting:
                weights *= patient_weights(ids, patient_weighting)
            if slice_weights:
                weights *= slice_weights(ds)
            sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True, generator=generator)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=train and sampler is None, sampler=sampler, drop_last=train,
                                    num_workers=num_workers, persistent_workers=num_workers > 0,
                                    pin_memory=torch.cuda.is_available(), generator=generator if train and sampler is None else None)
    return loaders

# folder work
def name_to_paths(root: Path, split: str, name: str):
    """image file name -> (image path, mask path). mask name is the image name with case_ swapped for seg_"""
    mask_name = "seg_" + name.removeprefix("case_")
    return root / f"keras_slices_{split}" / name, root / f"keras_slices_seg_{split}" / mask_name

def resplit_patients(all_slice_ids: dict[str, list[SliceID]], seed: int | None = None,
                     n_val: int = 4, n_test: int = 3):
    if seed is None:
        return all_slice_ids                        # default: keep the provided split

    pooled = [s for ids in all_slice_ids.values() for s in ids]
    patients = sorted(patients_in(pooled))          # fixed start order -> same seed, same split
    if n_val + n_test >= len(patients):
        raise ValueError(f"n_val + n_test must be < {len(patients)} patients")

    patients = np.random.default_rng(seed).permutation(patients)
    test_p  = patients[:n_test]
    val_p   = patients[n_test:n_test + n_val]
    train_p = patients[n_test + n_val:]

    split_of = {}
    for p in train_p: split_of[int(p)] = "train"
    for p in val_p:   split_of[int(p)] = "validate"
    for p in test_p:  split_of[int(p)] = "test"

    new = {split: [] for split in SPLITS}
    for s in pooled:
        new[split_of[s.patient]].append(s)
    new = {split: sorted(ids) for split, ids in new.items()}

    check_no_patient_leakages({split: patients_in(ids) for split, ids in new.items()})
    return new

def build_sliceids(root: Path, split: str):
    check_pairs(root=root, split=split)
    all_ids = []
    for name in file_names_in(root / f"keras_slices_{split}"):
        img_path, mask_path = name_to_paths(root, split, name)
        all_ids.append(parse_to_sliceid(name, img_path=img_path, mask_path=mask_path))
    return sorted(all_ids) # file_names_in is a set, so sort by patient, week, slice

def build_all_sliceids(root: Path):
    all_slice_ids = {split: build_sliceids(root, split) for split in SPLITS}
    check_no_patient_leakages({split: patients_in(ids) for split, ids in all_slice_ids.items()})
    for split, ids in all_slice_ids.items():
        bad = check_slice_count(Counter((s.patient, s.week) for s in ids))
        if bad:
            raise ValueError(f"{split}: scans without 60 slices: {bad}")
    return all_slice_ids

def parse_file_name(name: str):
    m = _NAME_RE.fullmatch(name)
    if m is None:
        raise ValueError(f"bad filename: {name}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)) #patient, week, slice

def parse_to_sliceid(name: str, img_path: Path, mask_path: Path) -> SliceID:
    patient, week, slice_idx = parse_file_name(name=name)
    return SliceID(
        patient=patient,
        week=week,
        slice=slice_idx,
        image_path=img_path,
        mask_path=mask_path
    )

# loading
def center_crop(a: np.ndarray, h: int = H, w: int = W):
    if a.shape[0] < h or a.shape[1] < w:
        raise ValueError(f"can't crop {a.shape} to {(h, w)}")
    top = (a.shape[0] - h) // 2
    left = (a.shape[1] - w) // 2
    return a[top:top + h, left:left + w]

def load_slice(sid: SliceID):
    image = nib.load(sid.image_path).get_fdata().astype(np.float32)
    mask = np.asarray(nib.load(sid.mask_path).dataobj)
    if image.shape != mask.shape:
        raise ValueError(f"{sid}: image {image.shape} vs mask {mask.shape}")
    if image.shape != (H, W):
        image, mask = center_crop(image), center_crop(mask) # same crop for both

    image = (image - image.mean()) / (image.std() + 1e-8) # per-slice z-score, eps for blank slices
    if mask.min() < 0 or mask.max() >= N_LABELS:
        raise ValueError(f"{sid}: labels {np.unique(mask)} outside 0-{N_LABELS - 1}")
    return image, mask.astype(np.uint8)

def build_cache(ids: list[SliceID], cache_dir: Path):
    ids = sorted(ids)
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = np.array([(s.patient, s.week, s.slice) for s in ids], dtype=np.int32)
    images = np.lib.format.open_memmap(cache_dir / "images.tmp.npy", mode="w+", dtype=np.float32, shape=(len(ids), H, W))
    masks = np.lib.format.open_memmap(cache_dir / "masks.tmp.npy", mode="w+", dtype=np.uint8, shape=(len(ids), H, W))
    for i, sid in enumerate(ids):
        images[i], masks[i] = load_slice(sid)
        if i % 1000 == 0:
            print(f"cache {i}/{len(ids)}")
    images.flush(); masks.flush()
    del images, masks 
    os.replace(cache_dir / "images.tmp.npy", cache_dir / "images.npy")
    os.replace(cache_dir / "masks.tmp.npy", cache_dir / "masks.npy")
    np.save(cache_dir / "keys.npy", keys)
    print(f"cache built: {len(ids)} slices -> {cache_dir}")

def load_cache(ids: list[SliceID], cache_dir: Path):
    ids = sorted(ids) # the cache is always in sorted order, whatever order the caller pooled in
    if not (cache_dir / "keys.npy").exists():
        build_cache(ids, cache_dir)
    keys = np.load(cache_dir / "keys.npy")
    expected = np.array([(s.patient, s.week, s.slice) for s in ids], dtype=np.int32)
    if not np.array_equal(keys, expected):
        raise ValueError(f"cache in {cache_dir} doesn't match these slices, delete it and rebuild")
    images = np.load(cache_dir / "images.npy", mmap_mode="r") 
    masks = np.load(cache_dir / "masks.npy", mmap_mode="r")
    return images, masks

def file_names_in(folder: Path):
    return {f.name for f in folder.iterdir()}

def patients_in(ids: list[SliceID]):
    return {s.patient for s in ids}

def summarise_split(split: str, ids: list[SliceID]):
    scans = {(s.patient, s.week) for s in ids} # a scan = patient + week
    print(f"{split:9} patients={len(patients_in(ids)):3}  scans={len(scans):4}  slices={len(ids):5}")

# data checks/investigation -> class soon?
def check_pairs(root: Path, split: str):
    # compare parsed (patient, week, slice) tuples, the names themselves differ (case_ vs seg_)
    images = {parse_file_name(n) for n in file_names_in(root / f"keras_slices_{split}")}
    masks  = {parse_file_name(n) for n in file_names_in(root / f"keras_slices_seg_{split}")}
    only_images = images - masks     # set difference
    only_masks  = masks - images
    if only_images or only_masks:
        raise ValueError(
            f"{split}: {len(only_images)} images without masks, "
            f"{len(only_masks)} masks without images"
        )
    print(f"{split}: {len(images)} images/masks paired OK")

def check_no_patient_leakages(patients: dict[str, set[int]]):
    problems = []
    for a, b in combinations(patients, 2):
        shared = patients[a] & patients[b]          # patients in both splits
        if shared:
            problems.append(f"{a} & {b} share {sorted(shared)}")
    if problems:
        raise ValueError("patient leakage: " + "; ".join(problems))
    print("no leakage")

def check_slice_count(slices_per_scan):
    return {scan: n for scan, n in slices_per_scan.items() if n!= 60}

def check_img_shape(image_metadata):
    return image_metadata.shape

def check_labels(root:Path):
    shapes, dtypes, zooms, labels = Counter(), Counter(), Counter(), Counter()
    for f in root.iterdir():
        img = nib.load(f)
        shapes[img.shape] += 1
        zooms[tuple(round(z, 3) for z in img.header.get_zooms())] += 1
        dtypes[str(img.get_data_dtype())] += 1
        if "seg" in root.name:
            v, c = np.unique(np.asarray(img.dataobj), return_counts=True)
            for a, b in zip(v, c): labels[int(a)] += int(b)
    print(root.name, dict(shapes), dict(dtypes), dict(zooms))
    if labels:
        tot = sum(labels.values())
        print("   labels:", {k: f"{v/tot:.4%}" for k, v in sorted(labels.items())})
    return

def check_images_manually(sid: SliceID):
    image = nib.load(sid.image_path).get_fdata()
    mask = np.asarray(nib.load(sid.mask_path).dataobj)
    labels = np.unique(mask)
    fig, axes = plt.subplots(1, len(labels), figsize=(3 * len(labels), 5))
    for ax, k in zip(axes, labels):
        ax.imshow(image.T, cmap="gray")                                  # .T: see note below
        ax.imshow(np.ma.masked_where(mask != k, mask).T, alpha=0.5, cmap="autumn")
        ax.set_title(f"label {k}  ({(mask == k).mean():.2%})")
        ax.axis("off")
    fig.suptitle(f"patient {sid.patient:03d} week {sid.week} slice {sid.slice}")
    plt.show()

def check_weeks_per_case(split: str, ids: list[SliceID]):
    weeks_per_case = {}
    for s in ids:
        weeks_per_case.setdefault(s.patient, set())
        weeks_per_case[s.patient].add(s.week)

    for patient, weeks in sorted(weeks_per_case.items() ):
        print(f"{split} patient {patient:03d}: {len(weeks)} weeks {sorted(weeks)}")
    return weeks_per_case

### REFER TO https://github.com/cosmic-cortex/pytorch-UNet/blob/master/unet/dataset.py
def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

###

def slice_name(sid: SliceID):
    return f"case_{sid.patient:03d}_week_{sid.week}_slice_{sid.slice}"

class HipMRISlices(Dataset):
    """
    2D slices of one split, read from the cache.
    split_ids: the SliceIDs of this split, pooled: the sorted SliceIDs of every split (the cache order).
    cache_dir: folder holding (or to build) the slice cache.
    joint_transform: callable (image, mask) -> (image, mask) on tensors, applied to both together.
    context: neighbouring slices on each side of the same scan stacked around it (2.5D), the edge slice repeats past the scan's ends.
    returns image [1+2*context,H,W] float32 (centre slice in the middle), mask [H,W] long (centre slice), slice name (a str so the DataLoader can batch it)
    """
    def __init__(self, split_ids: list[SliceID], pooled: list[SliceID], cache_dir: Path, joint_transform=None, context: int = 0):
        pooled = sorted(pooled) # cache order, so callers can pool the splits in any order
        index_of = {s: i for i, s in enumerate(pooled)}
        self.ids = split_ids
        self.indices = [index_of[s] for s in split_ids]
        key_index = {(s.patient, s.week, s.slice): i for i, s in enumerate(pooled)}
        self.stacks = [[self.neighbour(key_index, s, o) for o in range(-context, context + 1)] for s in split_ids]
        self.context = context
        self.pooled = pooled
        self.joint_transform = joint_transform
        self.cache_dir = cache_dir
        self.images, self.masks = load_cache(pooled, cache_dir)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["images"] = state["masks"] = None
        return state

    @staticmethod
    def neighbour(key_index, sid: SliceID, offset: int):
        """cache index of the slice offset away in the same scan, stepping back towards sid while that slice doesn't exist"""
        step = -1 if offset > 0 else 1
        while (sid.patient, sid.week, sid.slice + offset) not in key_index:
            offset += step
        return key_index[(sid.patient, sid.week, sid.slice + offset)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.images is None:
            self.images, self.masks = load_cache(self.pooled, self.cache_dir)
        i = self.indices[idx]
        image = torch.from_numpy(np.stack([self.images[j] for j in self.stacks[idx]]))
        mask = to_long_tensor(self.masks[i])
        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)
        return image, mask, slice_name(self.ids[idx])


#Original Base form by Tivadar Danka: https://github.com/cosmic-cortex/pytorch-UNet/blob/master/unet/unet.py
class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Changed for HipMRI: works on the tensors HipMRISlices gives (image [1,H,W] z-scored float, mask [H,W] long)
    instead of PIL, which would clip the z-scored values to 0-255. randomness from torch so workers/seeds behave.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken. must fit in (H, W), off by default so every batch stays 256x128.
        p_flip: float, the probability of performing a random horizontal flip.
        intensity_params: (contrast, brightness), image * U(1-c, 1+c) + U(-b, b) in z-score units.
            replaces ColorJitter, which assumes [0,1] images and has no use for saturation/hue on grayscale.
            If bool(intensity_params) evaluates to false, no intensity transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform.
        affine_params: (degrees, translate, scale) ranges for T.RandomAffine.get_params. small by default,
            the reference's 180 deg / 45 deg shear isn't realistic for aligned pelvic slices.
    """
    def __init__(self, crop=None, p_flip=0.5, intensity_params=(0.1, 0.1),
                 p_random_affine=0.5, affine_params=(10, (0.05, 0.05), (0.9, 1.1))):
        if crop and (crop[0] > H or crop[1] > W):
            raise ValueError(f"crop {crop} bigger than slices {(H, W)}")
        self.crop = crop
        self.p_flip = p_flip
        self.intensity_params = intensity_params
        self.p_random_affine = p_random_affine
        self.affine_params = affine_params

    def __call__(self, image, mask):
        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        if torch.rand(1) < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        # intensity transforms || ONLY ON IMAGE
        if self.intensity_params:
            c, b = self.intensity_params
            image = image * float(torch.empty(1).uniform_(1 - c, 1 + c)) + float(torch.empty(1).uniform_(-b, b))

        # random affine transform, nearest for the mask so labels stay 0-5
        if torch.rand(1) < self.p_random_affine:
            degrees, translate, scale = self.affine_params
            size = [image.shape[-1], image.shape[-2]] # get_params wants (w, h)
            angle, shift, s, shear = T.RandomAffine.get_params((-degrees, degrees), translate, scale, None, size)
            image = F.affine(image, angle, shift, s, shear,
                             interpolation=T.InterpolationMode.BILINEAR, fill=[float(image.min())]) # z-scored bg isn't 0
            mask = F.affine(mask.unsqueeze(0), angle, shift, s, shear,
                            interpolation=T.InterpolationMode.NEAREST, fill=[0]).squeeze(0) # out of frame = background

        return image, mask





if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Smoke test: build the loaders and print one batch per split")
    p.add_argument("--data-root", type=Path, required=True, help="HipMRI keras_slices_data folder")
    p.add_argument("--cache-dir", type=Path, required=True, help="slice cache folder (~2 GB), keep it outside the repo")
    cli = p.parse_args()
    loaders = full_load_dataset(seed=11, batch_size=8, augment=True, num_workers=2, n_val=5, n_test=5, root=cli.data_root, cache_dir=cli.cache_dir)
    for split, loader in loaders.items():
        images_b, masks_b, names_b = next(iter(loader))
        print(f"{split:9} slices={len(loader.dataset):5} batches={len(loader):4}  "
              f"{tuple(images_b.shape)} {images_b.dtype}  {tuple(masks_b.shape)} {masks_b.dtype}  {names_b[0]}")

    # all_ids = build_all_sliceids(cli.data_root)
    # all_ids = resplit_patients(all_ids, 1, 5, 5)
    
    # for split, ids in all_ids.items():
    #     summarise_split(split, ids)
    #     print("   first:", ids[:3])
    
    
        #_ = check_weeks_per_case(split, ids)
        #check_labels(cli.data_root / f"keras_slices_seg_{split}")
    # #check_images_manually(all_ids["train"][45])

   
    # pooled = [s for ids in all_ids.values() for s in ids]
    # cropped = next(s for s in pooled if s.patient == 19 and s.week == 1)
    # # for sid in (pooled[0], cropped):
    # #     image, mask = load_slice(sid)
    # #     print(f"patient {sid.patient:03d} week {sid.week} slice {sid.slice}: "
    # #           f"image {image.shape} {image.dtype} mean={image.mean():.3f} std={image.std():.3f}  "
    # #           f"mask {mask.shape} {mask.dtype} labels={np.unique(mask)}")

    # # pooled = sorted(pooled)
    # # images, masks = load_cache(pooled)
    # # print("cache:", images.shape, images.dtype, masks.shape, masks.dtype)
    # # for i in (0, pooled.index(cropped), len(pooled) - 1):
    # #     image, mask = load_slice(pooled[i])
    # #     if not (np.array_equal(images[i], image) and np.array_equal(masks[i], mask)):
    # #         raise ValueError(f"cache slice {i} ({pooled[i]}) doesn't match load_slice")
    # # print("cache matches load_slice")

    # datasets = {split: HipMRISlices(ids, pooled, JointTransform2D() if split == "train" else None)
    #             for split, ids in all_ids.items()}
    # for split, ds in datasets.items():
    #     image, mask, name = ds[0]
    #     print(f"{split:9} len={len(ds):5}  {name}: image {tuple(image.shape)} {image.dtype}  mask {tuple(mask.shape)} {mask.dtype}")
    # loader = DataLoader(datasets["train"], batch_size=8, shuffle=True, num_workers=2) # workers: checks the reopen path
    # images_b, masks_b, names_b = next(iter(loader))
    # print("batch:", tuple(images_b.shape), images_b.dtype, tuple(masks_b.shape), masks_b.dtype, names_b[:2])

    # # augmentation check: shapes/dtypes kept, mask labels stay a subset of the original, same seed -> same output
    # plain = HipMRISlices(all_ids["train"], pooled)
    # aug = JointTransform2D()
    # for idx in (0, len(plain) // 2):
    #     image, mask, name = plain[idx]
    #     torch.manual_seed(0); a_img, a_mask = aug(image, mask)
    #     torch.manual_seed(0); b_img, b_mask = aug(image, mask)
    #     if a_img.shape != image.shape or a_mask.shape != mask.shape or a_mask.dtype != torch.long:
    #         raise ValueError(f"{name}: augmentation changed shape/dtype")
    #     if not set(a_mask.unique().tolist()) <= set(mask.unique().tolist()):
    #         raise ValueError(f"{name}: augmentation made new labels {a_mask.unique().tolist()}")
    #     if not (torch.equal(a_img, b_img) and torch.equal(a_mask, b_mask)):
    #         raise ValueError(f"{name}: same seed gave different augmentations")
    #     print(f"aug {name}: labels {mask.unique().tolist()} -> {a_mask.unique().tolist()}  "
    #           f"fg {(mask > 0).float().mean():.3f} -> {(a_mask > 0).float().mean():.3f}")
    # print("augmentation OK")






