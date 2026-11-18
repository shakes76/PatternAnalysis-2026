# Containing the data loader for loading and preprocessing your data, including leakage-free 
# train/validation/test splitting functions and data augmentations

import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
import torch.nn.functional as F

import os
import math
from glob import glob


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1

    return res


# load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage: bool (z-score the image: zero mean, unit std)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, orient=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    normImage: bool (z-score the image: zero mean, unit std)
    orient: Apply orientation and resample image? Good for images with large slice thickness or anisotropic resolution
    dtype: Type of the data. If dtype=np.uint8, it is assumed that the data is labels
    early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    interp = 'linear'
    if dtype == np.uint8:  # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :] = inImage
        else:
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

def get_pairs(data_dir):
    # match each MR to its label by name: Case_004_Week0_LFOV <-> Case_004_Week0_SEMANTIC_LFOV
    labels = {os.path.basename(f).replace("_SEMANTIC", ""): f
              for f in glob(os.path.join(data_dir, "semantic_labels_anon", "*.nii.gz"))}
    mrs = sorted(glob(os.path.join(data_dir, "semantic_MRs_anon", "*.nii.gz")))
    return [(mr, labels[os.path.basename(mr)]) for mr in mrs if os.path.basename(mr) in labels]


def split_pairs(pairs, val_frac=0.15, test_frac=0.15, seed=42):
    # split by patient (Case_XXX), so no patient's other weeks leak between splits
    # fixed seed so the 2D baseline and the 3D model are tested on exactly the same cases
    cases = sorted({os.path.basename(mr).split("_")[1] for mr, _ in pairs})
    np.random.default_rng(seed).shuffle(cases)
    n_val, n_test = round(len(cases) * val_frac), round(len(cases) * test_frac)
    groups = {"test": set(cases[:n_test]), "val": set(cases[n_test:n_test + n_val]), "train": set(cases[n_test + n_val:])}
    return {name: [p for p in pairs if os.path.basename(p[0]).split("_")[1] in group] for name, group in groups.items()}


def load_split(pairs):
    images = load_data_3D([mr for mr, _ in pairs], normImage=True)
    labels = load_data_3D([seg for _, seg in pairs], dtype=np.uint8)
    return torch.utils.data.TensorDataset(torch.from_numpy(images).unsqueeze(1), torch.from_numpy(labels))


def get_loaders(splits, batch_size=2):
    # one loader per split in splits (e.g. train/val/test), only train is shuffled and batched
    return {name: torch.utils.data.DataLoader(load_split(pairs), batch_size=batch_size if name == "train" else 1,
                                              shuffle=name == "train", pin_memory=True)
            for name, pairs in splits.items()}


def check_shapes(pairs):
    # load_data_3D sizes everything from the first file, so flag any volume with a different shape
    shapes = {nib.load(f).shape for pair in pairs for f in pair}
    print("Volume shapes found:", shapes)


def get_data(pairs, norm=True, early_stop=False, threeD=True):
    # takes one split's pairs (from split_pairs) so test patients never leak into training
    mr_files, seg_files = zip(*pairs)
    images = load_data_3D(list(mr_files), normImage=norm, early_stop=early_stop)
    labels = load_data_3D(list(seg_files), dtype=np.uint8, early_stop=early_stop)
    if not threeD:
        # (N, H, W, D) -> (N*D, H, W): every slice k of volume i stays paired with label slice k of volume i
        images = images.transpose(0, 3, 1, 2).reshape(-1, *images.shape[1:3])
        labels = labels.transpose(0, 3, 1, 2).reshape(-1, *labels.shape[1:3])
    return images, labels


def augment(images, labels, p=0.5):
    # random flip, rotate+zoom and intensity jitter, each applied to a volume with probability p (training only)
    # only 26 training patients, so this helps stop the model memorising the training set
    # images: (B, 1, H, W, D) float, labels: (B, H, W, D) long, already on the gpu. H is left-right (LPS), H x W is axial
    B, dev = images.shape[0], images.device
    on = lambda: (torch.rand(B, device=dev) < p).float()
    u = lambda: torch.rand(B, device=dev) * 2 - 1 # uniform in [-1, 1]

    # left-right mirror, the body is roughly symmetric this way
    # (no front-back flip, that would put the bladder below the prostate)
    flip = on().bool()
    images = torch.where(flip.view(-1, 1, 1, 1, 1), images.flip(2), images)
    labels = torch.where(flip.view(-1, 1, 1, 1), labels.flip(1), labels)

    # rotate up to 15 degrees in the axial plane and zoom +-10%, same grid for image and label
    # simulates small differences in patient positioning and organ size
    sel = on()
    angle, scale = sel * u() * math.radians(15), 1 + sel * u() * 0.1
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.zeros(B, 3, 4, device=dev)
    theta[:, 0, 0] = 1 # grid x is D, untouched. grid y, z are W, H
    theta[:, 1, 1], theta[:, 1, 2], theta[:, 2, 1], theta[:, 2, 2] = cos, -sin, sin, cos
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    # border padding copies edge values, 0 isn't background after z-score normalisation
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
    # nearest so labels stay whole class numbers
    labels = F.grid_sample(labels[:, None].float(), grid, mode="nearest", padding_mode="border", align_corners=False)[:, 0].long()

    # brightness and contrast on the image only (alternative to gamma for simplicity)
    # x' = a*x + b, so the model relies on shape and relative contrast, not exact values
    sel = on().view(-1, 1, 1, 1, 1)
    images = images * (1 + sel * u().view(-1, 1, 1, 1, 1) * 0.1) + sel * u().view(-1, 1, 1, 1, 1) * 0.1
    return images, labels
