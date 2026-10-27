"""
dataset.py  (3D HipMRI)

Pairs each MR volume with its label volume, splits BY PATIENT, downsamples
256x256x128 volumes to 128x128x64, and returns batched tf.data datasets.
Images are stored as float16 and labels as uint8 in RAM; one-hot encoding
happens per batch (avoids the huge float32 one-hot arrays that crashed a node
in the 2D work).
"""
import glob
import os
import random
import numpy as np
import nibabel as nib
import tensorflow as tf
from tqdm import tqdm

BASE_DIR = "/home/groups/comp3710/HipMRI_Study_open"
NUM_CLASSES = 6
TARGET_SHAPE = (128, 128, 64)
SEED = 42


def list_pairs():
    """Return [(mr_path, label_path)]; B006_Week0_LFOV <-> B006_Week0_SEMANTIC."""
    mrs = sorted(glob.glob(os.path.join(BASE_DIR, "semantic_MRs", "*_LFOV.nii.gz")))
    pairs = []
    for m in mrs:
        name = os.path.basename(m).replace("_LFOV", "_SEMANTIC")
        lab = os.path.join(BASE_DIR, "semantic_labels_only", name)
        if not os.path.exists(lab):
            raise FileNotFoundError(f"No label for {m}")
        pairs.append((m, lab))
    return pairs


def patient_id(path):
    """'B006_Week0_LFOV.nii.gz' -> 'B006'."""
    return os.path.basename(path).split("_")[0]


def split_by_patient(pairs, seed=SEED, frac=0.16):
    """All volumes of a patient go to exactly one split (~68/16/16 by patient)."""
    patients = sorted({patient_id(m) for m, _ in pairs})
    random.Random(seed).shuffle(patients)
    n = max(1, round(frac * len(patients)))
    test_p, val_p, train_p = set(patients[:n]), set(patients[n:2 * n]), set(patients[2 * n:])
    assert not (train_p & val_p) and not (train_p & test_p) and not (val_p & test_p)
    pick = lambda ps: [p for p in pairs if patient_id(p[0]) in ps]
    return pick(train_p), pick(val_p), pick(test_p)


def fix_depth(arr, depth=128):
    """Centre-crop or zero-pad the last axis to `depth` (one volume has 144)."""
    d = arr.shape[2]
    if d > depth:
        s = (d - depth) // 2
        return arr[:, :, s:s + depth]
    if d < depth:
        before = (depth - d) // 2
        return np.pad(arr, ((0, 0), (0, 0), (before, depth - d - before)))
    return arr


def load_volume(mr_path, lab_path):
    img = np.squeeze(nib.load(mr_path).get_fdata(dtype=np.float32))
    lab = np.squeeze(np.asarray(nib.load(lab_path).dataobj)).astype(np.uint8)
    img, lab = fix_depth(img), fix_depth(lab)
    assert img.shape == (256, 256, 128), img.shape
    # image: 2x2x2 block mean; labels: nearest (strided) so values stay valid
    img = img.reshape(128, 2, 128, 2, 64, 2).mean(axis=(1, 3, 5))
    lab = lab[::2, ::2, ::2]
    img = (img - img.mean()) / (img.std() + 1e-8)
    return img.astype(np.float16), lab


def load_split(pairs):
    xs, ys = [], []
    for m, l in tqdm(pairs):
        x, y = load_volume(m, l)
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.stack(ys)


def _prepare(x, y, augment):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    if augment:  # random flip along axis 0 (assumed ~left-right symmetric anatomy)
        flip = tf.random.uniform(()) > 0.5
        x = tf.cond(flip, lambda: tf.reverse(x, [0]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, [0]), lambda: y)
    return x[..., tf.newaxis], tf.one_hot(y, NUM_CLASSES, dtype=tf.float32)


def build_dataset(x, y, batch_size, shuffle, augment):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=SEED)
    ds = ds.map(lambda a, b: _prepare(a, b, augment), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_hipmri_3d(batch_size=2, max_volumes=None):
    """
    Returns train_ds, val_ds, test_ds, test_names.
    max_volumes: use only the first N volumes (quick smoke test).
    """
    pairs = list_pairs()
    if max_volumes:
        pairs = pairs[:max_volumes]
    train_p, val_p, test_p = split_by_patient(pairs)
    print(f"Volumes train/val/test: {len(train_p)}/{len(val_p)}/{len(test_p)}")
    print("Test patients:", sorted({patient_id(m) for m, _ in test_p}))
    xtr, ytr = load_split(train_p)
    xva, yva = load_split(val_p)
    xte, yte = load_split(test_p)
    test_names = [os.path.basename(m) for m, _ in test_p]
    return (build_dataset(xtr, ytr, batch_size, True, True),
            build_dataset(xva, yva, batch_size, False, False),
            build_dataset(xte, yte, 1, False, False),
            test_names)
