"""3D pipeline checks on real HipMRI volumes (skipped unless the NIfTI files are readable).

Run on the GPU Lab / Rangpur with HIPMRI_METADATA pointing at the frozen manifest.
CPU only: a few scans are loaded and no network is trained.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from dataset import (
    NUM_CLASSES,
    PatchDataset,
    PatchSampler,
    SliceDataset,
    VolumeSet,
    extract_patch,
    load_manifest,
    to_model_layout,
)
from sliding_window import sliding_window_logits

pytestmark = pytest.mark.requires_data
SEED = 3710


@pytest.fixture(scope="module")
def real_frame():
    return load_manifest(os.environ["HIPMRI_METADATA"])  # frozen 26/6/6 split and SHA-256


@pytest.fixture(scope="module")
def train_volumes(real_frame):
    """First two train scans plus the one deeper (x144) train scan, if present."""
    train = real_frame[real_frame["split"] == "train"]
    chosen = list(train["scan_id"].iloc[:2])
    deep = train.loc[~train["shape"].str.endswith("x128"), "scan_id"]
    chosen += list(deep.iloc[:1])
    return VolumeSet.from_manifest(real_frame, "train", scan_ids=chosen)


@pytest.mark.parametrize("size", [64, 96])
@pytest.mark.parametrize("mode,fg_prob", [("uniform", None), ("class_balanced", 1.0)])
def test_real_patches_shapes_labels_and_determinism(train_volumes, size, mode, fg_prob):
    sampler = PatchSampler(train_volumes, size, mode, seed=SEED, fg_prob=fg_prob)
    plan = sampler.plan(epoch=1, count=16)
    again = PatchSampler(train_volumes, size, mode, seed=SEED, fg_prob=fg_prob).plan(1, 16)
    assert plan == again
    for v, d, h, w in plan:
        height, width, depth = train_volumes.masks[v].shape
        assert 0 <= d <= depth - size and 0 <= h <= height - size and 0 <= w <= width - size
    for image, mask in PatchDataset(train_volumes, plan, size):
        assert image.shape == (1, size, size, size) and image.dtype == torch.float32
        assert mask.shape == (size, size, size) and mask.dtype == torch.int64
        assert torch.isfinite(image).all()
        assert set(torch.unique(mask).tolist()) <= set(range(NUM_CLASSES))
        if mode == "class_balanced":
            assert (mask > 0).any()


def test_real_patch_planes_match_2d_slices(train_volumes):
    """Each depth plane of a 3D patch equals the cropped 2D slice used by the 2D baselines."""
    slices = SliceDataset(train_volumes)
    offsets = np.cumsum([0] + [m.shape[2] for m in train_volumes.masks])
    plan = PatchSampler(train_volumes, 64, "class_balanced", SEED, fg_prob=1.0).plan(2, 4)
    for v, d, h, w in plan:
        image, mask = extract_patch(
            train_volumes.images[v], train_volumes.masks[v], (d, h, w), (64,) * 3
        )
        for k in (0, 31, 63):
            slice_image, slice_mask = slices[int(offsets[v]) + d + k]
            assert torch.equal(image[:, k], slice_image[:, h : h + 64, w : w + 64])
            assert torch.equal(mask[k], slice_mask[h : h + 64, w : w + 64])


def test_real_variable_depth_is_kept(train_volumes, real_frame):
    shapes = real_frame.set_index("scan_id").loc[train_volumes.scan_ids, "shape"]
    for mask, expected in zip(train_volumes.masks, shapes, strict=True):
        assert "x".join(map(str, mask.shape)) == expected


def test_real_train_patches_never_touch_other_splits(real_frame):
    """Asking for scans of every split still loads (and samples) training scans only."""
    requested = real_frame.groupby("split")["scan_id"].head(3).tolist()  # 3 per split
    volumes = VolumeSet.from_manifest(real_frame, "train", scan_ids=requested, normalise=False)
    assert len(volumes) == 3
    plan = PatchSampler(volumes, 96, "uniform", seed=SEED).plan(1, 100)
    patients = {volumes.patient_ids[v] for v, *_ in plan}
    train = set(real_frame.loc[real_frame["split"] == "train", "patient_id"])
    other = set(real_frame.loc[real_frame["split"] != "train", "patient_id"])
    assert patients and patients <= train and not patients & other


@pytest.mark.parametrize("size", [64, 96])
def test_real_sliding_window_covers_and_reconstructs(train_volumes, size):
    """A pointwise model gives the same logits by sliding window as on the whole volume."""
    torch.manual_seed(0)
    model = torch.nn.Conv3d(1, NUM_CLASSES, kernel_size=1).eval()
    for v in (0, len(train_volumes) - 1):  # a x128 scan and (if present) the x144 scan
        image = torch.from_numpy(to_model_layout(train_volumes.images[v])).unsqueeze(0)
        logits = sliding_window_logits(model, image, (size,) * 3, NUM_CLASSES, batch_size=8)
        with torch.no_grad():
            direct = model(image.unsqueeze(0))[0]
        assert logits.shape == (NUM_CLASSES, *image.shape[1:])
        assert torch.allclose(logits, direct, atol=1e-4)
        prediction = logits.argmax(0)
        assert prediction.shape == to_model_layout(train_volumes.masks[v]).shape
