"""3D patch pipeline tests: layout, alignment, padding, seeded sampling, split integrity.

All synthetic. Synthetic images satisfy image[h, w, z] = 1 + 10 * mask[h, w, z] + 100 * z
(see conftest), so any axis swap or image/mask misalignment breaks that identity.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import make_synthetic_volume
from torch.utils.data import DataLoader

from analysis.class_presence3d import parse_args, run_analysis
from dataset import (
    PatchDataset,
    PatchSampler,
    SliceDataset,
    VolumeSet,
    as_patch_size,
    extract_patch,
    load_manifest,
    to_model_layout,
    volume_equivalent_patches,
)

SYNTHETIC_COUNTS = {"train": 3, "val": 3, "test": 3}
SIZE = (4, 8, 8)  # (d, h, w); synthetic volumes are 32 x 32 x 6 ([H, W, D])


def synthetic_volumes(shapes, seed=0) -> VolumeSet:
    rng = np.random.default_rng(seed)
    pairs = [make_synthetic_volume(rng, shape) for shape in shapes]
    return VolumeSet(
        scan_ids=[f"S{i}" for i in range(len(pairs))],
        patient_ids=[f"P{i}" for i in range(len(pairs))],
        images=[image for image, _ in pairs],
        masks=[mask.astype(np.uint8) for _, mask in pairs],
    )


def assert_aligned(image: torch.Tensor, mask: torch.Tensor, d0: int) -> None:
    """image [1, D, H, W] and mask [D, H, W] satisfy the synthetic identity."""
    depth = torch.arange(d0, d0 + mask.shape[0]).view(-1, 1, 1)
    assert torch.equal(image[0], (1 + 10 * mask + 100 * depth).float())


# -----------------------------------------------------------------------------
# Layout and extraction
# -----------------------------------------------------------------------------
def test_to_model_layout_moves_depth_first():
    array = np.arange(2 * 3 * 4).reshape(2, 3, 4)  # [H, W, D]
    out = to_model_layout(array)
    assert out.shape == (4, 2, 3)
    for h, w, d in np.ndindex(array.shape):
        assert out[d, h, w] == array[h, w, d]


def test_as_patch_size():
    assert as_patch_size(64) == (64, 64, 64)
    assert as_patch_size([4, 8, 16]) == (4, 8, 16)
    with pytest.raises(ValueError):
        as_patch_size([4, 8])


def test_patch_shapes_dtypes_and_alignment():
    volumes = synthetic_volumes([(32, 32, 6)])
    image, mask = extract_patch(volumes.images[0], volumes.masks[0], (1, 5, 9), SIZE)
    assert image.shape == (1, *SIZE) and image.dtype == torch.float32
    assert mask.shape == SIZE and mask.dtype == torch.int64
    assert_aligned(image, mask, d0=1)
    expected = to_model_layout(volumes.masks[0])[1:5, 5:13, 9:17]
    assert np.array_equal(mask.numpy(), expected)


def test_full_volume_patch_preserves_every_label():
    volumes = synthetic_volumes([(32, 32, 6)])
    image, mask = extract_patch(volumes.images[0], volumes.masks[0], (0, 0, 0), (6, 32, 32))
    assert np.array_equal(mask.numpy(), to_model_layout(volumes.masks[0]))
    assert set(torch.unique(mask).tolist()) == {0, 1, 2, 3, 4, 5}
    assert_aligned(image, mask, d0=0)


def test_boundary_padding_is_zero_and_inside_is_unchanged():
    volumes = synthetic_volumes([(32, 32, 6)])
    image, mask = extract_patch(volumes.images[0], volumes.masks[0], (0, 28, 0), (8, 8, 8))
    # Depth 6 < 8 and h 28 + 8 > 32: both run past the edge.
    assert torch.all(image[0, 6:] == 0) and torch.all(mask[6:] == 0)
    assert torch.all(image[0, :, 4:] == 0) and torch.all(mask[:, 4:] == 0)
    assert_aligned(image[:, :6, :4], mask[:6, :4], d0=0)


def test_patch_planes_match_2d_slices():
    """The 3D path and the existing 2D SliceDataset read the same voxels."""
    volumes = synthetic_volumes([(32, 32, 6)])
    image, mask = extract_patch(volumes.images[0], volumes.masks[0], (0, 0, 0), (6, 32, 32))
    slices = SliceDataset(volumes)
    for z in range(6):
        slice_image, slice_mask = slices[z]
        assert torch.equal(image[:, z], slice_image)
        assert torch.equal(mask[z], slice_mask)


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
def test_sampling_mode_must_be_explicit_and_valid():
    volumes = synthetic_volumes([(32, 32, 6)])
    for mode in (None, "", "random"):
        with pytest.raises(ValueError, match="Sampling mode"):
            PatchSampler(volumes, SIZE, mode, seed=3710)
    with pytest.raises(ValueError, match="fg_prob"):
        PatchSampler(volumes, SIZE, "class_balanced", seed=3710)
    with pytest.raises(ValueError, match="fg_prob"):
        PatchSampler(volumes, SIZE, "class_balanced", seed=3710, fg_prob=1.5)
    with pytest.raises(ValueError, match="fg_prob"):
        PatchSampler(volumes, SIZE, "uniform", seed=3710, fg_prob=0.3)


@pytest.mark.parametrize("mode,fg_prob", [("uniform", None), ("class_balanced", 0.5)])
def test_plans_are_deterministic(mode, fg_prob):
    volumes = synthetic_volumes([(32, 32, 6), (32, 32, 10)])

    def plan(seed, epoch):
        return PatchSampler(volumes, SIZE, mode, seed, fg_prob).plan(epoch, 50)

    assert plan(3710, 1) == plan(3710, 1)  # fresh sampler, same result: no hidden state
    assert plan(3710, 1) != plan(3710, 2)
    assert plan(3710, 1) != plan(3711, 1)


@pytest.mark.parametrize("mode,fg_prob", [("uniform", None), ("class_balanced", 1.0)])
def test_starts_respect_each_volume_depth_and_patches_stay_aligned(mode, fg_prob):
    shapes = [(32, 32, 6), (32, 32, 10), (24, 40, 8)]  # variable depth and in-plane size
    volumes = synthetic_volumes(shapes)
    sampler = PatchSampler(volumes, SIZE, mode, seed=3710, fg_prob=fg_prob)
    plan = sampler.plan(epoch=1, count=300)
    assert {v for v, *_ in plan} == {0, 1, 2}
    for v, d, h, w in plan:
        height, width, depth = shapes[v]
        assert 0 <= d <= depth - SIZE[0]
        assert 0 <= h <= height - SIZE[1] and 0 <= w <= width - SIZE[2]
    for v, *start in plan[:40]:
        image, mask = extract_patch(volumes.images[v], volumes.masks[v], start, SIZE)
        assert_aligned(image, mask, d0=start[0])


def test_patch_larger_than_volume_starts_at_zero():
    volumes = synthetic_volumes([(32, 32, 6)])
    plan = PatchSampler(volumes, (8, 8, 8), "uniform", seed=3710).plan(1, 20)
    assert all(d == 0 for _, d, _, _ in plan)


def test_class_balanced_raises_rare_class_exposure():
    """One tiny class-5 cube: uniform patches rarely contain it, centred ones often do."""
    mask = np.zeros((32, 32, 16), dtype=np.uint8)
    mask[:16] = 2
    mask[20:22, 20:22, 10:12] = 5
    image = (1 + 10 * mask + 100 * np.arange(16)[None, None, :]).astype(np.float32)
    volumes = VolumeSet(["S0"], ["P0"], [image], [mask])

    def share_with_class5(sampler):
        hits = 0
        for v, *start in sampler.plan(1, 400):
            _, patch_mask = extract_patch(volumes.images[v], volumes.masks[v], start, SIZE)
            hits += bool((patch_mask == 5).any())
        return hits / 400

    uniform = share_with_class5(PatchSampler(volumes, SIZE, "uniform", seed=3710))
    balanced = share_with_class5(
        PatchSampler(volumes, SIZE, "class_balanced", seed=3710, fg_prob=1.0)
    )
    assert balanced >= 0.4  # half the centred patches target class 5 (classes 2 and 5 present)
    assert balanced > uniform + 0.2


@pytest.mark.parametrize("size", [64, 96])
@pytest.mark.parametrize("mode,fg_prob", [("uniform", None), ("class_balanced", 1.0)])
def test_hipmri_patch_sizes_on_real_scan_shapes(size, mode, fg_prob):
    """64^3 and 96^3 patches from 256x256x128 and 256x256x144 volumes (the HipMRI shapes)."""
    volumes = synthetic_volumes([(256, 256, 128), (256, 256, 144)])
    sampler = PatchSampler(volumes, size, mode, seed=3710, fg_prob=fg_prob)
    plan = sampler.plan(epoch=1, count=12)
    assert plan == PatchSampler(volumes, size, mode, seed=3710, fg_prob=fg_prob).plan(1, 12)
    for v, *start in plan:
        depth = volumes.masks[v].shape[2]
        assert 0 <= start[0] <= depth - size and all(0 <= s <= 256 - size for s in start[1:])
        image, mask = extract_patch(volumes.images[v], volumes.masks[v], start, sampler.size)
        assert image.shape == (1, size, size, size) and image.dtype == torch.float32
        assert mask.shape == (size, size, size) and mask.dtype == torch.int64
        assert set(torch.unique(mask).tolist()) <= {0, 1, 2, 3, 4, 5}
        assert_aligned(image, mask, d0=start[0])
        if mode == "class_balanced":
            assert (mask > 0).any()


@pytest.mark.parametrize("size", [64, 96])
@pytest.mark.parametrize("mode,fg_prob", [("uniform", None), ("class_balanced", 1.0)])
def test_volume_shallower_than_patch_is_zero_padded(size, mode, fg_prob):
    volumes = synthetic_volumes([(100, 120, 40)])  # every axis shorter than 96, depth < 64
    sampler = PatchSampler(volumes, size, mode, seed=3710, fg_prob=fg_prob)
    for v, *start in sampler.plan(epoch=1, count=10):
        assert start[0] == 0
        image, mask = extract_patch(volumes.images[v], volumes.masks[v], start, sampler.size)
        assert image.shape == (1, size, size, size)
        assert torch.all(image[0, 40:] == 0) and torch.all(mask[40:] == 0)
        h_end, w_end = min(100 - start[1], size), min(120 - start[2], size)
        assert_aligned(image[:, :40, :h_end, :w_end], mask[:40, :h_end, :w_end], d0=0)


def test_volume_equivalent_patches_rounds_up():
    volumes = synthetic_volumes([(32, 32, 6), (32, 32, 6)])  # 12,288 voxels
    assert volume_equivalent_patches(volumes, (8, 8, 8)) == 24  # exact
    assert volume_equivalent_patches(volumes, (4, 8, 8)) == 48
    assert volume_equivalent_patches(volumes, (5, 8, 8)) == 39  # 38.4 -> 39


def test_patch_dataset_batches_through_dataloader():
    volumes = synthetic_volumes([(32, 32, 6), (32, 32, 10)])
    plan = PatchSampler(volumes, SIZE, "uniform", seed=3710).plan(1, 10)
    loader = DataLoader(PatchDataset(volumes, plan, SIZE), batch_size=4, shuffle=False)
    images, masks = next(iter(loader))
    assert images.shape == (4, 1, *SIZE) and masks.shape == (4, *SIZE)
    assert len(PatchDataset(volumes, plan, SIZE)) == 10


# -----------------------------------------------------------------------------
# Split integrity and the class-presence analysis
# -----------------------------------------------------------------------------
def test_patches_come_only_from_the_requested_split(synthetic_manifest):
    frame = load_manifest(synthetic_manifest, SYNTHETIC_COUNTS, expected_sha256=None)
    volumes = VolumeSet.from_manifest(frame, "train")
    plan = PatchSampler(volumes, SIZE, "uniform", seed=3710).plan(1, 100)
    patients = {volumes.patient_ids[v] for v, *_ in plan}
    train = set(frame.loc[frame["split"] == "train", "patient_id"])
    other = set(frame.loc[frame["split"] != "train", "patient_id"])
    assert patients <= train and not patients & other


def test_class_presence_analysis_reads_training_split_only(synthetic_manifest, tmp_path):
    # Remove every val/test volume: the analysis must still run.
    frame = pd.read_csv(synthetic_manifest)
    for path in frame.loc[frame["split"] != "train", ["image_path", "mask_path"]].values.ravel():
        os.remove(path)
    out = tmp_path / "presence.json"
    args = parse_args(
        [
            "--metadata",
            str(synthetic_manifest),
            "--out",
            str(out),
            "--patches",
            "200",
            "--patch-sizes",
            "4",
            "--fg-prob",
            "1.0",
        ]
    )
    report = run_analysis(args, SYNTHETIC_COUNTS, expected_sha256=None)
    assert report["split"] == "train" and report["volumes"] == 6
    assert set(report["results"]) == {"4^3 uniform", "4^3 class_balanced fg_prob=1.0"}
    balanced = report["results"]["4^3 class_balanced fg_prob=1.0"]
    assert balanced["patches_without_foreground"] == 0.0
    assert json.loads(out.read_text()) == report
    again = run_analysis(args, SYNTHETIC_COUNTS, expected_sha256=None)
    assert again["results"] == report["results"]  # seeded
