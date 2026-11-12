"""Sliding-window inference tests: window placement, coverage, averaging, padding, cropping."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

import sliding_window
from sliding_window import pad_to_patch, sliding_window_logits, window_starts


def two_channel(batch: torch.Tensor) -> torch.Tensor:
    """Position-independent 'model': logits = [x, -2x]."""
    return torch.cat([batch, -2 * batch], dim=1)


def test_window_starts_for_hipmri_shapes():
    assert window_starts(256, 96) == [0, 48, 96, 144, 160]
    assert window_starts(128, 96) == [0, 32]
    assert window_starts(144, 96) == [0, 48]
    assert window_starts(256, 64) == [0, 32, 64, 96, 128, 160, 192]
    assert window_starts(128, 64) == [0, 32, 64]
    assert window_starts(64, 64) == [0]
    assert window_starts(40, 64) == [0]  # shorter than the patch: padded later
    with pytest.raises(ValueError):
        window_starts(128, 64, overlap=1.0)


@pytest.mark.parametrize("patch", [4, 5, 8, 64, 96])
def test_windows_cover_every_position_with_at_least_half_overlap(patch):
    for length in range(patch, 3 * patch + 7):
        starts = window_starts(length, patch)
        covered = torch.zeros(length, dtype=torch.bool)
        for s in starts:
            assert 0 <= s <= length - patch
            covered[s : s + patch] = True
        assert covered.all()
        assert all(b - a <= patch // 2 for a, b in pairwise(starts))


@pytest.mark.parametrize(
    "shape,size",
    [((10, 13, 7), (4, 6, 4)), ((3, 5, 20), (4, 8, 8)), ((8, 8, 8), (8, 8, 8))],
)
def test_reconstruction_is_exact_for_position_independent_model(shape, size):
    torch.manual_seed(0)
    image = torch.randn(1, *shape)
    logits = sliding_window_logits(two_channel, image, size, num_classes=2)
    assert logits.shape == (2, *shape) and logits.dtype == torch.float32
    assert torch.allclose(logits, two_channel(image.unsqueeze(0))[0], atol=1e-6)


def test_overlapping_windows_are_averaged_uniformly():
    image = torch.zeros(1, 1, 1, 6)
    calls = []

    def window_index(batch):  # each window predicts its own call number everywhere
        calls.append(len(calls))
        return torch.full_like(batch, float(len(calls) - 1))

    logits = sliding_window_logits(window_index, image, (1, 1, 4), num_classes=1)
    # Windows [0, 4) -> 0 and [2, 6) -> 1; positions 2 and 3 are covered by both.
    assert logits[0, 0, 0].tolist() == [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]


def test_padding_is_cropped_and_never_scored():
    image = torch.ones(1, 3, 5, 20)
    padded, original = pad_to_patch(image, (4, 8, 8))
    assert padded.shape == (1, 4, 8, 20) and original == (3, 5, 20)
    assert padded[:, 3:].abs().sum() == 0 and padded[:, :, 5:].abs().sum() == 0
    logits = sliding_window_logits(two_channel, image, (4, 8, 8), num_classes=2)
    assert logits.shape == (2, 3, 5, 20)
    assert torch.all(logits[0] == 1) and torch.all(logits[1] == -2)


def test_uncovered_voxels_raise(monkeypatch):
    monkeypatch.setattr(sliding_window, "window_starts", lambda length, patch, overlap: [0])
    with pytest.raises(RuntimeError, match="not covered"):
        sliding_window_logits(two_channel, torch.ones(1, 4, 4, 12), (4, 4, 4), num_classes=2)


def test_batch_size_does_not_change_result():
    torch.manual_seed(0)
    model = torch.nn.Conv3d(1, 6, kernel_size=3, padding=1).eval()
    image = torch.randn(1, 12, 16, 16)
    one = sliding_window_logits(model, image, (8, 8, 8), num_classes=6, batch_size=1)
    many = sliding_window_logits(model, image, (8, 8, 8), num_classes=6, batch_size=5)
    assert torch.allclose(one, many, atol=1e-5)


def test_pointwise_model_matches_whole_volume_prediction():
    torch.manual_seed(0)
    model = torch.nn.Conv3d(1, 6, kernel_size=1).eval()
    image = torch.randn(1, 20, 24, 12)
    with torch.no_grad():
        direct = model(image.unsqueeze(0))[0]
    logits = sliding_window_logits(model, image, (8, 8, 8), num_classes=6, batch_size=4)
    assert torch.allclose(logits, direct, atol=1e-5)


@pytest.mark.parametrize("size", [64, 96])
@pytest.mark.parametrize("shape", [(128, 256, 256), (144, 256, 256)])
def test_hipmri_volumes_fully_covered(size, shape):
    """Every voxel of a real-sized volume lies in at least one 50%-overlap window."""
    count = torch.zeros(shape, dtype=torch.int16)
    starts = [window_starts(n, size) for n in shape]
    for d in starts[0]:
        for h in starts[1]:
            for w in starts[2]:
                count[d : d + size, h : h + size, w : w + size] += 1
    assert count.min() >= 1


@pytest.mark.parametrize("size", [64, 96])
def test_reconstruction_with_hipmri_patch_sizes_and_padding(size):
    torch.manual_seed(0)
    image = torch.randn(1, 40, 130, 100)  # depth shorter than both patches
    logits = sliding_window_logits(two_channel, image, (size,) * 3, num_classes=2, batch_size=4)
    assert logits.shape == (2, 40, 130, 100)
    assert torch.allclose(logits, two_channel(image.unsqueeze(0))[0], atol=1e-6)
