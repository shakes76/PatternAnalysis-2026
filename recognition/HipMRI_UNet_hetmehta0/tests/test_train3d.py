"""Tests of train3d.py on tiny synthetic volumes (CPU only).

These check that 3D training, sliding-window validation, checkpoints and logs work
and are reproducible. They say nothing about how well the model learns on real MRI.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import train3d
from dataset import VolumeSet
from losses import DiceCrossEntropyLoss
from modules import UNet3D

SYNTHETIC_COUNTS = {"train": 3, "val": 3, "test": 3}


def _args(manifest, out_dir, *extra):
    return train3d.parse_args(
        [
            "--model",
            "unet3d_256",
            "--metadata",
            str(manifest),
            "--out-dir",
            str(out_dir),
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--patch-size",
            "16",
            "--patches-per-epoch",
            "4",
            "--device",
            "cpu",
            *extra,
        ]
    )


def _run(args):
    return train3d.run_training(args, SYNTHETIC_COUNTS, expected_sha256=None)


def _remove_test_split(manifest):
    """Delete every test-split NIfTI file: training must not need them."""
    frame = pd.read_csv(manifest)
    for path in frame.loc[frame["split"] == "test", ["image_path", "mask_path"]].values.ravel():
        os.remove(path)


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
def test_defaults(tmp_path):
    args = train3d.parse_args(["--model", "unet3d_512", "--metadata", "m.csv", "--out-dir", "r"])
    assert args.patch_size == 96 and args.sampling == "class_balanced" and args.fg_prob == 0.33
    assert args.overlap == 0.5 and args.seed == 3710 and args.amp is True
    assert args.patches_per_epoch is None  # volume-equivalent by default


def test_argument_validation():
    base = ["--model", "unet3d_256", "--out-dir", "r"]
    with pytest.raises(SystemExit):
        train3d.parse_args([*base, "--metadata", "m.csv", "--patch-size", "40"])
    with pytest.raises(SystemExit):
        train3d.parse_args(base)  # --metadata needed for training
    with pytest.raises(SystemExit):
        train3d.parse_args(["--model", "unet2d_256", "--metadata", "m.csv", "--out-dir", "r"])
    uniform = train3d.parse_args([*base, "--metadata", "m.csv", "--sampling", "uniform"])
    assert uniform.fg_prob is None
    assert train3d.parse_args([*base, "--memory-probe"]).metadata is None


# -----------------------------------------------------------------------------
# Full-volume validation
# -----------------------------------------------------------------------------
class LabelFromImage(nn.Module):
    """'Perfect' model for synthetic images, which encode the label.

    Synthetic images satisfy image = 1 + 10 * label + 100 * z (see conftest).
    """

    def forward(self, x):
        label = torch.round(torch.remainder(x - 1, 100) / 10).clamp(0, 5).long()[:, 0]
        return 10.0 * nn.functional.one_hot(label, 6).movedim(-1, 1).float()


def test_sliding_window_validation_scores_whole_volumes():
    from conftest import make_synthetic_volume

    rng = np.random.default_rng(0)
    pairs = [make_synthetic_volume(rng, shape) for shape in [(32, 32, 6), (24, 40, 10)]]
    volumes = VolumeSet(
        ["S0", "S1"], ["P0", "P1"], [i for i, _ in pairs], [m.astype(np.uint8) for _, m in pairs]
    )
    result = train3d.evaluate_volumes3d(
        LabelFromImage(),
        volumes,
        DiceCrossEntropyLoss(),
        torch.device("cpu"),
        (8, 16, 16),
        3,
        False,
        0.5,
    )
    assert result["global"]["mean_foreground_dice"] == 1.0
    assert all(d == 1.0 for d in result["global"]["dice_per_class"].values())
    assert result["per_volume"]["volumes"] == 2
    assert sum(map(sum, result["confusion_matrix"])) == 32 * 32 * 6 + 24 * 40 * 10
    assert result["loss"]["ce"] < 1e-3


# -----------------------------------------------------------------------------
# Training runs
# -----------------------------------------------------------------------------
def test_train_smoke_without_test_split(synthetic_manifest, tmp_path):
    _remove_test_split(synthetic_manifest)
    out_dir = tmp_path / "run"
    summary = _run(_args(synthetic_manifest, out_dir))

    for name in (
        "history.csv",
        "summary.json",
        "best.pt",
        "last.pt",
        "loss_curves.png",
        "dice_curves.png",
    ):
        assert (out_dir / name).is_file(), name
    assert not (out_dir / "test_metrics.json").exists()
    saved = json.loads((out_dir / "summary.json").read_text())
    assert saved["trainable_parameters"] == 14_975_238
    assert saved["patch_size_dhw"] == [16, 16, 16] and saved["patches_per_epoch"] == 4
    assert saved["best_epoch"] in (1, 2)
    assert saved["best_metrics"]["per_volume"]["volumes"] == 6  # every val scan, whole
    assert set(saved["best_metrics"]["global"]["dice_per_class"]) >= {"bladder", "prostate"}
    assert saved["runtime"]["training_seconds"] > 0
    assert "peak_gpu_train_allocated_mib" in saved["memory"]  # None on CPU
    assert summary["args"]["fg_prob"] == 0.33

    history = pd.read_csv(out_dir / "history.csv")
    assert list(history["epoch"]) == [1, 2]
    assert {"val_mean_foreground_dice", "val_dice_prostate", "plan_seconds"} <= set(history)

    checkpoint = torch.load(out_dir / "best.pt", weights_only=False)
    assert checkpoint["variant"] == "unet3d_256" and checkpoint["bottleneck_channels"] == 256
    assert checkpoint["patch_size"] == 16
    model = UNet3D(256)
    model.load_state_dict(checkpoint["model_state"])


def test_training_is_reproducible(synthetic_manifest, tmp_path):
    first = _run(_args(synthetic_manifest, tmp_path / "a"))
    second = _run(_args(synthetic_manifest, tmp_path / "b"))
    a = pd.read_csv(tmp_path / "a" / "history.csv")
    b = pd.read_csv(tmp_path / "b" / "history.csv")
    assert np.allclose(a["train_loss"], b["train_loss"])
    assert first["final_metrics"]["confusion_matrix"] == second["final_metrics"]["confusion_matrix"]
    wa = torch.load(tmp_path / "a" / "last.pt", weights_only=False)["model_state"]
    wb = torch.load(tmp_path / "b" / "last.pt", weights_only=False)["model_state"]
    assert all(torch.allclose(wa[k].float(), wb[k].float()) for k in wa)


def test_different_seed_changes_the_patch_plan(synthetic_manifest, tmp_path):
    a = _run(_args(synthetic_manifest, tmp_path / "a"))
    b = _run(_args(synthetic_manifest, tmp_path / "b", "--seed", "1"))
    assert a["final_metrics"]["loss"] != b["final_metrics"]["loss"]


def test_refuses_to_overwrite_run_dir(synthetic_manifest, tmp_path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "best.pt").write_text("existing")
    with pytest.raises(FileExistsError):
        _run(_args(synthetic_manifest, out_dir))


def test_uniform_sampling_runs(synthetic_manifest, tmp_path):
    summary = _run(_args(synthetic_manifest, tmp_path / "run", "--sampling", "uniform"))
    assert summary["args"]["sampling"] == "uniform" and summary["args"]["fg_prob"] is None


def test_overfit_mode_smoke(synthetic_manifest, tmp_path):
    out_dir = tmp_path / "overfit"
    summary = _run(_args(synthetic_manifest, out_dir, "--overfit-patches", "2", "--epochs", "3"))
    assert summary["mode"] == "overfit_patches"
    assert len(summary["overfit_plan"]) == 2
    assert summary["overfit_diagnostic_target"] == 0.90
    assert 0.0 <= summary["overfit_final_mean_foreground_dice"] <= 1.0
    history = pd.read_csv(out_dir / "history.csv")
    assert "subset_mean_foreground_dice" in history and "val_loss" not in history


# -----------------------------------------------------------------------------
# Memory probe
# -----------------------------------------------------------------------------
def test_memory_probe_needs_no_data(tmp_path):
    out_dir = tmp_path / "probe"
    args = train3d.parse_args(
        [
            "--model",
            "unet3d_256",
            "--out-dir",
            str(out_dir),
            "--memory-probe",
            "--device",
            "cpu",
            "--patch-size",
            "32",  # 16^3 at batch 1 leaves a 1x1x1 bottleneck, which BatchNorm cannot train on
            "--probe-batch-sizes",
            "1",
            "2",
            "--probe-steps",
            "1",
            "--probe-volume",
            "20",
            "40",
            "32",
        ]
    )
    report = train3d.run_memory_probe(args)
    assert set(report["training_steps"]) == {"1", "2"}
    assert all(r["status"] == "ok" for r in report["training_steps"].values())
    assert report["training_steps"]["1"]["allocated_mib"] is None  # CPU: no CUDA counters
    assert report["sliding_window_inference"]["volume_dhw"] == [20, 40, 32]
    assert json.loads((out_dir / "memory_probe.json").read_text()) == report
    assert [p.name for p in out_dir.iterdir()] == ["memory_probe.json"]
