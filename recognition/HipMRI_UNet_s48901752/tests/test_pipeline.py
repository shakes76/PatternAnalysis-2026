"""End-to-end smoke tests of train.py and predict.py on tiny synthetic volumes.

These check that the pipeline runs and writes its outputs. They say nothing
about how well the model learns on real MRI.
"""

from __future__ import annotations

import json

import torch

import predict
import train


def _train_args(manifest, out_dir, *extra):
    return train.parse_args(
        [
            "--model",
            "unet2d_256",
            "--metadata",
            str(manifest),
            "--out-dir",
            str(out_dir),
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            *extra,
        ]
    )


def test_train_and_predict_smoke(synthetic_manifest, tmp_path):
    out_dir = tmp_path / "run"
    summary = train.run_training(_train_args(synthetic_manifest, out_dir), None, None)

    for name in (
        "history.csv",
        "summary.json",
        "best.pt",
        "last.pt",
        "loss_curves.png",
        "dice_curves.png",
    ):
        assert (out_dir / name).is_file(), name
    assert not (out_dir / "test_metrics.json").exists()  # test split untouched by default
    saved = json.loads((out_dir / "summary.json").read_text())
    assert saved["trainable_parameters"] == summary["trainable_parameters"]
    assert saved["best_epoch"] in (1, 2)
    assert set(saved["best_metrics"]["global"]["dice_per_class"]) >= {"bladder", "prostate"}
    assert saved["best_metrics"]["per_volume"]["volumes"] == 6

    checkpoint = torch.load(out_dir / "best.pt", weights_only=False)
    assert checkpoint["variant"] == "unet2d_256" and checkpoint["bottleneck_channels"] == 256

    pred_dir = tmp_path / "pred"
    results = predict.run_prediction(
        predict.parse_args(
            [
                "--checkpoint",
                str(out_dir / "best.pt"),
                "--metadata",
                str(synthetic_manifest),
                "--split",
                "val",
                "--out-dir",
                str(pred_dir),
                "--device",
                "cpu",
                "--num-overlays",
                "1",
            ]
        ),
        None,
        None,
    )
    assert results["per_volume"]["volumes"] == 6
    assert (pred_dir / "metrics.json").is_file()
    assert len(list(pred_dir.glob("*_overlay.png"))) == 1


def test_refuses_to_overwrite_run_dir(synthetic_manifest, tmp_path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("x")
    try:
        train.run_training(_train_args(synthetic_manifest, out_dir), None, None)
    except FileExistsError:
        return
    raise AssertionError("expected FileExistsError")


def test_overfit_mode_smoke(synthetic_manifest, tmp_path):
    out_dir = tmp_path / "overfit"
    summary = train.run_training(
        _train_args(synthetic_manifest, out_dir, "--overfit-subset", "8", "--batch-size", "8"),
        None,
        None,
    )
    assert summary["mode"] == "overfit_subset"
    assert len(summary["overfit_slices"]) == 8
    assert summary["overfit_diagnostic_target"] == 0.90
    assert summary["train_slices"] == 8
