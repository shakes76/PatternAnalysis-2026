"""Tiny temporary NIfTI integration test; does not train on the real dataset."""

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

import train
import predict
import evaluate_3d
import prepare_3d
from create_split import create_split


def main():
    """Exercise both architectures and all report-producing entrypoints on tiny scans."""
    torch.set_num_threads(2)
    dice, iou, _, _ = evaluate_3d.overlap_metrics(np.array([0, 1, 5, 5]), np.array([0, 1, 1, 5]), 6)
    assert np.isclose(dice[1], 2 / 3) and np.isclose(iou[1], 0.5)
    assert dice[0] == 1 and np.isnan(dice[2])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        images, masks = root / "semantic_MRs", root / "semantic_labels_only"
        images.mkdir()
        masks.mkdir()
        rng = np.random.default_rng(42)
        for patient in ("B001", "B002", "B003"):
            image = rng.normal(size=(16, 17, 18)).astype(np.float32)
            mask = rng.integers(0, 6, size=image.shape, dtype=np.int16)
            nib.save(nib.Nifti1Image(image, np.eye(4)), images / f"{patient}_Week0_LFOV.nii.gz")
            nib.save(nib.Nifti1Image(mask, np.eye(4)), masks / f"{patient}_Week0_SEMANTIC.nii.gz")
        output = root / "run"
        generated = create_split(root, root / "reference_split.json")
        assert create_split(root, root / "reference_split.json") == generated
        assert sorted(Path(pair[0]).name for pair in generated["train"]) == [
            "B003_Week0_LFOV.nii.gz"
        ]
        changed = dict(generated, train=[])
        (root / "wrong_split.json").write_text(json.dumps(changed))
        try:
            create_split(root, root / "wrong_split.json")
        except ValueError:
            assert json.loads((root / "wrong_split.json").read_text()) == changed
        else:
            raise AssertionError("Conflicting reference split was overwritten")
        arguments = [
            "train.py",
            "--data-root",
            str(root),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--patch-size",
            "16",
            "16",
            "16",
            "--base-channels",
            "4",
            "--patches-per-volume",
            "1",
            "--cache-dir",
            str(root / "cache"),
            "--window-batch-size",
            "3",
            "--profile",
        ]
        with patch.object(sys, "argv", arguments + ["--epochs", "1"]):
            train.main()
        assert (output / "training_curves.png").is_file()
        # Older checkpoints lack the newly added command options; preserve compatibility.
        legacy = torch.load(output / "last.pt", weights_only=True)
        for key in ("evaluate_only", "evaluation_output"):
            legacy["config"].pop(key, None)
        torch.save(legacy, output / "last.pt")
        with patch.object(
            sys, "argv", arguments + ["--epochs", "2", "--resume", str(output / "last.pt")]
        ):
            train.main()
        state = torch.load(output / "last.pt", weights_only=True)
        assert state["epoch"] == 1 and state["best_dice"] >= 0
        with (output / "performance.csv").open() as stream:
            performance = list(csv.DictReader(stream))
        assert len(performance) == 2
        assert all(int(row["batches"]) == 1 for row in performance)
        assert all(
            float(row["epoch_seconds"]) >= float(row["training_seconds"]) for row in performance
        )
        with (output / "metrics.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == 2
        predicted = root / "prediction.nii.gz"
        with patch.object(
            sys,
            "argv",
            [
                "predict.py",
                "--checkpoint",
                str(output / "best.pt"),
                "--image",
                str(images / "B001_Week0_LFOV.nii.gz"),
                "--output",
                str(predicted),
                "--device",
                "cpu",
            ],
        ):
            predict.main()
        result = nib.load(predicted)
        assert result.shape == (16, 17, 18)
        assert np.allclose(result.affine, np.eye(4))
        assert set(np.unique(result.get_fdata())) <= set(range(6))
        assert (root / "prediction_overlay.png").is_file()
        # Optional comparison labels must not alter the segmentation itself.
        without_mask = result.get_fdata().copy()
        with patch.object(
            sys,
            "argv",
            [
                "predict.py",
                "--checkpoint",
                str(output / "best.pt"),
                "--image",
                str(images / "B001_Week0_LFOV.nii.gz"),
                "--output",
                str(predicted),
                "--device",
                "cpu",
                "--mask",
                str(masks / "B001_Week0_SEMANTIC.nii.gz"),
            ],
        ):
            predict.main()
        assert np.array_equal(without_mask, nib.load(predicted).get_fdata())
        assert (root / "prediction_comparison.png").is_file()
        # Evaluation must neither choose a new split nor modify the selected weights.
        original_checkpoint = hashlib.sha256((output / "best.pt").read_bytes()).hexdigest()
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--evaluate-only",
                "--resume",
                str(output / "best.pt"),
                "--evaluation-output",
                str(root / "evaluation"),
                "--device",
                "cpu",
            ],
        ):
            with patch.object(
                train, "discover_pairs", side_effect=AssertionError("Evaluation must not resplit")
            ):
                train.main()
        assert hashlib.sha256((output / "best.pt").read_bytes()).hexdigest() == original_checkpoint
        assert (root / "evaluation" / "summary.json").is_file()
        assert json.loads((root / "evaluation" / "summary.json").read_text())["split"] == "test"
        assert len(list((root / "evaluation").glob("*.png"))) == 1
        # Repeat resume with the baseline and demand identical patient membership.
        baseline_args = arguments.copy()
        baseline_args[baseline_args.index(str(output))] = str(root / "baseline")
        with patch.object(
            sys, "argv", baseline_args + ["--epochs", "1", "--architecture", "baseline"]
        ):
            train.main()
        with patch.object(
            sys,
            "argv",
            baseline_args
            + [
                "--epochs",
                "2",
                "--architecture",
                "baseline",
                "--resume",
                str(root / "baseline" / "last.pt"),
            ],
        ):
            train.main()
        baseline = torch.load(root / "baseline" / "best.pt", weights_only=True)
        assert baseline["split"] == state["split"]
        assert torch.load(root / "baseline" / "last.pt", weights_only=True)["epoch"] == 1
        # A skipped validation measurement must produce a gap, not a false zero.
        history = root / "plot_test.csv"
        history.write_text("epoch,train_loss,dice_5\n1,2.0,\n2,1.5,0.4\n3,1.2,\n")
        epochs, losses, values = train.plot_training_curves(history)
        assert epochs == [1, 2, 3] and losses == [2.0, 1.5, 1.2]
        assert np.isnan(values[0]) and values[1] == 0.4 and np.isnan(values[2])
        with patch.object(
            sys,
            "argv",
            [
                "prepare_3d.py",
                "--data-root",
                str(root),
                "--output-root",
                str(root / "downsampled"),
                "--scale",
                "0.5",
                "--reference-split",
                str(output / "split.json"),
            ],
        ):
            prepare_3d.main()
        down_image = nib.load(root / "downsampled" / "semantic_MRs" / "B001_Week0_LFOV.nii.gz")
        down_mask = nib.load(
            root / "downsampled" / "semantic_labels_only" / "B001_Week0_SEMANTIC.nii.gz"
        )
        assert down_image.shape == down_mask.shape == (8, 8, 9)
        assert np.allclose(down_image.affine, down_mask.affine)
        assert np.count_nonzero(down_mask.get_fdata() == 5) > 0
        manifest = json.loads((root / "downsampled" / "preprocessing_manifest.json").read_text())
        assert manifest["complete"] and manifest["cases"] == 3
        assert manifest["reference_split_sha256"] == prepare_3d.sha256(output / "split.json")
        assert manifest["panel_training_cases"] == ["B003_Week0_LFOV.nii.gz"]
        assert len(list((root / "downsampled").glob("*_preprocessing.png"))) == 1
        record = next(r for r in manifest["records"] if r["case"] == "B001_Week0_LFOV.nii.gz")
        assert record["alignment_verified"] and record["labels_before"] == record["labels_after"]
        assert np.allclose(record["output_spacing"], [2, 17 / 8, 2])
        assert np.allclose(down_image.affine[:3, 3], [0.5, (17 / 8 - 1) / 2, 0.5])
        expected_ratio = (
            record["prostate_voxels_after"]
            / record["prostate_voxels_before"]
            * abs(np.linalg.det(down_image.affine[:3, :3]))
        )
        assert np.isclose(record["prostate_physical_volume_ratio"], expected_ratio)
        assert record["source_image_sha256"] == prepare_3d.sha256(images / "B001_Week0_LFOV.nii.gz")
        assert record["output_mask_sha256"] == prepare_3d.sha256(
            root / "downsampled" / "semantic_labels_only" / "B001_Week0_SEMANTIC.nii.gz"
        )
        # A tiny foreground class at an unsampled position must fail closed.
        tiny = np.zeros((16, 17, 18), dtype=np.int16)
        tiny[0, 0, 0] = 5
        nib.save(nib.Nifti1Image(tiny, np.eye(4)), masks / "B001_Week0_SEMANTIC.nii.gz")
        with patch.object(
            sys,
            "argv",
            [
                "prepare_3d.py",
                "--data-root",
                str(root),
                "--output-root",
                str(root / "lost_class"),
                "--scale",
                "0.5",
            ],
        ):
            try:
                prepare_3d.main()
            except ValueError as error:
                assert "loses labels" in str(error)
            else:
                raise AssertionError("Disappearing class was not rejected")
        assert not (root / "lost_class" / "preprocessing_manifest.json").exists()
    print("TRAIN / RESUME / INFERENCE INTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()
