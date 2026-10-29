"""Small synthetic checks for alignment, leakage, model shapes, and gradients."""

import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import VolumeDataset, mirror_left_right, read_manifest, split_by_patient  # noqa: E402
from metrics import case_metrics  # noqa: E402
from modules import ImprovedUNet3D, UNet2D  # noqa: E402
from prepare_manifest import collect  # noqa: E402
from predict import require_test_results, select_failure_cases  # noqa: E402
from train import _mean_case_metrics, model_loss, native_case_metrics, predict_volume  # noqa: E402


def synthetic_manifest(tmp_path: Path) -> Path:
    rows = []
    for i in range(3):
        image = np.zeros((16, 16, 16), dtype=np.float32)
        image[2:14, 2:14, 2:14] = 1.0 + i
        image[6:10, 6:10, 5:11] = 2.0 + i
        mask = np.zeros_like(image)
        mask[6:10, 6:10, 5:11] = 5
        image_path, mask_path = tmp_path / f"p{i}_image.nii.gz", tmp_path / f"p{i}_mask.nii.gz"
        nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
        nib.save(nib.Nifti1Image(mask[..., None] if i == 0 else mask, np.eye(4)), mask_path)
        rows.append({"case_id": f"p{i}", "patient_id": f"patient_{i}",
                     "image": image_path.name, "mask": mask_path.name})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_patient_split_and_alignment(tmp_path: Path) -> None:
    cases = read_manifest(synthetic_manifest(tmp_path))
    split = split_by_patient(cases)
    groups = [set(c.patient_id for c in split[name]) for name in ("train", "val", "test")]
    assert all(groups)
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
    sample = VolumeDataset(cases[:1], (16, 16, 16), 5)[0]
    assert sample["image"].shape == (1, 16, 16, 16)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].sum() == 4 * 4 * 6
    assert set(torch.unique(sample["mask"]).tolist()) == {0, 1}
    native = native_case_metrics(sample["mask"].numpy(), cases[0], 5, None, None)
    assert native["native_prostate_dice"] == 1.0
    assert native["native_hd95_mm"] == 0.0
    with pytest.raises(ValueError, match="absent"):
        VolumeDataset(cases[:1], (16, 16, 16), 4)[0]


def test_left_right_mirror_keeps_image_and_labels_aligned() -> None:
    sample = {
        "image": torch.arange(8).reshape(1, 2, 2, 2),
        "mask": torch.arange(8).reshape(2, 2, 2),
        "anatomy": torch.arange(8).reshape(2, 2, 2),
        "case_id": "example",
    }
    mirrored = mirror_left_right(sample)
    assert torch.equal(mirrored["image"][0], mirrored["mask"])
    assert torch.equal(mirrored["mask"], mirrored["anatomy"])
    assert mirrored["mask"][0, 0].tolist() == [1, 0]
    assert mirrored["case_id"] == "example"


def test_manifest_suffix_pairing(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    (images / "Case_004_Week0_LFOV.nii.gz").touch()
    (masks / "Case_004_Week0_SEMANTIC_LFOV.nii.gz").touch()
    image_keys = collect(images, "_LFOV")
    mask_keys = collect(masks, "_SEMANTIC_LFOV")
    assert image_keys.keys() == mask_keys.keys() == {"Case_004_Week0"}
    with pytest.raises(ValueError, match="Expected suffix"):
        collect(masks, "_MISSING")


def test_failure_panels_cover_different_patients() -> None:
    cases = [
        {"sample": {"case_id": case_id, "patient_id": patient_id},
         "improved_metrics": {"prostate_dice": dice}}
        for case_id, patient_id, dice in (
            ("a", "patient_1", 0.0),
            ("b", "patient_1", 0.1),
            ("c", "patient_2", 0.2),
        )
    ]
    selected = select_failure_cases(cases, count=2)
    assert [case["sample"]["case_id"] for case in selected] == ["a", "c"]


def test_validation_only_run_cannot_open_test(tmp_path: Path) -> None:
    report = {"provenance": {"validation_only": True}}
    (tmp_path / "results.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="validation-only"):
        require_test_results(tmp_path)


def test_patient_balanced_summary() -> None:
    rows = [
        {"case_id": "a", "patient_id": "p1", "prostate_dice": 0.0},
        {"case_id": "b", "patient_id": "p1", "prostate_dice": 0.0},
        {"case_id": "c", "patient_id": "p2", "prostate_dice": 1.0},
    ]
    summary = _mean_case_metrics(rows)
    assert summary["prostate_dice"] == pytest.approx(1 / 3)
    assert summary["patient_mean_prostate_dice"] == pytest.approx(0.5)


def test_models_loss_prediction_and_metrics() -> None:
    torch.set_num_threads(2)
    volume = torch.randn(1, 1, 16, 16, 16)
    mask = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    mask[:, 5:11, 5:11, 5:11] = 1
    advanced = ImprovedUNet3D(width=4)
    output = advanced(volume)
    assert output["logits"].shape == (1, 2, 16, 16, 16)
    loss = model_loss(output, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model_loss(output, mask, foreground_weight=10.0))
    assert advanced.head.weight.grad is not None
    baseline = UNet2D(width=4)
    assert baseline(volume[:, :, 0]).shape == (1, 2, 16, 16)
    pred = predict_volume(advanced, volume[0], "3d", torch.device("cpu"))
    assert pred.shape == (16, 16, 16)
    perfect = case_metrics(mask[0].numpy(), mask[0].numpy())
    assert perfect["prostate_dice"] == 1.0
    assert perfect["hd95_resampled_voxels"] == 0.0
    extension = mask[0].numpy().copy()
    extension[0, 0, 0] = 1
    extension_metrics = case_metrics(extension, mask[0].numpy())
    assert extension_metrics["outside_extent_false_positive_voxels"] == 1
    assert extension_metrics["outside_extent_false_positive_slices"] == 1
    one_voxel_truth = np.zeros((4, 4, 4), dtype=bool)
    one_voxel_pred = np.zeros_like(one_voxel_truth)
    one_voxel_truth[1, 1, 1] = True
    one_voxel_pred[1, 1, 2] = True
    physical = case_metrics(
        one_voxel_pred, one_voxel_truth, spacing_dhw=(1.0, 1.0, 2.0)
    )
    assert physical["hd95_mm"] == pytest.approx(2.0)
    anatomy = np.zeros((16, 16, 16), dtype=int)
    anatomy[0, 0, :4] = 3
    pred_with_leak = mask[0].numpy().copy()
    pred_with_leak[0, 0, 0] = 1
    leakage = case_metrics(pred_with_leak, mask[0].numpy(), anatomy, bladder_label=3)
    assert leakage["false_positive_bladder_voxels"] == 1
    assert leakage["false_positive_bladder_fraction"] == 0.25
