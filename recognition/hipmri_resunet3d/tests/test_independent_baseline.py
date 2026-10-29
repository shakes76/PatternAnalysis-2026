"""A stronger 2D comparator must use the locked patient split and run once."""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_2d_baseline  # noqa: E402
from dataset import Case, manifest_sha256, save_split  # noqa: E402
from modules import UNet2D  # noqa: E402


def test_selected_2d_evaluation_requires_matching_split_and_blocks_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(2)
    cases = []
    for number in range(3):
        image = np.zeros((16, 16, 16), dtype=np.float32)
        image[2:14, 2:14, 2:14] = number + 1
        mask = np.zeros_like(image)
        mask[6:10, 6:10, 6:10] = 5
        image_path = tmp_path / f"image_{number}.nii.gz"
        mask_path = tmp_path / f"mask_{number}.nii.gz"
        nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
        nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
        cases.append(Case(f"case_{number}", f"patient_{number}", image_path, mask_path))

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,patient_id,image,mask\n" + "".join(
            f"{case.case_id},{case.patient_id},{case.image.name},{case.mask.name}\n"
            for case in cases
        )
    )
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    split = {"train": cases[:1], "val": cases[1:2], "test": cases[2:]}
    save_split(split, candidate / "split.json")
    save_split({"train": cases[:1], "val": cases[2:], "test": cases[1:2]},
               reference / "split.json")
    report = {
        "2d": {"validation": {"patient_mean_prostate_dice": 0.7}},
        "3d": {"validation": {}},
        "provenance": {
            "validation_only": True,
            "manifest": str(manifest.resolve()),
            "manifest_sha256": manifest_sha256(manifest),
            "size_dhw": [16, 16, 16],
            "prostate_label": 5,
            "crop_xy": 8,
            "bladder_label": 3,
            "rectum_label": 4,
        },
    }
    (candidate / "results.json").write_text(json.dumps(report))
    reference_report = {
        "3d": {"test": {}},
        "provenance": {
            "validation_only": False,
            "manifest": str(manifest.resolve()),
            "manifest_sha256": manifest_sha256(manifest),
            "prostate_label": 5,
            "bladder_label": 3,
            "rectum_label": 4,
        },
    }
    (reference / "results.json").write_text(json.dumps(reference_report))
    torch.save({
        "kind": "2d", "width": 4, "state_dict": UNet2D(width=4).state_dict(),
        "size": [16, 16, 16], "prostate_label": 5, "crop_xy": 8,
    }, candidate / "2d_best.pt")

    monkeypatch.setattr(sys, "argv", [
        "evaluate_2d_baseline.py", "--manifest", str(manifest),
        "--run-dir", str(candidate), "--reference-run", str(reference),
        "--device", "cpu",
    ])
    with pytest.raises(ValueError, match="different patient splits"):
        evaluate_2d_baseline.main()
    assert not (candidate / "test_evaluation_started.json").exists()

    save_split(split, reference / "split.json")
    reference_report["provenance"]["manifest"] = str(tmp_path / "other_manifest.csv")
    (reference / "results.json").write_text(json.dumps(reference_report))
    with pytest.raises(ValueError, match="Manifest path differs"):
        evaluate_2d_baseline.main()
    assert not (candidate / "test_evaluation_started.json").exists()

    reference_report["provenance"]["manifest"] = str(manifest.resolve())
    reference_report["provenance"]["prostate_label"] = 7
    (reference / "results.json").write_text(json.dumps(reference_report))
    with pytest.raises(ValueError, match="different anatomical labels"):
        evaluate_2d_baseline.main()
    assert not (candidate / "test_evaluation_started.json").exists()

    reference_report["provenance"]["prostate_label"] = 5
    (reference / "results.json").write_text(json.dumps(reference_report))
    evaluate_2d_baseline.main()
    result = json.loads((candidate / "2d_baseline_test.json").read_text())
    assert result["provenance"]["same_patient_split"] is True
    assert "patient_mean_native_prostate_dice" in result["test"]
    assert (candidate / "2d_baseline_per_case.csv").is_file()
    assert json.loads((candidate / "results.json").read_text()) == report
    with pytest.raises(ValueError, match="already started"):
        evaluate_2d_baseline.main()
