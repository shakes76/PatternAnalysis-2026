"""The held-out test split must not be reopened after evaluation starts."""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_test  # noqa: E402
from dataset import Case, manifest_sha256, save_split  # noqa: E402
from evaluate_test import claim_test_evaluation, load_pending_report  # noqa: E402
from modules import ImprovedUNet3D, UNet2D  # noqa: E402


def pending_run(run_dir: Path) -> dict:
    report = {
        "provenance": {"validation_only": True},
        "2d": {"validation": {}},
        "3d": {"validation": {}},
    }
    (run_dir / "results.json").write_text(json.dumps(report))
    return report


def test_claim_prevents_a_second_test_evaluation(tmp_path: Path) -> None:
    report = pending_run(tmp_path)
    assert load_pending_report(tmp_path) == report

    claim_test_evaluation(tmp_path, tmp_path / "manifest.csv")
    with pytest.raises(ValueError, match="already started"):
        load_pending_report(tmp_path)
    with pytest.raises(ValueError, match="already started"):
        claim_test_evaluation(tmp_path, tmp_path / "manifest.csv")


@pytest.mark.parametrize("leftover", [
    "per_case.csv", "per_case.pending.csv", "results.pending.json",
])
def test_interrupted_test_evaluation_requires_inspection(
    tmp_path: Path, leftover: str
) -> None:
    pending_run(tmp_path)
    (tmp_path / leftover).touch()
    with pytest.raises(ValueError, match="already started"):
        load_pending_report(tmp_path)


def test_already_completed_test_report_is_rejected(tmp_path: Path) -> None:
    report = pending_run(tmp_path)
    report["2d"]["test"] = {}
    (tmp_path / "results.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="already exist"):
        load_pending_report(tmp_path)


@pytest.mark.parametrize("crop_xy", [None, 8])
def test_finalizer_writes_native_results_then_blocks_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crop_xy: int | None
) -> None:
    torch.set_num_threads(2)
    cases = []
    for number in range(3):
        image = np.zeros((16, 16, 16), dtype=np.float32)
        image[2:14, 2:14, 2:14] = 1 + number
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
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_split({"train": cases[:1], "val": cases[1:2], "test": cases[2:]},
               run_dir / "split.json")
    report = pending_run(run_dir)
    report["provenance"].update({
        "manifest": str(manifest.resolve()),
        "manifest_sha256": manifest_sha256(manifest),
        "size_dhw": [16, 16, 16],
        "prostate_label": 5,
        "crop_xy": crop_xy,
    })
    (run_dir / "results.json").write_text(json.dumps(report))
    for kind, model in (("2d", UNet2D(width=4)),
                        ("3d", ImprovedUNet3D(width=4))):
        torch.save({
            "kind": kind,
            "width": 4,
            "state_dict": model.state_dict(),
            "size": [16, 16, 16],
            "prostate_label": 5,
            "crop_xy": crop_xy,
        }, run_dir / f"{kind}_best.pt")

    other_manifest = tmp_path / "other_manifest.csv"
    other_manifest.write_text(manifest.read_text())
    monkeypatch.setattr(sys, "argv", [
        "evaluate_test.py", "--manifest", str(other_manifest),
        "--run-dir", str(run_dir), "--device", "cpu",
    ])
    with pytest.raises(ValueError, match="Manifest path differs"):
        evaluate_test.main()
    assert not (run_dir / "test_evaluation_started.json").exists()

    original_manifest = manifest.read_text()
    manifest.write_text(original_manifest.replace("patient_2", "other_patient"))
    monkeypatch.setattr(sys, "argv", [
        "evaluate_test.py", "--manifest", str(manifest),
        "--run-dir", str(run_dir), "--device", "cpu",
    ])
    with pytest.raises(ValueError, match="Manifest contents differ"):
        evaluate_test.main()
    assert not (run_dir / "test_evaluation_started.json").exists()
    manifest.write_text(original_manifest)

    evaluate_test.main()
    completed = json.loads((run_dir / "results.json").read_text())
    assert completed["provenance"]["validation_only"] is False
    assert (run_dir / "test_evaluation_started.json").is_file()
    assert (run_dir / "per_case.csv").is_file()
    assert all(
        "patient_mean_native_prostate_dice" in completed[kind]["test"]
        for kind in ("2d", "3d")
    )
    with pytest.raises(ValueError, match="validation-only"):
        evaluate_test.main()
