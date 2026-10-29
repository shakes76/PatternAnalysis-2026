"""Check that repeated visits do not dominate paired model comparisons."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_results import paired_patient_summary  # noqa: E402


def test_paired_patient_summary_weights_patients_equally() -> None:
    rows = []
    for patient_id, case_id, baseline, improved in (
        ("p1", "a", 0.2, 0.4),
        ("p1", "b", 0.4, 0.6),
        ("p2", "c", 0.8, 0.7),
    ):
        for kind, value in (("2d", baseline), ("3d", improved)):
            rows.append({
                "model": kind, "patient_id": patient_id, "case_id": case_id,
                "native_prostate_dice": str(value),
            })
    summary = paired_patient_summary(rows, "native_prostate_dice", bootstrap_samples=100)
    assert summary["patients"] == 2
    assert summary["baseline_patient_mean"] == pytest.approx(0.55)
    assert summary["improved_patient_mean"] == pytest.approx(0.60)
    assert summary["improved_minus_baseline"] == pytest.approx(0.05)


def test_paired_patient_summary_rejects_unpaired_cases() -> None:
    with pytest.raises(ValueError, match="same patient/case pairs"):
        paired_patient_summary([
            {"model": "2d", "patient_id": "p1", "case_id": "a", "native_prostate_dice": "0.5"}
        ], "native_prostate_dice")
