"""The reported cross-run comparison must use exactly the same test cases."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_models import pair_rows  # noqa: E402


def write_rows(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "patient_id", "case_id", "native_prostate_dice"))
        for model, patient, case in rows:
            writer.writerow((model, patient, case, "0.7"))


def test_comparison_rejects_nonidentical_cases_and_duplicate_results(tmp_path: Path) -> None:
    selected = tmp_path / "selected.csv"
    baseline = tmp_path / "baseline.csv"
    write_rows(selected, [("2d", "p1", "c1"), ("3d", "p1", "c1")])
    write_rows(baseline, [("2d", "p2", "c2")])
    with pytest.raises(ValueError, match="Patient/case sets differ"):
        pair_rows(selected, baseline)

    write_rows(baseline, [("2d", "p1", "c1"), ("2d", "p1", "c1")])
    with pytest.raises(ValueError, match="Duplicate result"):
        pair_rows(selected, baseline)

    write_rows(baseline, [("2d", "p1", "c1")])
    fields, paired = pair_rows(selected, baseline)
    assert fields[0] == "model"
    assert [row["model"] for row in paired] == ["2d", "3d"]
