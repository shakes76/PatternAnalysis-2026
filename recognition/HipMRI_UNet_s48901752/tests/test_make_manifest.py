"""make_manifest.py on synthetic folders of empty placeholder files (no real data needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

import dataset
import make_manifest

pytestmark = pytest.mark.synthetic

# Anonymised HipMRI patient IDs (38). The split hash depends only on these IDs and the
# seed, so placeholder files are enough to check the frozen split end to end.
HIPMRI_PATIENTS = [
    "B006",
    "B037",
    "B038",
    "B040",
    "C032",
    "D031",
    "G021",
    "H007",
    "H017",
    "J005",
    "J026",
    "K008",
    "K018",
    "K019",
    "K042",
    "L011",
    "M004",
    "M013",
    "M015",
    "M020",
    "M023",
    "M030",
    "M036",
    "N010",
    "O025",
    "R016",
    "R024",
    "R039",
    "S022",
    "S028",
    "S033",
    "S035",
    "T009",
    "T014",
    "V027",
    "W012",
    "W029",
    "W041",
]


def make_data_root(root: Path, patients: list[str], weeks: int = 2) -> Path:
    (root / make_manifest.IMAGE_DIR).mkdir(parents=True)
    (root / make_manifest.MASK_DIR).mkdir(parents=True)
    for patient in patients:
        for week in range(weeks):
            scan = f"{patient}_Week{week}"
            (
                root / make_manifest.IMAGE_DIR / f"{scan}{make_manifest.IMAGE_SUFFIX}"
            ).touch()
            (
                root / make_manifest.MASK_DIR / f"{scan}{make_manifest.MASK_SUFFIX}"
            ).touch()
    return root


def test_patient_level_split_has_no_leakage(tmp_path):
    patients = [f"P{i:03d}" for i in range(38)]
    frame = make_manifest.build_manifest(make_data_root(tmp_path, patients, weeks=3))

    assert list(frame.columns) == list(dataset.REQUIRED_COLUMNS)
    assert len(frame) == 38 * 3 and frame["scan_id"].is_unique
    assert (frame.groupby("patient_id")["split"].nunique() == 1).all()
    counts = frame.groupby("split")["patient_id"].nunique().to_dict()
    assert counts == dataset.EXPECTED_PATIENTS_PER_SPLIT
    for row in frame.itertuples():
        assert Path(row.image_path).is_file() and Path(row.mask_path).is_file()
        assert row.scan_id.startswith(f"{row.patient_id}_Week")


def test_split_is_deterministic_and_seed_dependent(tmp_path):
    root = make_data_root(tmp_path, [f"P{i:03d}" for i in range(38)])
    first = make_manifest.build_manifest(root, seed=3710)
    again = make_manifest.build_manifest(root, seed=3710)
    other = make_manifest.build_manifest(root, seed=1)
    assert first.equals(again)
    assert dataset.split_sha256(first) != dataset.split_sha256(other)


def test_cli_reproduces_frozen_split_hash(tmp_path, capsys):
    root = make_data_root(tmp_path / "data", HIPMRI_PATIENTS)
    out = tmp_path / "metadata.csv"
    make_manifest.main(["--data-root", str(root), "--out", str(out)])

    frame = dataset.load_manifest(out)  # raises on leakage, wrong counts or wrong hash
    assert dataset.split_sha256(frame) == dataset.EXPECTED_SPLIT_SHA256
    assert dataset.EXPECTED_SPLIT_SHA256 in capsys.readouterr().out


def test_missing_mask_is_rejected(tmp_path):
    root = make_data_root(tmp_path, ["A001", "A002", "A003"])
    (root / make_manifest.MASK_DIR / f"A002_Week1{make_manifest.MASK_SUFFIX}").unlink()
    with pytest.raises(FileNotFoundError, match="A002_Week1"):
        make_manifest.discover_scans(root)
