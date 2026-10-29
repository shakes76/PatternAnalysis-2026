"""Keep exploratory training from opening the test split by accident."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import parse_arguments  # noqa: E402


@pytest.mark.parametrize(
    ("mode_flags", "expected_validation_only"),
    [([], True), (["--validation-only"], True), (["--evaluate-test"], False)],
)
def test_test_evaluation_requires_an_explicit_flag(
    monkeypatch: pytest.MonkeyPatch,
    mode_flags: list[str],
    expected_validation_only: bool,
) -> None:
    monkeypatch.setattr(
        sys, "argv",
        ["train.py", "--manifest", "manifest.csv", "--prostate-label", "5", *mode_flags],
    )
    assert parse_arguments().validation_only is expected_validation_only


def test_training_modes_are_mutually_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "argv",
        ["train.py", "--manifest", "manifest.csv", "--prostate-label", "5",
         "--validation-only", "--evaluate-test"],
    )
    with pytest.raises(SystemExit, match="2"):
        parse_arguments()
