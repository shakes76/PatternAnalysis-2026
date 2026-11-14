"""Audit the ISIC2018 cohort, split it by patient and export it for YOLO11.

Builds the cohort, audits its image/mask pairs, splits it by patient and exports
it for YOLO11 using dataset.py.

Each step saves its output before the next step runs so a failure always leaves a
diagnostic behind. The script stops when the audit, grouping or split validation fails.
The output directory must be new or empty to allow comparings reruns.

Usage:
    uv run python -m yolo_lesion_47041585.audit [--source DIR] [--output DIR]
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from yolo_lesion_47041585.dataset import (
    SPLITS,
    SplitManifest,
    audit_pairs,
    build_cohort,
    build_independence_groups,
    export_yolo_dataset,
    make_grouped_splits,
    validate_split_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_SEED = 47041585
MASK_THRESHOLD = 128

# original source sets and where their images and masks live under the dataset folder
IMAGE_DIRS = {
    "train": "ISIC2018_Task1-2_Training_Input_x2",
    "test": "ISIC2018_Task1-2_Test_Input",
}
MASK_DIRS = {
    "train": "ISIC2018_Task1_Training_GroundTruth_x2",
    "test": "ISIC2018_Task1_Test_GroundTruth",
}


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    """Write a CSV file with a header row.

    Args:
        path (Path): File to write.
        header (Sequence[str]): Column names.
        rows (Iterable[Sequence[Any]]): The rows, in column order.
    """
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def summarise_splits(manifest: SplitManifest) -> dict[str, dict[str, Any]]:
    """Summarise the image, patient and class counts of each split.

    Args:
        manifest (SplitManifest): The split assignment to summarise.

    Returns:
        dict[str, dict[str, Any]]: For each split, its numbers of images, patients,
            lesions, groups, benign and malignant images, and images per source set.
    """
    summary: dict[str, dict[str, Any]] = {}

    for split in SPLITS:
        entries = [e for e in manifest.records if e.split == split]
        records = [e.audited.record for e in entries]
        summary[split] = {
            "images": len(entries),
            "patients": len({r.patient_id for r in records}),
            "lesions": len({r.lesion_id for r in records if r.lesion_id}),
            "groups": len({e.group_id for e in entries}),
            "benign": sum(r.class_id == 0 for r in records),
            "malignant": sum(r.class_id == 1 for r in records),
            "source_sets": dict(Counter(r.source_set for r in records)),
        }

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit, split and export pipeline.

    Args:
        argv (Sequence[str] | None): Command-line arguments, or None to use
            sys.argv.

    Returns:
        int: 0 on success, 1 if a step stopped the run, or 2 if the output
            directory is not empty.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts/isic2018"
    )
    args = parser.parse_args(argv)
    source: Path = args.source
    output: Path = args.output

    if output.exists() and any(output.iterdir()):
        print(f"Output directory is not empty: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)

    # summary.json is rewritten after every step, so it records how far the run got
    summary: dict[str, Any] = {
        "split_seed": SPLIT_SEED,
        "mask_threshold": MASK_THRESHOLD,
        "status": "incomplete",
        "steps_completed": [],
    }

    def save(step: str | None = None, **info: Any) -> None:
        if step:
            summary["steps_completed"].append(step)
        summary.update(info)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n"
        )

    def stop(reason: str) -> int:
        save(status="stopped", reason=reason)
        print(f"Stopped: {reason}", file=sys.stderr)
        return 1

    # a spinner shows the current stage; each finished stage is printed with a tick
    console = Console()
    status = console.status("")
    current = ""

    def stage(label: str | None = None) -> None:
        nonlocal current
        if current:
            console.print(f"[green]✓[/green] {current}")
        current = label or ""
        if label:
            status.update(label)

    status.start()
    stage("Building cohort...")

    try:
        # 1. cohort: filter the raw metadata and pair images with masks
        cohort = build_cohort(
            source / "metadata_raw.csv",
            {name: source / folder for name, folder in IMAGE_DIRS.items()},
            {name: source / folder for name, folder in MASK_DIRS.items()},
        )
        write_csv(
            output / "cohort.csv",
            (
                "isic_id",
                "source_set",
                "patient_id",
                "lesion_id",
                "diagnosis",
                "class_id",
                "image_path",
                "mask_path",
            ),
            (
                (
                    r.isic_id,
                    r.source_set,
                    r.patient_id,
                    r.lesion_id or "",
                    r.diagnosis,
                    r.class_id,
                    r.image_path,
                    r.mask_path,
                )
                for r in cohort.records
            ),
        )
        write_csv(
            output / "exclusions.csv",
            ("metadata_row", "isic_id", "reasons"),
            (
                (e.metadata_row, e.isic_id or "", ";".join(e.reasons))
                for e in cohort.exclusions
            ),
        )
        save(
            "cohort",
            raw_metadata_sha256=cohort.raw_metadata_sha256,
            raw_rows=cohort.input_row_count,
            retained=len(cohort.records),
            retained_by_diagnosis=dict(Counter(r.diagnosis for r in cohort.records)),
            retained_by_source=dict(Counter(r.source_set for r in cohort.records)),
            excluded=len(cohort.exclusions),
            # a row can have several reasons, so these counts overlap
            exclusion_reasons=dict(
                Counter(reason for e in cohort.exclusions for reason in e.reasons)
            ),
        )

        # 2. pair audit: decode every pair and derive its box
        stage("Auditing image/mask pairs...")
        audited = audit_pairs(cohort.records, MASK_THRESHOLD)
        save("audit", audited=len(audited))

        # 3. groups: records that share a patient must stay together
        stage("Grouping by patient...")
        groups = build_independence_groups(audited)
        save(
            "groups",
            patients=len({a.record.patient_id for a in audited}),
            groups=len(set(groups.values())),
        )

        # 4. split, then check the split independently
        stage("Splitting...")
        manifest = make_grouped_splits(audited, groups, SPLIT_SEED)
        write_csv(
            output / "split_manifest.csv",
            (
                "isic_id",
                "split",
                "group_id",
                "source_set",
                "patient_id",
                "lesion_id",
                "class_id",
                "image_width",
                "image_height",
                "x_center",
                "y_center",
                "width",
                "height",
                "image_sha256",
                "mask_sha256",
            ),
            (
                (
                    e.audited.record.isic_id,
                    e.split,
                    e.group_id,
                    e.audited.record.source_set,
                    e.audited.record.patient_id,
                    e.audited.record.lesion_id or "",
                    e.audited.record.class_id,
                    e.audited.image_width,
                    e.audited.image_height,
                    f"{e.audited.box.x_center:.6f}",
                    f"{e.audited.box.y_center:.6f}",
                    f"{e.audited.box.width:.6f}",
                    f"{e.audited.box.height:.6f}",
                    e.audited.image_sha256,
                    e.audited.mask_sha256,
                )
                for e in manifest.records
            ),
        )
        violations = validate_split_manifest(manifest, audited)
        save(
            "split",
            split_manifest_sha256=hashlib.sha256(
                (output / "split_manifest.csv").read_bytes()
            ).hexdigest(),
            splits=summarise_splits(manifest),
            split_violations=list(violations),
        )
        if violations:
            status.stop()
            return stop(f"{len(violations)} split violation(s), see summary.json")

        # 5. export the verified split for YOLO
        stage("Exporting YOLO dataset...")
        yaml_path = export_yolo_dataset(manifest, output / "yolo")
        save("export", status="complete", dataset_yaml=yaml_path)
        stage()

    except ValueError as exc:
        # expected failures, such as bad roots, missing columns or a lesion under two patients
        return stop(str(exc))
    finally:
        status.stop()

    table = Table(title=f"Cohort: {summary['retained']} images retained")
    table.add_column("Split", style="bold")
    for column in ("Images", "Patients", "Groups", "Benign", "Malignant"):
        table.add_column(column, justify="right")
    for split, info in summary["splits"].items():
        table.add_row(
            split,
            *(
                str(info[key])
                for key in ("images", "patients", "groups", "benign", "malignant")
            ),
        )
    console.print()
    console.print(table)
    console.print(f"Done. Artifacts written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
