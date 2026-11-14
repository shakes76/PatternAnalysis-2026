"""Join cohort chunks (consecutive seed ranges of one model) into one folder.

Every PNG is checked against its chunk's recorded SHA-256 before it is copied;
file names and seeds must not repeat. The first chunk's grids are kept, so the
uncurated grid still shows the first 32 seeds per class.

    python tools/merge_cohorts.py --output outputs/final-e2 \
        outputs/final-e2-s37200-<job> outputs/final-e2-s37328-<job> ...
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_common import cohort_records  # noqa: E402


def merge(chunks: list[Path], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    records, seen, sources = [], set(), []
    for chunk in chunks:
        report = json.loads((chunk / "report.json").read_text())
        for record in cohort_records(chunk):
            key = (record["label"], record["seed"])
            if record["file"] in seen or key in seen:
                raise ValueError(f"Repeated sample {record['file']}")
            seen.update({record["file"], key})
            shutil.copy2(chunk / record["file"], output / record["file"])
            records.append(record)
        sources.append(
            {
                "chunk": str(chunk),
                "images": len(report["records"]),
                **{k: v for k, v in report.items() if k not in ("records", "summary")},
            }
        )
    for grid in chunks[0].glob("grid*.png"):
        shutil.copy2(grid, output / grid.name)
    merged = {
        "merged_from": sources,
        "records": sorted(records, key=lambda r: (r["label"], r["seed"])),
        "scope": f"{len(records)} predeclared images from {len(chunks)} chunks",
    }
    (output / "report.json").write_text(json.dumps(merged, indent=2) + "\n")
    cohort_records(output)  # every copied file still matches its hash
    return merged


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("chunks", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merged = merge(args.chunks, args.output)
    print(json.dumps({"images": len(merged["records"]), "chunks": len(args.chunks)}))


if __name__ == "__main__":
    main()
