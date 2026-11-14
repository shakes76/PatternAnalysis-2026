"""Apply the E2 scan-exclusion rule to the training and validation splits.

Uses dataset.defective_scans on 256 px loader output, the resolution train.py
sees. Writes aggregate counts (safe to commit), a private per-scan record, and
private panels of every slice of each flagged scan next to a typical scan of
the same class, for visual confirmation. The test split is not read.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import (  # noqa: E402
    NEAR_EMPTY_BELOW,
    ADNISliceDataset,
    defective_scans,
    foreground_share,
)

TILE = 96


def scan_panel(data, indices, path):
    """All slices of one scan in a row, ordered by slice index."""
    indices = sorted(indices, key=lambda i: int(data.rows[i]["slice_index"]))
    panel = Image.new("L", (len(indices) * TILE, TILE), 0)
    for column, index in enumerate(indices):
        pixels = (data[index]["pixel_values"][0] + 1) / 2
        tile = Image.frombytes(
            "L",
            pixels.shape[::-1],
            bytes(pixels.mul(255).round().byte().flatten().tolist()),
        ).resize((TILE, TILE))
        panel.paste(tile, (column * TILE, 0))
    panel.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    aggregate, private = {"near_empty_below": NEAR_EMPTY_BELOW, "splits": {}}, {}
    for split in ("train", "validation"):
        data = ADNISliceDataset(
            Path("/home/groups/comp3710/ADNI"),
            Path("data/manifest-seed3710.csv"),
            split,
            256,
            verify_hash=True,
        )
        shares = []
        for batch in DataLoader(data, batch_size=64, num_workers=2):
            shares.extend(foreground_share(batch["pixel_values"]).tolist())
        flagged = defective_scans(data.rows, shares)
        by_scan = {}
        for i, (row, share) in enumerate(zip(data.rows, shares)):
            if row["scan_id"] in flagged:
                by_scan.setdefault(row["scan_id"], []).append((i, share))
        aggregate["splits"][split] = {
            "slices": len(data.rows),
            "near_empty_slices": sum(s < NEAR_EMPTY_BELOW for s in shares),
            "flagged_scans": len(flagged),
            "flagged_scans_by_class": {
                label: sum(
                    1
                    for scan in flagged
                    if data.rows[by_scan[scan][0][0]]["class_name"] == label
                )
                for label in ("AD", "NC")
            },
            "flagged_subjects": len(
                {data.rows[by_scan[s][0][0]]["patient_id"] for s in flagged}
            ),
            "slices_removed": sum(len(v) for v in by_scan.values()),
            "near_empty_slices_per_flagged_scan": sorted(
                sum(share < NEAR_EMPTY_BELOW for _, share in v)
                for v in by_scan.values()
            ),
        }
        private[split] = {}
        for scan, items in by_scan.items():
            row = data.rows[items[0][0]]
            private[split][scan] = {
                "patient_id": row["patient_id"],
                "class_name": row["class_name"],
                "shares": [round(s, 4) for _, s in items],
            }
            scan_panel(
                data, [i for i, _ in items], args.output / f"private-{split}-{scan}.png"
            )
            typical = next(
                r["scan_id"]
                for r in data.rows
                if r["class_name"] == row["class_name"] and r["scan_id"] not in flagged
            )
            scan_panel(
                data,
                [i for i, r in enumerate(data.rows) if r["scan_id"] == typical],
                args.output / f"private-{split}-{scan}-typical.png",
            )
    (args.output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    (args.output / "private-scans.json").write_text(
        json.dumps(private, indent=2) + "\n"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
