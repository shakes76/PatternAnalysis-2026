"""Foreground share of real training and validation slices, by class and slice.

Interprets near-empty generated images: if real data holds near-empty edge
slices at a similar rate, they are faithful; if not, they are a failure. Uses
the same threshold as the cohort check (pixel > 0.1 at 128 px, near-empty below
5% foreground). No test-split rows are read.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import ADNISliceDataset  # noqa: E402

THRESHOLD, EMPTY = 0.1, 0.05


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"threshold": THRESHOLD, "near_empty_below": EMPTY, "splits": {}}
    for split in ("train", "validation"):
        data = ADNISliceDataset(
            Path("/home/groups/comp3710/ADNI"),
            Path("data/manifest-seed3710.csv"),
            split,
            128,
            verify_hash=True,
        )
        by_class = defaultdict(list)
        by_slice = defaultdict(lambda: defaultdict(list))
        for i in range(len(data)):
            item = data[i]
            pixels = (item["pixel_values"][0] + 1) / 2
            share = (pixels > THRESHOLD).float().mean().item()
            by_class[item["class_name"]].append(share)
            by_slice[item["class_name"]][item["slice_index"]].append(share)
        report["splits"][split] = {
            label: {
                "n": len(values),
                "mean_foreground": sum(values) / len(values),
                "near_empty_fraction": sum(v < EMPTY for v in values) / len(values),
                "near_empty_by_slice": {
                    str(s): sum(v < EMPTY for v in vs) / len(vs)
                    for s, vs in sorted(by_slice[label].items())
                },
            }
            for label, values in sorted(by_class.items())
        }
    (args.output / "aggregate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
