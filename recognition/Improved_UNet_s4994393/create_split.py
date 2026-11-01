"""Regenerate the seed-42 patient split from the supplied HipMRI filenames."""

import argparse
import json
from pathlib import Path

from dataset import discover_pairs, split_three_way


def create_split(data_root, output):
    """Write a reference split without replacing a different existing assignment."""
    split = split_three_way(discover_pairs(Path(data_root).resolve()), 0.15, 0.15, 42)
    serializable = {group: [list(pair) for pair in pairs] for group, pairs in split.items()}
    output = Path(output)
    if output.exists():
        if json.loads(output.read_text()) != serializable:
            raise ValueError(
                "Existing reference differs; use a new output path or investigate the dataset"
            )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(serializable, indent=2) + "\n")
    return serializable


def main():
    """Save and summarize the reference patient assignments used by preprocessing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/reference_split.json"))
    args = parser.parse_args()
    split = create_split(args.data_root, args.output)
    for group, pairs in split.items():
        patients = {Path(pair[0]).name.split("_")[0] for pair in pairs}
        print(f"{group}: {len(patients)} patients / {len(pairs)} scans")
    print(f"Reference split: {args.output}")


if __name__ == "__main__":
    main()
