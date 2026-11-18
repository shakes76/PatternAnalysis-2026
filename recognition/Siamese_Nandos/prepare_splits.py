"""Create audited splits using the supplied development notebook's algorithm."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import sklearn

from split_utils import define_splits, validate_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--duplicates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if any((output / name).exists() for name in ("train.csv", "val.csv", "test.csv", "split_definition.json")):
        raise FileExistsError("Output already contains split files; choose a new directory to protect frozen manifests")
    metadata = pd.read_csv(args.metadata, dtype={"image_name": str, "patient_id": str, "lesion_id": str})
    frames, definition = define_splits(metadata)
    frames = validate_splits(frames, pd.read_csv(args.duplicates, dtype=str))
    output.mkdir(parents=True, exist_ok=True)
    definition.update({"algorithm": "two-stage GroupShuffleSplit; notebook cell 15",
                       "scikit_learn": sklearn.__version__,
                       "metadata_sha256": hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest(),
                       "duplicates_sha256": hashlib.sha256(Path(args.duplicates).read_bytes()).hexdigest(),
                       "splits": {}})
    for name, frame in frames.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        definition["splits"][name] = {"images": len(frame), "patients": frame.patient_id.nunique(),
                                     "melanomas": int(frame.target.sum()),
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / "split_definition.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")
    print(json.dumps(definition, indent=2))


if __name__ == "__main__":
    main()
