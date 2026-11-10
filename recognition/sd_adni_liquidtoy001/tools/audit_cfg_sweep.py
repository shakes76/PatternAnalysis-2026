"""Verify the completed CFG sweep and summarize paired descriptive changes."""

import json
from pathlib import Path

import numpy as np
from audit_appearance import measure, summarize, verify
from PIL import Image


def main():
    root = Path("outputs/cfg-636273")
    source = json.loads((root / "report.json").read_text())
    historical_root = Path("outputs/cohort-632557/adapted")
    historical = json.loads((historical_root / "report.json").read_text())
    for report in (source, historical):
        # This field is a set of defaulted keys serialized in process-dependent order.
        config = report["scheduler_config"]
        config["_use_default_values"] = sorted(config["_use_default_values"])
    for key in (
        "checkpoint_sha256",
        "scheduler_config",
        "prompts",
        "inference_steps",
        "eta",
        "aligned_scheduler_precision",
        "torch",
    ):
        if source[key] != historical[key]:
            raise ValueError(f"Historical contract changed: {key}")
    records = source["records"]
    expected = {
        (c, label, seed)
        for c in (3.0, 5.0, 7.5)
        for label in ("AD", "NC")
        for seed in range(37200, 37232)
    }
    indexed = {(r["guidance"], r["label"], r["seed"]): r for r in records}
    if set(indexed) != expected or len(records) != len(expected):
        raise ValueError("Incomplete or duplicate factorial records")
    for row in records:
        verify(root / row["file"], row["png_sha256"])
        with Image.open(root / row["file"]) as image:
            if image.size != (256, 256):
                raise ValueError("Unexpected image resolution")
            row.update(measure(image))
    equal = 0
    for row in historical["records"]:
        verify(historical_root / row["file"], row["png_sha256"])
        current = indexed[(7.5, row["label"], row["seed"])]
        equal += current["png_sha256"] == row["png_sha256"]
    for label in ("AD", "NC"):
        for seed in range(37200, 37232):
            if (
                len(
                    {
                        indexed[(c, label, seed)]["initial_noise_sha256"]
                        for c in (3.0, 5.0, 7.5)
                    }
                )
                != 1
            ):
                raise ValueError("Initial noise mismatch")
    pairs = {}
    for low, high in ((3.0, 5.0), (5.0, 7.5), (3.0, 7.5)):
        metrics = {}
        for metric in ("color", "border", "edge"):
            deltas = np.array(
                [
                    [
                        indexed[(high, label, seed)][metric]
                        - indexed[(low, label, seed)][metric]
                        for label in ("AD", "NC")
                    ]
                    for seed in range(37200, 37232)
                ]
            )
            blocks = deltas.mean(axis=1)
            metrics[metric] = {
                "mean_delta": float(blocks.mean()),
                "median_seed_block_delta": float(np.median(blocks)),
                "sample_decreases_of64": int((deltas < 0).sum()),
                "seed_block_decreases_of32": int((blocks < 0).sum()),
                "seed_block_increases_of32": int((blocks > 0).sum()),
            }
        pairs[f"{high}-minus-{low}"] = metrics
    result = {
        "job": "636273",
        "elapsed": "00:09:06",
        "exit_code": "0:0",
        "verified_images": len(records),
        "historical_equal_of64": equal,
        "matched_noise_triplets": 64,
        "groups": {
            str(c): summarize([r for r in records if r["guidance"] == c])
            for c in (3.0, 5.0, 7.5)
        },
        "paired": pairs,
        "cases": [
            {k: row[k] for k in ("file", "guidance", "color", "border", "edge")}
            for row in records
            if (row["label"], row["seed"]) in (("AD", 37220), ("NC", 37217))
        ],
        "limits": "32 seed blocks, shared AD/NC noise; descriptive only; test unused",
    }
    (root / "paired-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"historical_equal": equal, "paired": pairs, "cases": result["cases"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
