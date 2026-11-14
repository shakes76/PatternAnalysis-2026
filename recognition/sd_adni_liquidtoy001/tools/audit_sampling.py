"""Retrospective appearance comparison of the existing fixed-seed factorial probe."""

import json
from pathlib import Path

from audit_appearance import measure, summarize, verify
from PIL import Image


def main():
    root = Path("outputs/inference-632530")
    source = json.loads((root / "report.json").read_text())
    records = []
    for record in source["records"]:
        path = root / record["file"]
        verify(path, record["png_sha256"])
        with Image.open(path) as image:
            records.append({**record, **measure(image)})
    keys = {(r["size"], r["guidance"], r["label"], r["seed"]) for r in records}
    expected = {
        (size, cfg, label, seed)
        for size in (256, 512)
        for cfg in (1.0, 7.5)
        for label in ("AD", "NC")
        for seed in (37100, 37101)
    }
    if keys != expected or len(records) != len(expected):
        raise ValueError("Expected the complete sixteen-image factorial probe")
    pairs = []
    for low in [r for r in records if r["guidance"] == 1.0]:
        high = next(
            r
            for r in records
            if r["size"] == low["size"]
            and r["label"] == low["label"]
            and r["seed"] == low["seed"]
            and r["guidance"] == 7.5
        )
        if low["initial_noise_sha256"] != high["initial_noise_sha256"]:
            raise ValueError("CFG comparison does not share initial noise")
        pairs.append(
            {
                "size": low["size"],
                "label": low["label"],
                "seed": low["seed"],
                "high_minus_low": {
                    m: high[m] - low[m] for m in ("color", "border", "edge")
                },
            }
        )
    report = {
        "source_job": "632530",
        "retrospective": True,
        "precision": "original factorial run, not aligned-scheduler cohort",
        "limitations": [
            "two independent seeds, labels share seed noise",
            "different resolutions have different-shaped noise",
            "descriptive proxies, not anatomy or quality scores",
        ],
        "groups": {
            f"{size}-cfg{cfg}": summarize(
                [r for r in records if r["size"] == size and r["guidance"] == cfg]
            )
            for size in (256, 512)
            for cfg in (1.0, 7.5)
        },
        "cfg_pairs": pairs,
    }
    output = Path("outputs/sampling-appearance-20261007.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key, group in report["groups"].items():
        print(key, {m: round(v["mean"], 6) for m, v in group["all"]["metrics"].items()})
    print("CFG paired changes:", json.dumps(pairs))


if __name__ == "__main__":
    main()
