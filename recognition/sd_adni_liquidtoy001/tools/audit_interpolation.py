"""Verify interpolation completeness, source integrity, and endpoint parity."""

import json
from pathlib import Path

from audit_appearance import verify


def main():
    root = Path("outputs/interpolation-639671")
    source = json.loads((root / "report.json").read_text())
    old_root = Path("outputs/cohort-632557/adapted")
    old = json.loads((old_root / "report.json").read_text())
    for key in (
        "checkpoint_sha256",
        "prompts",
        "inference_steps",
        "eta",
        "aligned_scheduler_precision",
        "torch",
    ):
        if source[key] != old[key]:
            raise ValueError(f"Contract mismatch: {key}")
    for report in (source, old):
        config = report["scheduler_config"]
        config["_use_default_values"] = sorted(config["_use_default_values"])
    if source["scheduler_config"] != old["scheduler_config"]:
        raise ValueError("Scheduler configuration mismatch")
    expected = {
        (label, a, b, t)
        for label in ("AD", "NC")
        for a, b in ((37200, 37201), (37216, 37217))
        for t in (0.0, 0.25, 0.5, 0.75, 1.0)
    }
    rows = source["records"]
    if (
        len(rows) != 20
        or {(r["label"], r["seed"], r["end_seed"], r["alpha"]) for r in rows}
        != expected
    ):
        raise ValueError("Incomplete interpolation coverage")
    reference = {(r["label"], r["seed"]): r for r in old["records"]}
    endpoints = 0
    for row in rows:
        verify(root / row["file"], row["png_sha256"])
        if row["size"] != 256 or row["guidance"] != 7.5:
            raise ValueError("Sampling settings mismatch")
        if row["alpha"] in (0.0, 1.0):
            seed = row["seed"] if row["alpha"] == 0 else row["end_seed"]
            historical = reference[(row["label"], seed)]
            verify(old_root / historical["file"], historical["png_sha256"])
            for key in ("png_sha256", "initial_noise_sha256"):
                if row[key] != historical[key]:
                    raise ValueError(f"Endpoint mismatch: {key}")
            endpoints += 1
    paths = []
    for label in ("AD", "NC"):
        for start in (37200, 37216):
            path = sorted(
                [r for r in rows if r["label"] == label and r["seed"] == start],
                key=lambda r: r["alpha"],
            )
            paths.append(
                {
                    "label": label,
                    "start_seed": start,
                    "alphas": [r["alpha"] for r in path],
                    "noise_norms": [r["initial_noise_norm"] for r in path],
                }
            )
    result = {
        "job": "639671",
        "elapsed": "00:01:05",
        "exit_code": "0:0",
        "verified_images": 20,
        "exact_endpoints": endpoints,
        "paths": paths,
        "limits": "Linear initial-noise interpolation; interior variance shrinks; not disease interpolation; no test images",
    }
    (root / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
