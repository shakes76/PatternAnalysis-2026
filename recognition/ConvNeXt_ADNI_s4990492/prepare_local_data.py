"""Map audited remote manifests to local paths without changing membership."""
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

EXPECTED = {
    "train": "ef7e7b83dcf0f4933a05fa955945daab9e85d926461a6cae17742bb1c48704bb",
    "val": "e1217d294d7d0c93749e7661b1dff3d929488a65ad1ab8aecbad19498e16945a",
    "test": "4cb9fa4d33870a7af4606c98285088076a8f781c2eff8e760e21637283ff7d88",
}


def main():
    root = Path(__file__).resolve().parents[2] / "local_data"
    destination = root / "splits_local"
    if destination.exists():
        raise FileExistsError("splits_local already exists; inspect rather than overwrite")
    groups, fields, hashes = {}, {}, {}
    for name in EXPECTED:
        source = root / "splits" / f"{name}.csv"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != EXPECTED[name]:
            raise ValueError(f"Original {name} manifest hash differs; stop and inspect")
        with source.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fields[name] = reader.fieldnames
            rows = list(reader)
        for row in rows:
            remote = row["path"].replace(chr(92), "/")
            marker = "/AD_NC/"
            if marker not in remote:
                raise ValueError("Unexpected original path layout")
            relative = remote.split(marker, 1)[1]
            image = (root / "AD_NC" / relative).resolve()
            if not image.is_relative_to((root / "AD_NC").resolve()):
                raise ValueError("Image path escapes local data root")
            if not image.is_file():
                raise FileNotFoundError(f"Missing local image in {name}: {image.name}")
            row["path"] = str(image)
        groups[name], hashes[name] = rows, digest
    for left, right in combinations(groups, 2):
        for key in ("patient_id", "scan_id", "path"):
            if {r[key] for r in groups[left]} & {r[key] for r in groups[right]}:
                raise ValueError(f"Overlap: {left}/{right}, {key}")
    destination.mkdir()
    report = {"original_manifest_sha256": hashes, "splits": {}}
    for name, rows in groups.items():
        target = destination / f"{name}.csv"
        with target.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields[name])
            writer.writeheader()
            writer.writerows(rows)
        report["splits"][name] = {
            "images": len(rows), "patients": len({r["patient_id"] for r in rows}),
            "local_manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    (destination / "mapping_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("PASS: membership retained; all local paths exist; no cross-split overlap.")
    print("Image contents were not decoded or compared in this path check.")


if __name__ == "__main__":
    main()
