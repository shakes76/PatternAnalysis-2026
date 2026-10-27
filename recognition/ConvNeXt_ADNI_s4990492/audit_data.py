import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/groups/comp3710/ADNI")
OUT = Path(__file__).resolve().parent / "audit"
OUT.mkdir(exist_ok=True)

with (ROOT / "meta_data_with_label.json").open() as f:
    metadata = json.load(f)

rows = []
patient_classes = defaultdict(set)
split_patients = defaultdict(set)
summary = {}
errors = []

for split in ("train", "test"):
    for group in ("AD", "NC"):
        folder = ROOT / "AD_NC" / split / group
        files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        patients = set()
        scans = set()

        for image in files:
            scan_id = image.stem.split("_")[0]
            record = metadata.get(scan_id, {})
            match = re.search(
                r"(?<!\d)\d{3}_S_\d{4}(?!\d)",
                str(record.get("raw", ""))
            )
            expected_label = 2 if group == "AD" else 0

            if not match or record.get("label") != expected_label:
                errors.append(str(image))
                continue

            patient_id = match.group()
            patients.add(patient_id)
            scans.add(scan_id)
            split_patients[split].add(patient_id)
            patient_classes[patient_id].add(group)

            rows.append({
                "path": str(image),
                "original_split": split,
                "class_name": group,
                "target": 1 if group == "AD" else 0,
                "scan_id": scan_id,
                "patient_id": patient_id,
            })

        summary[f"{split}/{group}"] = {
            "images_found": len(files),
            "matched_patients": len(patients),
            "matched_scans": len(scans),
        }

summary["total_matched_images"] = len(rows)
summary["total_unique_patients"] = len(patient_classes)
summary["patients_in_both_original_splits"] = len(
    split_patients["train"] & split_patients["test"]
)
summary["patients_with_multiple_classes"] = sum(
    len(groups) > 1 for groups in patient_classes.values()
)
summary["invalid_records"] = len(errors)

with (OUT / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)

with (OUT / "invalid_records.json").open("w") as f:
    json.dump(errors, f, indent=2)

print(json.dumps(summary, indent=2))

if errors:
    raise SystemExit("Found invalid records; inspect audit/invalid_records.json")

fields = [
    "path", "original_split", "class_name",
    "target", "scan_id", "patient_id"
]
with (OUT / "image_manifest.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved audit summary and image manifest to:", OUT)