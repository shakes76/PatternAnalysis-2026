import csv
import json
from dataset import split_patient_rows
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "audit" / "image_manifest.csv"
OUT = BASE / "splits"
SEED = 42

with SOURCE.open(newline="") as f:
    reader = csv.DictReader(f)
    fields = list(reader.fieldnames)
    rows = list(reader)

# Each patient must have exactly one class in this dataset.
patient_class = {}
for row in rows:
    patient = row["patient_id"]
    label = row["class_name"]
    if patient in patient_class and patient_class[patient] != label:
        raise ValueError("Patient has conflicting classes")
    patient_class[patient] = label

split_rows = split_patient_rows(rows, seed=SEED)

# Verify that no patient appears in more than one split.
patient_sets = {
    name: {row["patient_id"] for row in items}
    for name, items in split_rows.items()
}
assert not (patient_sets["train"] & patient_sets["val"])
assert not (patient_sets["train"] & patient_sets["test"])
assert not (patient_sets["val"] & patient_sets["test"])
assert sum(len(items) for items in split_rows.values()) == len(rows)
assert set.union(*patient_sets.values()) == set(patient_class)

# Avoid silently overwriting an existing split.
if OUT.exists():
    raise SystemExit("splits already exists; stop and check before replacing it")
OUT.mkdir()

summary = {
    "seed": SEED,
    "status": "provisional_pending_tutor_confirmation",
    "patient_overlap_between_splits": 0,
    "splits": {},
}

for name, items in split_rows.items():
    with (OUT / f"{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)

    summary["splits"][name] = {
        "patients": len(patient_sets[name]),
        "patients_by_class": dict(Counter(
            patient_class[p] for p in patient_sets[name]
        )),
        "scans": len({row["scan_id"] for row in items}),
        "images": len(items),
        "images_by_class": dict(Counter(
            row["class_name"] for row in items
        )),
    }

with (OUT / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("\nSaved split manifests to:", OUT)