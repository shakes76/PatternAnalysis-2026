
from collections import defaultdict
from dataset import load_splits

ds = load_splits()

def normalize(text):
    return str(text).strip().casefold()

# Build a lookup of training reports and their reference summaries.
train_targets = defaultdict(set)

for row in ds["train"]:
    source = normalize(row["radiology_report"])
    target = normalize(row["layman_report"])
    train_targets[source].add(target)

source_overlap = 0
pair_overlap = 0
clean_validation = 0

for row in ds["validation"]:
    source = normalize(row["radiology_report"])
    target = normalize(row["layman_report"])

    if source in train_targets:
        source_overlap += 1

        if target in train_targets[source]:
            pair_overlap += 1
    else:
        clean_validation += 1

print("\n===== DATA OVERLAP AUDIT =====")
print("Training samples:", len(ds["train"]))
print("Validation samples:", len(ds["validation"]))
print("Validation samples sharing training input:", source_overlap)
print("Validation samples sharing input and target:", pair_overlap)
print("Validation samples without exact training input:", clean_validation)
print(
    "Validation input overlap rate:",
    f"{100 * source_overlap / len(ds['validation']):.2f}%"
)


