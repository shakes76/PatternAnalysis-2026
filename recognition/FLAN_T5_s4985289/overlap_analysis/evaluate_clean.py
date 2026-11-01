
import json
from pathlib import Path

from dataset import load_splits
from evaluate_utils import score_predictions

# Load official dataset splits.
ds = load_splits()

def normalize(text):
    return str(text).strip().casefold()

# Collect all training input reports.
train_sources = {
    normalize(row["radiology_report"])
    for row in ds["train"]
}

# Identify validation reports not found in training.
validation = ds["validation"]

clean_indices = [
    i for i, row in enumerate(validation)
    if normalize(row["radiology_report"]) not in train_sources
]

overlap_indices = [
    i for i, row in enumerate(validation)
    if normalize(row["radiology_report"]) in train_sources
]

print("Clean validation samples:", len(clean_indices))
print("Overlapping validation samples:", len(overlap_indices))

# Evaluate all three models on exactly the same subsets.
for mode in ["zero", "lora", "full"]:

    path = Path(
        f"outputs/eval_{mode}_validation/predictions.jsonl"
    )

    with open(path, encoding="utf-8") as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    assert len(rows) == len(validation)

    # Verify prediction alignment with official validation data.
    for i, row in enumerate(rows):
        assert row["source"] == validation[i]["radiology_report"]
        assert row["reference"] == validation[i]["layman_report"]

    def evaluate_subset(indices):
        predictions = [rows[i]["prediction"] for i in indices]
        references = [rows[i]["reference"] for i in indices]
        return score_predictions(predictions, references)

    clean_metrics = evaluate_subset(clean_indices)
    overlap_metrics = evaluate_subset(overlap_indices)

    print(f"\n===== {mode.upper()} =====")
    print("Clean validation ROUGE:")
    print(json.dumps(clean_metrics, indent=2))

    print("Overlapping validation ROUGE:")
    print(json.dumps(overlap_metrics, indent=2))

    output_dir = Path("outputs/clean_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        output_dir / f"{mode}_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "clean_samples": len(clean_indices),
                "overlap_samples": len(overlap_indices),
                "clean": clean_metrics,
                "overlap": overlap_metrics,
            },
            file,
            indent=2
        )

print("\nClean validation evaluation completed.")

