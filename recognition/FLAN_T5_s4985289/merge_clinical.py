
import csv
import json
from pathlib import Path

base_dir = Path("outputs/clinical")
model_names = ["zero", "lora", "full"]

predictions = {}

# Load the predictions from each model.
for model_name in model_names:
    file_path = base_dir / model_name / "predictions.jsonl"

    with open(file_path, "r", encoding="utf-8") as file:
        predictions[model_name] = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

# Check that the three models evaluated the same number of cases.
number_of_cases = len(predictions["zero"])

for model_name in model_names:
    assert len(predictions[model_name]) == number_of_cases, (
        f"Different case count for {model_name}"
    )

rows = []

for index in range(number_of_cases):
    zero = predictions["zero"][index]
    lora = predictions["lora"][index]
    full = predictions["full"][index]

    # Check that cases are aligned across all models.
    assert zero["source"] == lora["source"] == full["source"], (
        f"Source mismatch at case {index + 1}"
    )

    assert zero["reference"] == lora["reference"] == full["reference"], (
        f"Reference mismatch at case {index + 1}"
    )

    rows.append({
        "case_id": index + 1,
        "original_report": zero["source"],
        "reference_summary": zero["reference"],
        "zero_shot": zero["prediction"],
        "lora": lora["prediction"],
        "full": full["prediction"]
    })

# Export a CSV file for clinical safety analysis.
output_path = base_dir / "clinical_comparison.csv"

with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Successfully merged {len(rows)} cases.")
print(f"Saved to: {output_path}")
