"""Compare review thresholds on validation predictions."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "training_results" / "gcn_20261030_002758"
SPLIT = "val"
THRESHOLDS = [0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]

PRED = RUN / "prediction"
CLASSES = ["company", "government", "politician", "tvshow"]


# Read saved validation predictions.
data = []
predictions_path = PRED / "predictions.csv"
with predictions_path.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"] == SPLIT:
            confidences = []
            for c in CLASSES:
                confidences.append(float(row[f"p_{c}"]))
            confidence = max(confidences)
            correct = row["true"] == row["pred"]
            data.append((confidence, correct))

print(f"{SPLIT}: {len(data)} nodes")
print(f"{'Threshold':>9} {'Auto accuracy':>14} {'Review count':>14} {'Review %':>10}")

# Accept predictions at or above each threshold.
for threshold in THRESHOLDS:
    automatic = []
    for score, correct in data:
        if score >= threshold:
            automatic.append(correct)

    if automatic:
        accuracy = sum(automatic) / len(automatic) * 100
    else:
        accuracy = float("nan")

    review = len(data) - len(automatic)
    percent = review / len(data) * 100

    print(f"{threshold:>9.2f} {accuracy:>13.2f}% {review:>14} {percent:>9.2f}%")