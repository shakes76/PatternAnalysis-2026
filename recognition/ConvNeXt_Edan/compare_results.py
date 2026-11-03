import json
from pathlib import Path

MODELS = ["baseline", "convnext"]

SECTIONS = {
    "slice_level": "Slice-level (0.50)",
    "patient_level_default": "Patient-level (0.50)",
    "patient_level_selected": "Patient-level (selected)",
}

METRICS = [
    ("threshold", "Threshold", ".2f"),
    ("accuracy", "Accuracy", ".4f"),
    ("balanced_accuracy", "Balanced accuracy", ".4f"),
    ("sensitivity", "Sensitivity", ".4f"),
    ("specificity", "Specificity", ".4f"),
    ("precision", "Precision", ".4f"),
    ("f1", "F1 score", ".4f"),
    ("auroc", "AUROC", ".4f"),
    ("brier_score", "Brier score", ".4f"),
    ("tn", "True negatives", "d"),
    ("fp", "False positives", "d"),
    ("fn", "False negatives", "d"),
    ("tp", "True positives", "d"),
]

data = {}

for model in MODELS:
    path = Path(f"outputs/{model}/test/metrics.json")

    if not path.exists():
        raise FileNotFoundError(f"Missing results: {path}")

    with open(path) as f:
        data[model] = json.load(f)

print("\n" + "=" * 85)
print("FINAL HELD-OUT TEST COMPARISON")
print("=" * 85)

for section, title in SECTIONS.items():
    print(f"\n{title}")
    print("-" * 85)

    print(
        f"{'Metric':<25}"
        f"{'Baseline CNN':>20}"
        f"{'ConvNeXt-Tiny':>20}"
    )

    print("-" * 65)

    for key, label, fmt in METRICS:
        values = []

        for model in MODELS:
            value = data[model][section].get(key)

            if value is None:
                formatted = "N/A"
            else:
                formatted = format(value, fmt)

            values.append(formatted)

        print(
            f"{label:<25}"
            f"{values[0]:>20}"
            f"{values[1]:>20}"
        )

print("\n" + "=" * 85)
print("MODEL COMPLEXITY AND INFERENCE")
print("=" * 85)

for model in MODELS:
    metrics = data[model]["slice_level"]

    print(f"\n{model.upper()}")
    print(f"Parameters: {metrics.get('total_parameters', 'N/A'):,}")
    print(
        "Inference latency: "
        f"{metrics.get('latency_ms_per_image', 0):.4f} ms/image"
    )

print("\nNOTE:")
print("Patient-level thresholds were selected using validation data only.")
print("Test results must not be used to re-optimise thresholds.")
print("Inference latency was measured using different batch sizes.")
