"""Plot saved ADNI results without rerunning model inference."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score

BASE = Path.home() / "data"
EVAL = BASE / "comp3710_a3_evaluation"
OUT = BASE / "comp3710_a3_figures"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    "cnn": BASE / "comp3710_a3_runs_lr1e4/cnn/full",
    "convnext": BASE / "comp3710_a3_runs_pretrained/convnext/full",
}
NAMES = {"cnn": "CNN", "convnext": "Pretrained ConvNeXt-Tiny"}
COLOURS = {"cnn": "#2878B5", "convnext": "#D95319"}
LEVELS = ("slice", "scan", "patient")
LABELS = {0: "NC", 1: "AD"}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["label"] = int(row["label"])
        row["prob_ad"] = float(row["prob_ad"])
        row["prediction"] = int(row["prediction"])
    return rows


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", name)


results = {
    model: read_json(EVAL / model / "test/metrics.json")
    for model in RUNS
}

# 1. Training curves: slice-level train and validation metrics.
fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
for i, (model, folder) in enumerate(RUNS.items()):
    history = read_json(folder / "history.json")["history"]
    epochs = [h["epoch"] for h in history]
    best_epoch = results[model]["saved_epoch"]

    for j, (metric, title) in enumerate([
        ("loss", "Cross-entropy loss"),
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
    ]):
        ax = axes[i, j]
        for split, style in (("train", "-"), ("val", "--")):
            ax.plot(
                epochs, [h[split][metric] for h in history],
                style, label=split.capitalize(), linewidth=1.7,
            )
        ax.axvline(
            best_epoch, color="grey", linestyle=":",
            label=f"Selected epoch {best_epoch}",
        )
        ax.set_title(f"{NAMES[model]}\n{title}")
        ax.set_xlabel("Epoch")
        if metric != "loss":
            ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

fig.suptitle("Training and validation curves (slice level)")
save(fig, "training_curves")

# 2. Test ROC curves at all three evaluation levels.
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for ax, level in zip(axes, LEVELS):
    for model in RUNS:
        rows = read_rows(EVAL / model / "test" / f"{level}_predictions.csv")
        y = [r["label"] for r in rows]
        scores = [r["prob_ad"] for r in rows]
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        assert np.isclose(auc, results[model][level]["auroc"])
        ax.plot(
            fpr, tpr, color=COLOURS[model], linewidth=2,
            label=f"{NAMES[model]}\nAUROC = {auc:.4f}",
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(
        title=f"{level.capitalize()} level",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1), ylim=(0, 1.02),
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)

fig.suptitle("Test ROC curves")
save(fig, "test_roc")

# 3. Confusion matrices: colours are row-normalised; text gives raw counts.
fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
for i, model in enumerate(RUNS):
    for j, level in enumerate(LEVELS):
        ax = axes[i, j]
        cm = np.asarray(results[model][level]["confusion_matrix"])
        normalised = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)

        for row in range(2):
            for col in range(2):
                ax.text(
                    col, row,
                    f"{cm[row, col]}\n({normalised[row, col]:.1%})",
                    ha="center", va="center",
                    color="white" if normalised[row, col] > 0.5 else "black",
                )
        ax.set_xticks([0, 1], ["NC", "AD"])
        ax.set_yticks([0, 1], ["NC", "AD"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"{NAMES[model]}\n{level.capitalize()} level")

fig.suptitle("Test confusion matrices (threshold = 0.5)")
save(fig, "test_confusion_matrices")

# 4. Export validation and test metrics for the report.
fields = [
    "split", "model", "level", "n", "accuracy", "macro_f1",
    "auroc", "sensitivity_ad", "specificity_nc",
]
with open(OUT / "metrics_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for split in ("val", "test"):
        for model in RUNS:
            data = read_json(EVAL / model / split / "metrics.json")
            for level in LEVELS:
                writer.writerow({
                    "split": split, "model": model, "level": level,
                    **{key: data[level][key] for key in fields[3:]},
                })

# 5. Five different misclassified ConvNeXt patients:
#    three most confident false negatives and two most confident false positives.
patients = read_rows(EVAL / "convnext/test/patient_predictions.csv")
slices = read_rows(EVAL / "convnext/test/slice_predictions.csv")
cnn_patients = {
    r["patient_id"]: r
    for r in read_rows(EVAL / "cnn/test/patient_predictions.csv")
}

false_negatives = sorted(
    [r for r in patients if r["label"] == 1 and r["prediction"] == 0],
    key=lambda r: (r["prob_ad"], r["patient_id"]),
)[:3]
false_positives = sorted(
    [r for r in patients if r["label"] == 0 and r["prediction"] == 1],
    key=lambda r: (-r["prob_ad"], r["patient_id"]),
)[:2]
selected = false_negatives + false_positives
assert len(selected) == 5, "Expected three FN and two FP patients."

fig, axes = plt.subplots(1, 5, figsize=(16, 4.5))
case_records = []
for index, (ax, patient) in enumerate(zip(axes, selected), start=1):
    pid = patient["patient_id"]
    members = [r for r in slices if r["patient_id"] == pid]

    # Illustration only: choose the slice score nearest the patient score.
    representative = min(
        members,
        key=lambda r: (abs(r["prob_ad"] - patient["prob_ad"]), r["path"]),
    )
    with Image.open(representative["path"]) as image:
        ax.imshow(image.convert("L"), cmap="gray")

    cnn = cnn_patients[pid]
    error = "False negative" if patient["label"] == 1 else "False positive"
    ax.set_title(
        f"Case {index}: {error}\n"
        f"True: {LABELS[patient['label']]}\n"
        f"ConvNeXt: {LABELS[patient['prediction']]} "
        f"(P(AD)={patient['prob_ad']:.3f})\n"
        f"CNN: {LABELS[cnn['prediction']]} "
        f"(P(AD)={cnn['prob_ad']:.3f})",
        fontsize=9,
    )
    ax.axis("off")
    case_records.append({
        "case": index,
        "error_type": error,
        "patient_id": pid,
        "true_label": patient["label"],
        "convnext_patient_prob_ad": patient["prob_ad"],
        "cnn_patient_prob_ad": cnn["prob_ad"],
        "representative_scan_id": representative["scan_id"],
        "representative_slice_path": representative["path"],
        "representative_slice_prob_ad": representative["prob_ad"],
    })

fig.suptitle("Selected ConvNeXt patient-level test errors")
fig.text(
    0.5, 0.04,
    "Scores are patient-level mean probabilities. Each image is one illustrative "
    "slice; it does not represent all scans of the patient.",
    ha="center", fontsize=9,
)
fig.subplots_adjust(top=0.72, bottom=0.15, wspace=0.12)
save(fig, "test_failure_cases")

with open(OUT / "failure_case_details.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(case_records[0]))
    writer.writeheader()
    writer.writerows(case_records)

print("\nOutput directory:", OUT)
print("Failure-case selection: 3 most confident FN + 2 most confident FP.")
print("Models, thresholds and saved predictions were not changed.")
