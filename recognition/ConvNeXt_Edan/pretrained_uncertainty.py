"""Patient-stratified bootstrap evaluation of three model configurations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)

ROOT = Path(__file__).resolve().parent
N_BOOTSTRAPS = 5000
SEED = 42

models = {
    "Baseline CNN": (
        "outputs/baseline/test/patient_predictions.csv", 0.78
    ),
    "Scratch ConvNeXt": (
        "outputs/convnext/test/patient_predictions.csv", 0.45
    ),
    "Pretrained ConvNeXt": (
        "outputs/pretrained_trial/convnext/test/patient_predictions.csv",
        0.64,
    ),
}

frames = {}
for name, (filename, threshold) in models.items():
    frame = pd.read_csv(ROOT / filename).set_index("patient_id")
    assert frame.index.is_unique, f"Duplicate patient: {name}"
    frames[name] = frame

ids = sorted(frames["Pretrained ConvNeXt"].index)
assert len(ids) == 102

y = frames["Pretrained ConvNeXt"].loc[ids, "true_label"].to_numpy(dtype=int)
assert (y == 0).sum() == 69
assert (y == 1).sum() == 33

probs = {}
preds = {}

for name, (_, threshold) in models.items():
    assert set(frames[name].index) == set(ids)
    labels = frames[name].loc[ids, "true_label"].to_numpy(dtype=int)
    np.testing.assert_array_equal(labels, y)

    probs[name] = frames[name].loc[ids, "probability_ad"].to_numpy(dtype=float)
    assert np.isfinite(probs[name]).all()
    preds[name] = (probs[name] >= threshold).astype(int)

assert accuracy_score(y, preds["Pretrained ConvNeXt"]) == 88 / 102


def evaluate(labels, predictions, probabilities):
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "Balanced accuracy": balanced_accuracy_score(labels, predictions),
        "AD sensitivity": recall_score(labels, predictions),
        "NC specificity": recall_score(labels, predictions, pos_label=0),
        "AD precision": precision_score(labels, predictions, zero_division=0),
        "AD F1": f1_score(labels, predictions, zero_division=0),
        "Macro-F1": f1_score(labels, predictions, average="macro"),
        "AUROC": roc_auc_score(labels, probabilities),
        "Brier score": brier_score_loss(labels, probabilities),
    }


pretrained = "Pretrained ConvNeXt"
point = evaluate(y, preds[pretrained], probs[pretrained])

rng = np.random.default_rng(SEED)
nc_indices = np.flatnonzero(y == 0)
ad_indices = np.flatnonzero(y == 1)

samples = {metric: [] for metric in point}
paired = {
    name: {"Accuracy": [], "Balanced accuracy": []}
    for name in models if name != pretrained
}

for _ in range(N_BOOTSTRAPS):
    indices = np.concatenate([
        rng.choice(nc_indices, size=len(nc_indices), replace=True),
        rng.choice(ad_indices, size=len(ad_indices), replace=True),
    ])

    yy = y[indices]
    pre = evaluate(yy, preds[pretrained][indices], probs[pretrained][indices])

    for metric, value in pre.items():
        samples[metric].append(value)

    for name in paired:
        comparison = preds[name][indices]
        paired[name]["Accuracy"].append(
            pre["Accuracy"] - accuracy_score(yy, comparison)
        )
        paired[name]["Balanced accuracy"].append(
            pre["Balanced accuracy"]
            - balanced_accuracy_score(yy, comparison)
        )


def interval(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


lines = [
    "### 7.7 Supplementary Pretrained-Model Statistical Uncertainty",
    "",
    "Statistical uncertainty for the supplementary pretrained model was "
    "estimated using 5,000 stratified patient-level bootstrap resamples "
    "with seed 42. Each resample retained 69 NC and 33 AD patient "
    "observations. The validation-selected thresholds remained fixed.",
    "",
    "| Patient-level metric | Estimate | 95% bootstrap interval |",
    "|---|---:|---:|",
]

for metric, value in point.items():
    low, high = interval(samples[metric])
    lines.append(
        f"| {metric} | {value:.4f} | [{low:.4f}, {high:.4f}] |"
    )

lines += [
    "",
    "#### Paired Comparisons",
    "",
    "The same resampled patient indices were used across all models. "
    "Differences below are pretrained ConvNeXt minus the comparison model.",
    "",
    "| Comparison | Accuracy difference (pp) | "
    "Balanced-accuracy difference (pp) |",
    "|---|---:|---:|",
]

for name in paired:
    other = evaluate(y, preds[name], probs[name])
    formatted = []

    for metric in ["Accuracy", "Balanced accuracy"]:
        difference = 100 * (point[metric] - other[metric])
        low, high = interval(paired[name][metric])
        formatted.append(
            f"{difference:+.2f} [{100*low:+.2f}, {100*high:+.2f}]"
        )

    lines.append(
        f"| Pretrained vs {name} | {formatted[0]} | {formatted[1]} |"
    )

lines += [
    "",
    "These intervals describe sampling uncertainty conditional on the "
    "existing cohort and trained models. They do not capture variation "
    "across training seeds, patient partitions, threshold selection or "
    "external clinical populations. The pretrained experiment reused "
    "the previously examined test cohort; these retrospective intervals "
    "are not independent validation.",
]

section = "\n".join(lines)
start = "<!-- PRETRAINED_BOOTSTRAP_START -->"
end = "<!-- PRETRAINED_BOOTSTRAP_END -->"
block = start + "\n" + section + "\n" + end

readme_path = ROOT / "README.md"
readme = readme_path.read_text()

# Preserve a backup outside the Git repository.
backup = Path.home() / "COMP3710_README_before_final_uncertainty.md"
if not backup.exists():
    backup.write_text(readme)

# Insert or refresh the uncertainty section.
if start in readme:
    a = readme.index(start)
    b = readme.index(end, a) + len(end)
    readme = readme[:a] + block + readme[b:]
else:
    marker = "## 8. Prediction Confidence and Calibration"
    assert readme.count(marker) == 1
    readme = readme.replace(marker, block + "\n\n" + marker, 1)

def add_scope(marker, content):
    global readme
    assert marker in readme
    if content not in readme:
        readme = readme.replace(
            marker, marker + "\n\n" + content, 1
        )

add_scope(
    "## 7. Experimental Results",
    "Sections 7.1–7.5 report the original baseline CNN and "
    "scratch-trained ConvNeXt experiments. Section 7.6 compares all "
    "three configurations, and Section 7.7 reports the supplementary "
    "pretrained-model uncertainty analysis.",
)

add_scope(
    "## 8. Prediction Confidence and Calibration",
    "The original calibration figures and reject-option analyses below "
    "refer to the baseline CNN and scratch-trained ConvNeXt unless "
    "otherwise stated. The supplementary pretrained model's Brier "
    "score is reported separately.",
)

add_scope(
    "## 9. Computational Efficiency",
    "The initial two-column resource table compares the baseline CNN "
    "and scratch-trained ConvNeXt. The pretrained ConvNeXt measurements "
    "are described immediately afterward.",
)

old = "This is the primary patient-level comparison."
new = (
    "This is the primary patient-level comparison for the original "
    "scratch-trained experiments. The complete three-model comparison "
    "appears in Section 7.6."
)
if old in readme:
    readme = readme.replace(old, new, 1)

old = (
    "ConvNeXt's lower Brier score is a meaningful advantage even "
    "though its final balanced accuracy was similar to the baseline."
)
new = (
    "The scratch-trained ConvNeXt's lower Brier score was an advantage "
    "over the baseline despite similar balanced accuracy. The "
    "supplementary pretrained model improved both measures."
)
if old in readme:
    readme = readme.replace(old, new, 1)

tree_old = "├── statistical_analysis.py"
tree_new = "├── statistical_analysis.py\n├── pretrained_uncertainty.py"
if tree_new not in readme:
    assert tree_old in readme
    readme = readme.replace(tree_old, tree_new, 1)

repro = (
    "### 13.11 Reproducing Pretrained Statistical Uncertainty\n\n"
    "Run `python pretrained_uncertainty.py` after producing all three "
    "models' test patient prediction CSVs. The script recomputes "
    "patient-stratified confidence intervals and paired comparisons "
    "with seed 42 and refreshes Section 7.7 of this README.\n\n"
)

if "### 13.11 Reproducing Pretrained Statistical Uncertainty" not in readme:
    assert "## 14. Conclusion" in readme
    readme = readme.replace(
        "## 14. Conclusion", repro + "## 14. Conclusion", 1
    )

readme_path.write_text(readme.rstrip() + "\n")

print("\nUPDATED README.md")
print(f"Patients: {len(ids)}")
print(f"Bootstrap resamples: {N_BOOTSTRAPS}")
print("\n" + section)
print(f"\nBackup: {backup}")
