# Shows example usage of trained model. load saved model weights, run inference on test cases, print out results, and provide visualisations where applicable
import csv
import time
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import modules
import dataset

# configuration and file paths

# select model architecture to train: "BasicCNN" or "ConvNeXt"
MODEL_NAME = "ConvNeXt"

OUTPUT_DIR = Path("outputs")

if MODEL_NAME == "BasicCNN":
    CHECKPOINT_PATH = OUTPUT_DIR / "basic_cnn.pth"
    METRICS_PATH = OUTPUT_DIR / "metrics.json"
    HISTORY_PATH = OUTPUT_DIR / "training_history.json"
    EVAL_METRICS_PATH = OUTPUT_DIR / "basic_cnn_evaluation_metrics.json"
else: # MODEL_NAME = ConvNeXt
    CHECKPOINT_PATH = OUTPUT_DIR / "convnext.pth"
    METRICS_PATH = OUTPUT_DIR / "convnext_metrics.json"
    HISTORY_PATH = OUTPUT_DIR / "convnext_history.json"
    EVAL_METRICS_PATH = OUTPUT_DIR / "convnext_evaluation_metrics.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the save model
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)

num_classes = checkpoint["num_classes"]
class_names = checkpoint["class_names"]
history = checkpoint["history"]

if checkpoint["model_name"] == "BasicCNN":
    model = modules.BasicCNN(num_classes=num_classes).to(device)
else:
    model = modules.ConvNeXt(num_classes=num_classes).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Device:", device)
print("\nCheckpoint loaded successfully.")
print("Model:", checkpoint["model_name"])
print("Epochs:", checkpoint["epoch"])
print("Class names:", class_names)

# load saved metrics and history
metrics = {}
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

if HISTORY_PATH.exists():
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)


# training and validation loss
epochs = range(1, len(history["train_loss"]) + 1)

for metric, ylabel, title, filename in [
    ("loss", "Cross-Entropy Loss", "Training and Validation Loss",
     "loss_curves.png"),
    ("accuracy", "Accuracy", "Training and Validation Accuracy",
     "accuracy_curves.png"),
]:
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history[f"train_{metric}"], label="Training")
    plt.plot(epochs, history[f"val_{metric}"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    if metric == "accuracy":
        plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{MODEL_NAME.lower()}_{filename}", dpi=300)
    plt.close()

def synchronise_cuda():
    """Ensure GPU operations finish before timing or reading memory."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)

# run inference to get test probabilities
all_probabilities = []
all_true_labels = []

synchronise_cuda()

inference_start = time.perf_counter()
with torch.no_grad():
    for images, labels in dataset.test_loader:
        images = images.to(device)
        outputs = model(images)
        probabilities = torch.softmax(model(images), dim=1)
        all_probabilities.extend(probabilities.cpu().tolist())
        all_true_labels.extend(labels.tolist())

synchronise_cuda()
inference_end = time.perf_counter()

total_inference_time = inference_end - inference_start
total_samples = len(all_true_labels)
latency_per_sample_ms = (total_inference_time / total_samples) * 1000

peak_gpu_mem_mib = 0.0
if device.type == "cuda":
    peak_gpu_mem_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

probabilities = np.asarray(all_probabilities)
true_labels = np.asarray(all_true_labels)
predicted_labels = np.argmax(probabilities, axis=1)
confidences = np.max(probabilities, axis=1)

print(f"\nInference completed on {total_samples} samples.")
print(f"Total Inference Time: {total_inference_time:.4f} s")
print(f"Latency per Sample: {latency_per_sample_ms:.2f} ms")
if device.type == "cuda":
    print(f"Peak GPU Memory Allocated: {peak_gpu_mem_mib:.2f} MiB")


# confusion matrix
confusion = metrics.get("test_confusion_matrix")
confusion = np.asarray(confusion, dtype=int)

# other metrics
per_class_metrics = {}
for c in range(num_classes):
    tp = confusion[c, c]
    fp = confusion[:, c].sum() - tp
    fn = confusion[c, :].sum() - tp
    tn = confusion.sum() - (tp + fp + fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    per_class_metrics[class_names[c]] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }

macro_f1 = np.mean([m["f1"] for m in per_class_metrics.values()])
recalls = [m["recall"] for m in per_class_metrics.values()]
balanced_accuracy = float(np.mean(recalls))

# brier score & NLL
one_hot_true = np.eye(num_classes)[true_labels]
brier_score = float(np.mean(np.sum((probabilities - one_hot_true) ** 2, axis=1)))
eps = 1e-15
true_class_probs = np.clip(probabilities[np.arange(total_samples), true_labels], eps, 1 - eps)
nll = float(-np.mean(np.log(true_class_probs)))

# ECE (expected calibration error)
def compute_ece(probs, labels, n_bins=10):
    """Computes Expected Calibration Error (ECE) and reliability bin statistics."""
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    accuracies = (preds == labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confs > bin_lower) & (confs <= bin_upper)
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confs[in_bin])
            ece += (bin_count / len(probs)) * np.abs(bin_acc - bin_conf)
            bin_stats.append({"bin_lower": float(bin_lower), "bin_upper": float(bin_upper), "accuracy": float(bin_acc), "confidence": float(bin_conf), "count": int(bin_count)})
    return float(ece), bin_stats

ece, reliability_bins = compute_ece(probabilities, true_labels)

# AUROC and average precision (one-vs-rest)
def compute_roc_pr(y_true_bin, y_score):
    """Computes ROC and Precision-Recall curve coordinates, AUROC, and Average Precision."""
    sorted_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true_bin[sorted_idx]
    y_score_sorted = y_score[sorted_idx]
    
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.concatenate([distinct_value_indices, [len(y_score_sorted) - 1]])
    
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = np.cumsum(1 - y_true_sorted)[threshold_idxs]
    
    total_pos = np.sum(y_true_bin)
    total_neg = len(y_true_bin) - total_pos
    
    tpr = tps / total_pos if total_pos > 0 else np.zeros_like(tps)
    fpr = fps / total_neg if total_neg > 0 else np.zeros_like(fps)
    precision = tps / (tps + fps) if (tps.sum() > 0) else np.ones_like(tps)
    recall = tpr.copy()
    
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    trapezoid_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    
    auroc = float(trapezoid_func(tpr, fpr))
    ap = float(trapezoid_func(precision, recall))
    return fpr, tpr, precision, recall, auroc, ap

auroc_dict = {}
ap_dict = {}
roc_curves_data = {}
pr_curves_data = {}

for c, c_name in enumerate(class_names):
    y_bin = (true_labels == c).astype(int)
    y_score = probabilities[:, c]
    fpr, tpr, prec, rec, auroc, ap = compute_roc_pr(y_bin, y_score)
    auroc_dict[c_name] = auroc
    ap_dict[c_name] = ap
    roc_curves_data[c_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    pr_curves_data[c_name] = {"precision": prec.tolist(), "recall": rec.tolist()}

macro_auroc = float(np.mean(list(auroc_dict.values())))
macro_ap = float(np.mean(list(ap_dict.values())))

# generate plots
# confusion matrix
plt.figure(figsize=(5, 4))
plt.imshow(confusion, cmap="Blues")
plt.colorbar(label="Number of samples")
plt.xticks(range(num_classes), class_names)
plt.yticks(range(num_classes), class_names)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Test Confusion Matrix")
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(confusion[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"{MODEL_NAME.lower()}_confusion_matrix.png", dpi=300)
plt.close()

# probability distributions plot
fig, axes = plt.subplots(1, num_classes, figsize=(10, 4), squeeze=False)
for i, class_name in enumerate(class_names):
    axes[0, i].hist(probabilities[:, i], bins=20, range=(0, 1), edgecolor="black")
    axes[0, i].set_title(f"Predicted P({class_name})")
    axes[0, i].set_xlabel("Softmax Probability")
    axes[0, i].set_ylabel("Number of Samples")
fig.suptitle("Test Set Class Probability Distributions")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / f"{MODEL_NAME.lower()}_probability_distributions.png", dpi=300)
plt.close(fig)

# ROC curves plot
plt.figure(figsize=(6, 5))
for c_name in class_names:
    plt.plot(roc_curves_data[c_name]["fpr"], roc_curves_data[c_name]["tpr"], label=f"{c_name} (AUROC = {auroc_dict[c_name]:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"{MODEL_NAME.lower()}_roc_curves.png", dpi=300)
plt.close()

# precision-recall curves plot
plt.figure(figsize=(6, 5))
for c_name in class_names:
    plt.plot(pr_curves_data[c_name]["recall"], pr_curves_data[c_name]["precision"], label=f"{c_name} (AP = {ap_dict[c_name]:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"{MODEL_NAME.lower()}_pr_curves.png", dpi=300)
plt.close()

# reliability diagram plot
plt.figure(figsize=(5, 5))
bin_accs = [b["accuracy"] for b in reliability_bins]
bin_confs = [b["confidence"] for b in reliability_bins]
plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
plt.plot(bin_confs, bin_accs, marker="o", label=f"Model (ECE = {ece:.4f})")
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.title("Reliability Diagram")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"{MODEL_NAME.lower()}_reliability_diagram.png", dpi=300)
plt.close()

# coverage-risk analysis plot
thresholds = np.linspace(0.0, 1.0, 101)
coverages = []
risks = []
corrects = (predicted_labels == true_labels)

for th in thresholds:
    kept = confidences >= th
    coverage = np.mean(kept)
    coverages.append(coverage)
    if np.sum(kept) > 0:
        risk = 1.0 - np.mean(corrects[kept])
    else:
        risk = 0.0
    risks.append(risk)

plt.figure(figsize=(6, 4))
plt.plot(coverages, risks, marker=".")
plt.xlabel("Coverage (Fraction of Dataset Kept)")
plt.ylabel("Risk (Error Rate on Kept Samples)")
plt.title("Coverage-Risk Curve (Confidence Thresholding)")
plt.gca().invert_xaxis()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"{MODEL_NAME.lower()}_coverage_risk.png", dpi=300)
plt.close()

# predictions per sample and failure cases
failure_cases = []
predictions_export = []

for idx, (t, p, conf, probs) in enumerate(zip(true_labels, predicted_labels, confidences, probabilities)):
    is_correct = (t == p)
    pred_info = {
        "sample_index": idx,
        "true_label_idx": int(t),
        "true_class_name": class_names[t],
        "predicted_label_idx": int(p),
        "predicted_class_name": class_names[p],
        "confidence": float(conf),
        "is_correct": bool(is_correct),
    }
    for c_idx, c_name in enumerate(class_names):
        pred_info[f"prob_{c_name}"] = float(probs[c_idx])
    predictions_export.append(pred_info)

    if not is_correct:
        failure_cases.append({
            "sample_index": idx,
            "true_class": class_names[t],
            "predicted_class": class_names[p],
            "confidence": float(conf),
        })


# export metrics
failure_csv_path = OUTPUT_DIR / f"{MODEL_NAME.lower()}_failure_cases.csv"
with open(failure_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["sample_index", "true_class", "predicted_class", "confidence"])
    writer.writeheader()
    for row in failure_cases:
        writer.writerow(row)

evaluation_metrics = {
    "model_name": MODEL_NAME,
    "total_samples": total_samples,
    "inference_time_seconds": float(total_inference_time),
    "latency_per_sample_ms": float(latency_per_sample_ms),
    "peak_gpu_allocated_mib": float(peak_gpu_mem_mib),
    "macro_f1": float(macro_f1),
    "balanced_accuracy": float(balanced_accuracy),
    "brier_score": float(brier_score),
    "negative_log_likelihood": float(nll),
    "expected_calibration_error": float(ece),
    "macro_auroc": float(macro_auroc),
    "macro_average_precision": float(macro_ap),
    "per_class_metrics": per_class_metrics,
    "per_class_auroc": auroc_dict,
    "per_class_average_precision": ap_dict,
}

with open(EVAL_METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(evaluation_metrics, f, indent=4)

print(f"\nEvaluation metrics successfully saved to {EVAL_METRICS_PATH}")
print(f"Failure cases saved to {failure_csv_path}")
print("\nGenerated plots:")
print(f" - {MODEL_NAME.lower()}_loss_curves.png")
print(f" - {MODEL_NAME.lower()}_accuracy_curves.png")
print(f" - {MODEL_NAME.lower()}_confusion_matrix.png")
print(f" - {MODEL_NAME.lower()}_probability_distributions.png")
print(f" - {MODEL_NAME.lower()}_roc_curves.png")
print(f" - {MODEL_NAME.lower()}_pr_curves.png")
print(f" - {MODEL_NAME.lower()}_reliability_diagram.png")
print(f" - {MODEL_NAME.lower()}_coverage_risk.png")
print(f" - {MODEL_NAME.lower()}_failure_cases.csv")
print(f" - {MODEL_NAME.lower()}_evaluation_metrics.json")