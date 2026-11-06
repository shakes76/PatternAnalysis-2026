"""
Evaluates the saved model on the test set or validation set.
It saves a confusion matrix and ROC curve, prints out metrics, and writes predictions to a CSV

python evaluate.py --data-root "C:/Users/elvin/OneDrive/Desktop/UQ/Year 3, Semester 2/COMP3710/Final Project/ADNI_Processed" --weights cnn.pt
"""
import argparse
import time
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score)
from dataset import CLASSES, get_loader
from modules import SimpleCNN
import os

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
parser.add_argument("--weights", default="cnn.pt")
parser.add_argument("--split", default="test") # test or val set
args = parser.parse_args()

# Use a GPU if possible
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)
ckpt = torch.load(args.weights, map_location=device)
model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
model.eval()

# Load images
loader = get_loader(args.data_root, args.split)

if device == "cuda":
    torch.cuda.reset_peak_memory_stats()


labels, preds, prob_ad = [], [], [] # Keep track of for stats

# Predict all images in the data loader
with torch.no_grad():
    for images, y in loader:
        # Applies a softmax to get P(AD) and P(NC)
        probs = torch.softmax(model(images.to(device)), dim=1).cpu()
        labels += y.tolist()
        preds += probs.argmax(1).tolist()
        prob_ad += probs[:, 1].tolist()

# AD is positive class
is_AD = [int(y == 1) for y in labels]

# Confusion Matrix
cm = confusion_matrix(labels, preds)
recall = cm[1, 1] / cm[1].sum() # AD correctly found
specificity = cm[0, 0] / cm[0].sum() # NC correctly found

# Statistics
print(f"\n SLICE LEVEL STATS")
print(classification_report(labels, preds, target_names=CLASSES, digits=3))
print(f"Accuracy:    {accuracy_score(labels, preds):.3f}")
print(f"Macro-F1:    {f1_score(labels, preds, average='macro'):.3f}")
print(f"Recall:      {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"AUROC:       {roc_auc_score(is_AD, prob_ad):.3f}")

print("\n SCAN LEVEL STATS")
# Scan-level statistics. Average the slice probabilities for each scan
scan_probs, scan_true = {}, {}
for (path, _), y, p in zip(loader.dataset.samples, labels, prob_ad):
    scan_id = os.path.basename(path).split("_")[0]  # "1003730_100.jpeg" -> "1003730"
    scan_probs.setdefault(scan_id, []).append(p)
    scan_true[scan_id] = y

# Group slices by scan
scan_ids = sorted(scan_probs)
scan_mean = [sum(scan_probs[s]) / len(scan_probs[s]) for s in scan_ids]
scan_labels = [scan_true[s] for s in scan_ids]
scan_preds = [int(m >= 0.5) for m in scan_mean]

print(f"Scan Accuracy: {accuracy_score(scan_labels, scan_preds):.3f}")
print(f"Scan Macro-F1: {f1_score(scan_labels, scan_preds, average='macro'):.3f}")
print(f"Scan AUROC:    {roc_auc_score(scan_labels, scan_mean):.3f}")

print("\n COMPUTATION STATS")
print(f"Parameters:  {sum(p.numel() for p in model.parameters()):,}")
if device == "cuda":
    print(f"Peak VRAM:   {torch.cuda.max_memory_allocated() / 1e6:.1f} MB (batch of {loader.batch_size})")

# Inference time 
image = next(iter(loader))[0][:1].to(device)
times = []
with torch.no_grad():
    for i in range(110):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(image)
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= 10:  # take median inference time of 100 runs excluding first 10
            # This is because the first 10 can be slow
            times.append((time.perf_counter() - start) * 1000)
times.sort()
print(f"Median Inference Time:     {times[len(times) // 2]:.2f}ms per image")

# Plots
# Confusion Matrix
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues")
plt.title(f"Confusion matrix ({args.split})")
plt.savefig("CNN_confusion_matrix.png")
plt.close()

# ROC Curve
RocCurveDisplay.from_predictions(is_AD, prob_ad, name="SimpleCNN")
plt.title(f"ROC curve, AD as positive ({args.split})")
plt.savefig("CNN_roc_curve.png")
plt.close()

# Write all predictions for all images into a CSV
with open("CNN_test_predictions.csv", "w") as file:
    file.write("path,true,pred,prob_AD,correct\n")
    for (path, _), y, p, prob in zip(loader.dataset.samples, labels, preds, prob_ad):
        file.write(f"{path},{CLASSES[y]},{CLASSES[p]},{prob:.3f},{int(y == p)}\n")

print("Done")