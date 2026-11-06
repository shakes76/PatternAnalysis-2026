"""
Makes predictions using a given model for any given image
For heatmaps, use recognition/GFNet_48907736/data_processing/heatmap.py

python predict.py --weights gfnet.pt image1.jpeg image2.jpeg
"""
import argparse

import torch
from PIL import Image
from torchvision import transforms

from dataset import CLASSES
from modules import load_model
import os
import matplotlib.pylab as plt
import csv


LABELS = ["0_NC", "1_AD"]

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+")
parser.add_argument("--weights", default="gfnet.pt")
args = parser.parse_args()

# Use a GPU if possible
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(args.weights, device)

# Same preprocessing as the val/test loader in dataset.py
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.CenterCrop(200),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Make predictions using the model
results = []
with torch.no_grad():
    for path in args.images:
        image = transform(Image.open(path)).unsqueeze(0).to(device)
        # Softmax is used to get P(AD) and P(NC)
        probs = torch.softmax(model(image), dim=1)[0]
        pred = probs.argmax().item()
        print(f"{path}: {CLASSES[pred]} (P(AD) = {probs[1]:.3f})")
        results.append((path, image[0, 0].cpu(), pred, probs[1].item()))

# Write predictios to CSV
with open("predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "prediction", "p_ad"])
    for path, _, pred, p_ad in results:
        writer.writerow([path, LABELS[pred], f"{p_ad:.3f}"])
print("Saved predictions.csv")

# Plot onto an image
fig, axes = plt.subplots(1, len(results), figsize=(3 * len(results), 3.4), squeeze=False)
for ax, (path, img, pred, p_ad) in zip(axes[0], results):
    folder = os.path.basename(os.path.dirname(path)) 
    title = f"{os.path.basename(path)}\nPred: {LABELS[pred]}  P(AD)={p_ad:.2f}"
    colour = "black"
    if folder in LABELS:
        title += f"\nTrue: {folder}"
        colour = "green" if LABELS.index(folder) == pred else "red"
    ax.imshow(img * 0.5 + 0.5, cmap="gray")
    ax.set_title(title, fontsize=8, color=colour)
    ax.axis("off")

plt.tight_layout()
plt.savefig("predictions.png", dpi=150)
plt.close(fig)