import random
import torch
import matplotlib.pyplot as plt
from PIL import Image

from dataset import scan_split, ADNIDataset, class_to_idx
from modules import ConvNeXt

torch.backends.cudnn.enabled = False  # MIOpen workaround for Windows ROCm
device = "cuda" if torch.cuda.is_available() else "cpu"

def pred_to_class(number):
    if number < 0.45:
        return "NC"
    return "AD"

# Load the trained model
model = ConvNeXt(num_classes=1)
model.load_state_dict(torch.load("convnext3.pt", map_location=device))
model.to(device).eval()

# Pick a random test image
record = random.choice(scan_split("test"))   # (path, label, patient_id)
path, label, _ = record


# Preprocess exactly as in training/testing, then predict
x = ADNIDataset([record], img_size=224)[0][0].unsqueeze(0).to(device)
with torch.no_grad():
    output = model(x)
    prob_ad = torch.sigmoid(output).item()
pred = 1 if prob_ad >=0.45 else 0
confidence = prob_ad if pred == 1 else 1 - prob_ad

# Display
plt.imshow(Image.open(path).convert("L"), cmap="gray")

plt.axis("off")
verdict = "Alzheimer's detected" if pred == 1 else "No Alzheimer's detected"
plt.title(f"True: {pred_to_class(label)} | Predicted: {pred_to_class(pred)} ({confidence:.1%} confidence)\n{verdict}")
plt.savefig("prediction.png")
plt.show()