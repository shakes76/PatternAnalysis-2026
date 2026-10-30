import random
import torch
import matplotlib.pyplot as plt
from PIL import Image

from dataset import scan_split, ADNIDataset, class_to_idx
from modules import ConvNeXt

torch.backends.cudnn.enabled = False
device = "cuda" if torch.cuda.is_available() else "cpu"

def pred_to_class(number):
    if number < 0.45:
        return "NC"
    return "AD"

# Load the trained model
model = ConvNeXt(num_classes=1)
model.load_state_dict(torch.load("convnext.pt", map_location=device))
model.to(device).eval()

# Get test records and shuffle them
records = scan_split("test")
random.shuffle(records)

# Find an incorrectly predicted image
incorrect_record = None
incorrect_prob = None
incorrect_pred = None

with torch.no_grad():
    for record in records:
        path, label, patient_id = record

        x = ADNIDataset([record], img_size=224)[0][0].unsqueeze(0).to(device)

        output = model(x)
        prob_ad = torch.sigmoid(output).item()
        pred = 1 if prob_ad >= 0.45 else 0

        if pred != label:
            incorrect_record = record
            incorrect_prob = prob_ad
            incorrect_pred = pred
            break

# Display the incorrectly predicted image
if incorrect_record is not None:
    path, label, patient_id = incorrect_record

    confidence = incorrect_prob if incorrect_pred == 1 else 1 - incorrect_prob

    plt.imshow(Image.open(path).convert("L"), cmap="gray")
    plt.axis("off")

    verdict = "Alzheimer's detected" if incorrect_pred == 1 else "No Alzheimer's detected"

    plt.title(
        f"Patient: {patient_id}\n"
        f"True: {pred_to_class(label)} | "
        f"Predicted: {pred_to_class(incorrect_pred)} "
        f"({confidence:.1%} confidence)\n{verdict}"
    )

    plt.savefig("incorrect_prediction.png")
    plt.show()

else:
    print("No incorrectly classified images found.")