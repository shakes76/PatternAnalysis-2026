import torch

from pathlib import Path
from PIL import Image
from torchvision import transforms
from modules import ConvNeXt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {0: "AD", 1: "NC"}
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

'''
Loads ConvNeXt model
'''
def load_model(model_path):
    model = ConvNeXt(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

'''
Uses best model to predict AD or NC
'''
def predict_image(model, image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output,dim=1)
        confidence, prediction = torch.max(probabilities,dim=1)

    predicted_class = prediction.item()
    confidence = confidence.item()

    return (predicted_class, confidence, probabilities.squeeze(0).cpu())


if __name__ == "__main__":
    model_path = "convnext_best.pth"
    image_path = input("Enter the path to an ADNI image: ").strip()
    image_path = Path(image_path)

    if not image_path.exists():
        print("Image file does not exist.")
        raise SystemExit

    model = load_model(model_path)
    predicted_class, confidence, probabilities = predict_image(model, image_path)

    print("\nPrediction Results")

    print("\nPredicted class:", CLASS_NAMES[predicted_class])

    print(f"Confidence: {confidence:.4f}")

    print(f"AD probability: {probabilities[0]:.4f}")

    print(f"NC probability: {probabilities[1]:.4f}")