# This file shows an example of how the model can be loaded in and used to make predictions on a singular image

import torch
from modules_convnext import classifier # import the model architecture
from PIL import Image
from torchvision import transforms
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = classifier.to(device)

# Load in the trained weights - this assumes the models weights are in convnext_model.pth
model.load_state_dict(
    torch.load("convnext_model.pth", map_location=device, weights_only=True)
)
model.eval()

# load image - here I have generated a random noisy image, but here is where you would replace image
# with an imported MRI image of these dimensions to be classified

random_pixels = np.random.randint(
    0, 256, (256, 256), dtype=np.uint8
)

# Convert to PIL image
image = Image.fromarray(random_pixels, mode="L")

# predict

# process the image ready for classification
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])

image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()

classification = "AD" if prediction == 1 else "NC"

print("Predicted class is: ", classification)
print(f"Model is {probabilities[0, prediction].item():.2%} confident") 
