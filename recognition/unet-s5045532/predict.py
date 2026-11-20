import torch
from modules import device, NUM_CLASSES, Improved2DUNet, batch_size, MODEL_SAVE_LOCATION
import matplotlib.pyplot as plt
from dataset import loadTestingData

model = Improved2DUNet(
    in_channels=1,
    num_classes=NUM_CLASSES,
    base_filters=32,
    dropout_rate=0.2
).to(device)
model.load_state_dict(torch.load(MODEL_SAVE_LOCATION, weights_only=True))
#Displaying of the images made with AI
model.eval()
testloader = loadTestingData(batch_size,earlyStop=True)
images, masks = next(iter(testloader))

# Ensure channels-first format
if images.ndim == 3:
    images = images.unsqueeze(1)
elif images.ndim == 4 and images.shape[-1] == 1:
    images = images.permute(0, 3, 1, 2).contiguous()

images = images.to(device, dtype=torch.float32)

with torch.no_grad():
    logits = model(images)
    predictions = logits.argmax(dim=1)

# Display the first sample in the batch
image = images[0, 0].cpu().numpy()
ground_truth = masks[0].cpu().numpy()
prediction = predictions[0].cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Input Image")

axes[1].imshow(ground_truth, cmap="viridis", vmin=0,
               vmax=NUM_CLASSES - 1)
axes[1].set_title("Ground Truth")

axes[2].imshow(prediction, cmap="viridis", vmin=0,
               vmax=NUM_CLASSES - 1)
axes[2].set_title("Prediction")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()