import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import time
import matplotlib.pyplot as plt
from Models.LearningModels import UNetClassifier
import numpy as np
import nibabel as nib
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("CUDA is not available. Using CPU.")
print("Using device:", device)
#Configuration
num_epochs = 10
learning_rate = 0.001

# Model Initialisation
model = UNetClassifier().to(device)
# Optimiser Initialisation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Loss function
criterion = nn.CrossEntropyLoss()

'''
Dataset class for loading images and masks from the OASIS dataset.
'''
class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path):
        self.image_path = Path(image_path)
        self.mask_path = Path(mask_path)

        self.image_files = sorted(self.image_path.glob("*.nii.gz"))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # Convert image filename to mask filename
        mask_name = img_path.name.replace("case_", "seg_", 1)
        mask_file = self.mask_path / mask_name

        # Load images  
        image = nib.load(str(img_path)).get_fdata()
        mask = nib.load(str(mask_file)).get_fdata()
        image = self.to_tensor(image)
        
        mask = torch.from_numpy(mask.astype(np.int64))

        image = F.interpolate(image.unsqueeze(0), size=(256, 128), mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(256, 128), mode='nearest').squeeze(0).squeeze(0).long()
        return image, mask

    
print("Loading dataset...")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data" / "HIP MRI" / "keras_slices_data"


train_path = DATA_DIR / "keras_slices_train"

# Training dataset
HipTrainset = Dataset(image_path=DATA_DIR / "keras_slices_train",mask_path=DATA_DIR / "keras_slices_seg_train")
HipTrainLoader = torch.utils.data.DataLoader(HipTrainset, batch_size=16, shuffle=False)

# Validation dataset
HipValset = Dataset(image_path=DATA_DIR / "keras_slices_validate",mask_path=DATA_DIR / "keras_slices_seg_validate")
HipValLoader = torch.utils.data.DataLoader(HipValset, batch_size=16, shuffle=False)

# Test dataset
HipTestset = Dataset(image_path=DATA_DIR / "keras_slices_test",mask_path=DATA_DIR / "keras_slices_seg_test")
HipTestLoader = torch.utils.data.DataLoader(HipTestset, batch_size=16,shuffle=False)

print(f"Number of training samples: {len(HipTrainset)}")
print(f"Number of validation samples: {len(HipValset)}")
print(f"Number of test samples: {len(HipTestset)}")
print(f"Batch size: {HipTrainLoader.batch_size}")
print(f"Number of batches in training set: {len(HipTrainLoader)}")
print(f"Number of batches in validation set: {len(HipValLoader)}")
print(f"Number of batches in test set: {len(HipTestLoader)}")
print(f"Image shape: {HipTrainset[0][0].shape}")
print(f"Mask shape: {HipTrainset[0][1].shape}")
print(f"Unique classes in mask: {torch.unique(HipTrainset[0][1])}")


#
# Training Loop
#
num_classes = 6
@torch.no_grad()
def evaluate_dice(model, loader, device, num_classes=6):
    model.eval()

    intersections = torch.zeros(num_classes, device=device)
    pred_counts = torch.zeros(num_classes, device=device)
    true_counts = torch.zeros(num_classes, device=device)

    for images, masks in loader:
        images = images.to(device).float()
        masks = masks.to(device).long()

        logits = model(images)
        predictions = torch.argmax(logits, dim=1)

        for c in range(num_classes):
            predicted = predictions == c
            actual = masks == c

            intersections[c] += (predicted & actual).sum()
            pred_counts[c] += predicted.sum()
            true_counts[c] += actual.sum()

    # Display the first prediction
    '''
    plt.imshow(predictions[0].cpu(), cmap="viridis", vmin=0, vmax=3)
    plt.title("Predicted Brain Segmentation")
    plt.axis("off")
    plt.show()

    '''
    dice = ((2 * intersections) / (pred_counts + true_counts).clamp(min=1))

    return dice.cpu().tolist()

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    time_start = time.time()

    for images, masks in HipTrainLoader:
        images = images.to(device).float()
        masks = masks.to(device).long()

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss = criterion(predictions, masks)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()
    time_end = time.time()

    '''
    Printing the loss and validation Dice scores after each epoch.
    '''

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Loss: {total_loss / len(HipTrainLoader):.4f}, "
        f"Time: {time_end - time_start:.2f}s"
    )
    dice_scores = evaluate_dice(model, HipValLoader, device, num_classes=6)

    print("Validation Dice:")
    for class_id, score in enumerate(dice_scores):
        print(f"  Class {class_id}: {score:.4f}")
    print(f"Mean Dice: {sum(dice_scores) / len(dice_scores):.4f}")

    Test_dice_scores = evaluate_dice(model, HipTestLoader, device, num_classes=6)
    print("Test Dice:")
    for class_id, score in enumerate(Test_dice_scores):
        print(f"  Class {class_id}: {score:.4f}")
    print(f"Mean Dice: {sum(Test_dice_scores) / len(Test_dice_scores):.4f}")
    
