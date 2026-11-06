"""
Trains the CNN model and save the weights that produce the best validation accuracy
It saves loss and accuracy curves.

python train.py --data-root "C:/Users/elvin/OneDrive/Desktop/UQ/Year 3, Semester 2/COMP3710/Final Project/ADNI_Processed" --epochs 100
"""
import argparse

import matplotlib.pyplot as plt
import torch
from torch import nn

from dataset import get_loader
from modules import SimpleCNN

# Makes training reproducible
torch.manual_seed(42)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()

# Use a GPU if possible
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

train_loader = get_loader(args.data_root, "train")
val_loader = get_loader(args.data_root, "val")

# Model uses ADAM with weight decay
model = SimpleCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
# Cosine scheduler decays the learning rate smoothly
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# What happens for each epoch
def run_epoch(loader, train):
    """
    For each epoch, this is what happens.
    It goes through the images provided, makes a prediction, calculates a loss, then updates the weights
    """
    model.train(train)
    total_loss, correct, count = 0, 0, 0
    with torch.set_grad_enabled(train):
        # Go through the slices
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Need to weight by batch size since the last one will be smaller
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            count += len(labels)
    return total_loss / count, correct / count

# Track these stats to make graphs at the end
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0

# Go through the epochs
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    scheduler.step()
    val_loss, val_acc = run_epoch(val_loader, train=False)
    for key, value in zip(history, [train_loss, val_loss, train_acc, val_acc]):
        history[key].append(value)
    print(f"Epoch {epoch}: train loss {train_loss:.3f} acc {train_acc:.3f} | val loss {val_loss:.3f} acc {val_acc:.3f}")

    # Keep the model that performs the best on the val set
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "cnn.pt")
        print("New highest val accuracy")

print(f"Best val accuracy: {best_val_acc:.3f}")

# Plots
epochs = range(1, len(history["train_loss"]) + 1)

# Training loss and validation loss vs number of epochs
plt.figure()
plt.plot(epochs, history["train_loss"], label="Training")
plt.plot(epochs, history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig("CNN_loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# Training accuracy and validation accuracy vs number of epochs
plt.figure()
plt.plot(epochs, history["train_acc"], label="Training")
plt.plot(epochs, history["val_acc"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig("CNN_accuracy_curve.png", dpi=150, bbox_inches="tight")
plt.close()