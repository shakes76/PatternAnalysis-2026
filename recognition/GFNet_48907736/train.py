"""
Trains the GFNet model and save the weights that produce the best validation accuracy
It saves loss and accuracy curves.

python train.py --data-root /content/ADNI_Processed --epochs 100 
Minimum 5 epochs
"""
import argparse

import matplotlib.pyplot as plt
import torch
from torch import nn

from dataset import get_loader
from modules import GFNet

# Makes training reproducible
torch.manual_seed(42)

# Get data and info on running script 
parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()

# Use a GPU if possible
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

# Load data
train_loader = get_loader(args.data_root, "train")
val_loader = get_loader(args.data_root, "val")

config = {"dim": 256, "depth": 12, "drop_path_rate": 0.3, "conv_stem": True}
model = GFNet(**config).to(device)

# Model uses cross entropy loss
loss_fn = nn.CrossEntropyLoss()

# The model uses ADAM with weight decay. Choice of optimiser, decay and scheduling made to fit
# in line with paper
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-4 * 64 / 512, weight_decay=0.05)
warmup = 5
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
    torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.016, total_iters=warmup),
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup, eta_min=1e-5),
], milestones=[warmup])

# Loop for each epoch
def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """
    For each epoch, this is what happens.
    It goes through the images provided, makes a prediction, calculates a loss, then updates the weights
    """
    train = optimizer is not None
    model.train(train)
    total_loss, correct, count = 0, 0, 0
    with torch.set_grad_enabled(train):
        # Go through the slices
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            # Need to weight by batch size since the last one will be smaller
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            count += len(labels)
    return total_loss / count, correct / count

# Keep track of these so plots can be made
train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc = 0

# Running the epochs
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
    scheduler.step()

    # Recording losses and accuracies for each epoch
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    print(f"Epoch {epoch}: train loss: {train_loss:.3f} acc: {train_acc:.3f} | "
            f"val loss: {val_loss:.3f} val acc: {val_acc:.3f}")

    # Keep the model that performs the best on the val set
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({"state_dict": model.state_dict(), "config": config}, "gfnet.pt")
        print("New highest val accuracy")

epochs = range(1, args.epochs + 1)

# Plot stats
# Training loss and validation loss vs number of epochs
plt.figure()
plt.plot(epochs, train_losses, label="Training")
plt.plot(epochs, val_losses, label="Validation")
plt.title("GFNet Loss")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("GFNet_loss_curve.png")
plt.close()

# Training accuracy and validation accuracy vs number of epochs
plt.figure()
plt.plot(epochs, train_accs, label="Training")
plt.plot(epochs, val_accs, label="Validation")
plt.title("GFNet Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("GFNet_accuracy_curve.png")
plt.close()