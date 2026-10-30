
"""Train and evaluate the GFNet or the baseline CNN on the ADNI slices.

Choose the model with `model_name` below. Each epoch the model is evaluated on
the patient-level validation set and the checkpoint with the best validation
accuracy is kept. The test set is used once at the end, with that checkpoint.

Saved to `out_dir` (Google Drive, not the repository):
    best_<model>.pt          best-validation checkpoint
    <model>_curves.png       training/validation loss and validation accuracy
    <model>_test_probs.pt    test softmax probabilities and labels

The training loop is adapted from the COMP3710 MNIST CNN practical.

Note: this file holds the regularised setup (AdamW, cosine learning-rate
schedule, 40 epochs; for GFNet weight decay 0.05, dropout 0.1, stochastic depth
0.3). The baseline CNN and plain GFNet results in the report were trained with
an earlier version (Adam, no schedule, 30 epochs); see the commit history.
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataset import get_dataloaders
from modules import ConvNetwork, GFNet
from utils import collect_probs

# Hyper-parameters
num_epochs = 40
model_name = "gfnet"   # "cnn" or "gfnet"
learning_rate = 3e-4 if model_name == "gfnet" else 1e-3
out_dir = "/content/drive/MyDrive/gfnet_runs"   # checkpoints/plots go to Drive, not git


def evaluate(model, loader, criterion, device):
    """Loss and accuracy on a loader, with no weight updates."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            n += labels.size(0)
    return total_loss / n, correct / n


def main():
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader = get_dataloaders("/content/AD_NC")
    model = (GFNet(drop=0.1, drop_path_rate=0.3) if model_name == "gfnet" else ConvNetwork()).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05 if model_name == "gfnet" else 0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    total_step = len(train_loader)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc, ckpt = 0.0, os.path.join(out_dir, f"best_{model_name}.pt")

    print("> Training")
    start = time.time()
    for epoch in range(num_epochs):
        model.train()                      # back to train mode after validation
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Validate once per epoch and keep the best checkpoint
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(running_loss / total_step)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch + 1}: train loss {history['train_loss'][-1]:.4f} | "
              f"val loss {val_loss:.4f} | val acc {val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt)

    elapsed = time.time() - start
    print("Training took " + str(elapsed) + " secs or " + str(elapsed / 60) + " mins in total")
    if device.type == "cuda":
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Loss and accuracy curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Validation accuracy"); ax[1].set_xlabel("epoch")
    fig.savefig(os.path.join(out_dir, f"{model_name}_curves.png"), dpi=150, bbox_inches="tight")

    # Test once, with the best validation checkpoint
    print("> Testing")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f} | Test accuracy {test_acc:.3f} (per slice)")
    probs, labels = collect_probs(model, test_loader, device)
    torch.save({"probs": probs, "labels": labels}, os.path.join(out_dir, f"{model_name}_test_probs.pt"))


if __name__ == "__main__":
    main()
