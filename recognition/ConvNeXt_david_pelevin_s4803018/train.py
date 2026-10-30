import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import modules
import dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Change these to fit current folder structure
train_folder = "./ADNI/AD_NC/train"
test_folder = "./ADNI/AD_NC/test"

## Removes Randomness and allows for Reproducibility
## Based off suggestions from https://docs.pytorch.org/docs/2.15/notes/randomness.html
torch.manual_seed(50)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

train_generator = torch.Generator()
train_generator.manual_seed(50)

## Splits the ADNI dataset into training, testing and validation datasets by patient id
train_split, val_split, test_split = dataset.train_val_test_split(train_folder, test_folder, "AD", "NC")

train_loader = DataLoader(dataset.ADNI_train(train_folder, test_folder, "AD", "NC", train_split), batch_size=32, shuffle=True, generator=train_generator)
val_loader = DataLoader(dataset.ADNI_test(train_folder, test_folder, "AD", "NC", val_split), batch_size=32, shuffle=False)
test_loader = DataLoader(dataset.ADNI_test(train_folder, test_folder, "AD", "NC", test_split), batch_size=32, shuffle=False)

## Hyper-parameters
epochs = 50
learning_rate = 4e-4

## ConvNeXt Model
model = modules.ConvNeXt()
model = model.to(device)

## Cross Entropy Loss, Label Smoothing, and AdamW Optimiser chosen using
## "A ConvNet for the 2020s" found at https://arxiv.org/pdf/2201.03545
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

## Validation Checkpoints
best_val = 0.0

print("Training Start")

## Training code is based off code from previous demo
## Originally adapated from
## https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
model.train()
for epoch in range(epochs):
    ## Training
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

    ## Validation Set Predictions
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Epoch: {epoch} Loss: {total_loss / len(train_loader)} Validation accuracy: {correct/total:.4f}")

    ## Validation Checkpoints - save model when validation accuracy peaks
    if (correct/total) > best_val:
        best_val = correct / total
        torch.save(model.state_dict(), f'/content/drive/My Drive/checkpoints/cnn_{epoch}.pth')

    model.train()

## Test Set Predictions
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test accuracy: {correct/total:.4f}")

## Save model for later
torch.save(model.state_dict(), './model_weights/convnext.pth')