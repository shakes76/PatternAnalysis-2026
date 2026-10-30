import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

# Based off code from previous demo
class ADNIDataset(Dataset):
    def __init__(self, root_directory, ad_directory, nc_directory):
        self.images = []
        self.labels = []

        ad_path = Path(root_directory + "/" + ad_directory)
        nc_path = Path(root_directory + "/" + nc_directory)

        for path in ad_path.glob("*.jpeg"):
            image = Image.open(path).convert("L")

            # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
            image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(torch.float32)

            self.images.append(image_tensor)
            self.labels.append(torch.tensor(1))

        for path in nc_path.glob("*.jpeg"):
            image = Image.open(path).convert("L")
            
            # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
            image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(torch.float32)

            self.images.append(image_tensor)
            self.labels.append(torch.tensor(0))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

train_loader = DataLoader(ADNIDataset("./ADNI/AD_NC/train", "AD", "NC"), batch_size=32, shuffle=True)
test_loader = DataLoader(ADNIDataset("./ADNI/AD_NC/test", "AD", "NC"), batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Code developed using the following websites
## https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
## https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(5824, 128)
        
        self.fc2 = nn.Linear(128, 2)

    def forward(self, input):
        input = self.pool(F.relu(self.conv1(input)))
        input = self.pool(F.relu(self.conv2(input)))
        input = self.pool(F.relu(self.conv3(input)))
        input = self.pool(F.relu(self.conv4(input)))

        input = torch.flatten(input, 1)
        input = F.relu(self.fc1(input))
        input = self.fc2(input)
        return input

# Hyper-parameters
epochs = 50
learning_rate = 1e-3

model = CNN()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Based off code from previous demo
model.train()
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

    print(f"Epoch: {epoch} Loss: {total_loss / len(train_loader)}")

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

torch.save(model.state_dict(), './model_weights/cnn.pth')