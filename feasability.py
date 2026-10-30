from pathlib import Path
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

'''
This was used for the feasability checkoff
'''


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Folder containing this Python script
BASE_DIR = Path(__file__).resolve().parent

# ADNI training folder
DATA_DIR = BASE_DIR / "ADNI" / "AD_NC" / "train"

BATCH_SIZE = 16
IMAGE_SIZE = 224
VALIDATION_RATIO = 0.2
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------
# Step 1: Collect patient IDs
# ---------------------------------------------------------

def get_patient_id(image_path):
    """
    Filename format:
        <patientID>_<sliceNo>.jpeg

    Example:
        1003730_100.jpeg

    Returns:
        1003730
    """
    return image_path.stem.split("_")[0]


def build_patient_split(data_dir, validation_ratio=0.2):
    """
    Creates patient-independent train and validation splits.
    """

    samples = []

    class_map = {
        "AD": 0,
        "NC": 1
    }

    for class_name, label in class_map.items():
        class_dir = data_dir / class_name

        for image_path in class_dir.glob("*.jpeg"):
            patient_id = get_patient_id(image_path)

            samples.append({
                "path": image_path,
                "label": label,
                "patient_id": patient_id
            })

    patient_ids = sorted(set(sample["patient_id"] for sample in samples))

    random.shuffle(patient_ids)

    validation_count = int(len(patient_ids) * validation_ratio)

    validation_patients = set(patient_ids[:validation_count])
    train_patients = set(patient_ids[validation_count:])

    train_samples = [
        sample for sample in samples
        if sample["patient_id"] in train_patients
    ]

    validation_samples = [
        sample for sample in samples
        if sample["patient_id"] in validation_patients
    ]

    return train_samples, validation_samples, train_patients, validation_patients


# ---------------------------------------------------------
# Step 2: Dataset
# ---------------------------------------------------------

class ADNIDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample["path"]).convert("RGB")
        label = sample["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------
# Step 3: Simple baseline CNN
# ---------------------------------------------------------

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


# ---------------------------------------------------------
# Step 4: Create patient-level split
# ---------------------------------------------------------

train_samples, val_samples, train_patients, val_patients = build_patient_split(
    DATA_DIR,
    VALIDATION_RATIO
)

print("Device:", DEVICE)

print("\nPatient split")
print("--------------------")
print("Training patients:", len(train_patients))
print("Validation patients:", len(val_patients))

print("\nImage split")
print("--------------------")
print("Training images:", len(train_samples))
print("Validation images:", len(val_samples))

overlap = train_patients.intersection(val_patients)

print("\nPatient overlap:", len(overlap))

assert len(overlap) == 0, "Patient leakage detected!"


# ---------------------------------------------------------
# Step 5: Transforms and DataLoaders
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

train_dataset = ADNIDataset(
    train_samples,
    transform=transform
)

val_dataset = ADNIDataset(
    val_samples,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ---------------------------------------------------------
# Step 6: Create model
# ---------------------------------------------------------

model = SimpleCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nModel information")
print("--------------------")
print("Total parameters:", total_parameters)
print("Trainable parameters:", trainable_parameters)


# ---------------------------------------------------------
# Step 7: Smoke-test training
# ---------------------------------------------------------

model.train()

start_time = time.time()

running_loss = 0.0

MAX_BATCHES = 20

for batch_index, (images, labels) in enumerate(train_loader):

    if batch_index >= MAX_BATCHES:
        break

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    running_loss += loss.item()

    print(
        f"Batch {batch_index + 1:02d} "
        f"Loss: {loss.item():.4f}"
    )


elapsed_time = time.time() - start_time

average_loss = running_loss / MAX_BATCHES

print("\nSmoke-test result")
print("--------------------")
print(f"Average loss: {average_loss:.4f}")
print(f"Training time: {elapsed_time:.2f} seconds")


# ---------------------------------------------------------
# Step 8: GPU memory usage
# ---------------------------------------------------------

if torch.cuda.is_available():

    peak_memory_bytes = torch.cuda.max_memory_allocated()

    peak_memory_gb = peak_memory_bytes / (1024 ** 3)

    print(f"Peak GPU memory: {peak_memory_gb:.3f} GB")

else:
    print("GPU memory: not available because CUDA is not being used.")


print("\nFeasibility smoke test completed successfully.")