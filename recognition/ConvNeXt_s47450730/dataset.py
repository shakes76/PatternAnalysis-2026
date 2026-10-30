# Here resides my logic for segmenting and preparing my train/test/validation dataloaders
#this code is imported into train.py
import torch
import os
import random
from collections import defaultdict

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms

# Data augmentation for training images only
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])


# Consistent preprocessing for validation and test images
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])


# Remap labels: AD = 1, NC = 0
class RemappedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, 1 - label


# Load training images with augmentation
full_trainset = ImageFolder(
    root="/home/groups/comp3710/ADNI/AD_NC/train",
    transform=train_transform
)

# Load training images without augmentation for validation
full_trainset_eval = ImageFolder(
    root="/home/groups/comp3710/ADNI/AD_NC/train",
    transform=eval_transform
)

# Load independent test images without augmentation
full_testset = ImageFolder(
    root="/home/groups/comp3710/ADNI/AD_NC/test",
    transform=eval_transform
)


# Keep the full test set for final evaluation.
# Split the original training data 70/30 by patient for training/validation.

# Group training images by patient ID
patient_to_indices = defaultdict(list)

# Filenames are in the form 1413656_107.jpeg.
# The patient ID is the part before the underscore.
for index, (path, _) in enumerate(full_trainset.samples):
    filename = os.path.basename(path)
    patient_id = filename.split("_", 1)[0]
    patient_to_indices[patient_id].append(index)


# Shuffle patient IDs reproducibly
patient_ids = list(patient_to_indices.keys())
random.Random(42).shuffle(patient_ids)


# Split patients: 70% training / 30% validation
split_index = round(0.70 * len(patient_ids))

train_patients = set(patient_ids[:split_index])
validation_patients = set(patient_ids[split_index:])


# Convert patient groups into image indices
train_indices = [
    index
    for patient_id in train_patients
    for index in patient_to_indices[patient_id]
]

validation_indices = [
    index
    for patient_id in validation_patients
    for index in patient_to_indices[patient_id]
]


# Data augmentation for training images only
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])


# Consistent preprocessing for validation and test images
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])


# Remap labels: AD = 1, NC = 0
class RemappedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, 1 - label


# Load training images with augmentation
full_trainset = ImageFolder(
    root="/home/groups/comp3710/ADNI/AD_NC/train",
    transform=train_transform
)

# Load training images without augmentation for validation
full_trainset_eval = ImageFolder(
    root="/home/groups/comp3710/ADNI/AD_NC/train",
    transform=eval_transform
)

# Load independent test images without augmentation
full_testset = ImageFolder(
    root="/home/groups/comp3710/ADNI/AD_NC/test",
    transform=eval_transform
)


# Keep the full test set for final evaluation.
# Split the original training data 70/30 by patient for training/validation.

# Group training images by patient ID
patient_to_indices = defaultdict(list)

# Filenames are in the form 1413656_107.jpeg.
# The patient ID is the part before the underscore.
for index, (path, _) in enumerate(full_trainset.samples):
    filename = os.path.basename(path)
    patient_id = filename.split("_", 1)[0]
    patient_to_indices[patient_id].append(index)


# Shuffle patient IDs reproducibly
patient_ids = list(patient_to_indices.keys())
random.Random(42).shuffle(patient_ids)


# Split patients: 70% training / 30% validation
split_index = round(0.70 * len(patient_ids))

train_patients = set(patient_ids[:split_index])
validation_patients = set(patient_ids[split_index:])


# Convert patient groups into image indices
train_indices = [
    index
    for patient_id in train_patients
    for index in patient_to_indices[patient_id]
]

validation_indices = [
    index
    for patient_id in validation_patients
    for index in patient_to_indices[patient_id]
]


# Training subset uses augmented images
trainset = Subset(full_trainset, train_indices)

# Validation subset uses deterministic preprocessing only
validationset = Subset(full_trainset_eval, validation_indices)


# Apply label remapping to all three datasets
trainset = RemappedDataset(trainset)
validationset = RemappedDataset(validationset)
testset = RemappedDataset(full_testset)


# Data loaders
train_loader = DataLoader(
    trainset,
    batch_size=128,
    shuffle=True
)

validation_loader = DataLoader(
    validationset,
    batch_size=100,
    shuffle=False
)

test_loader = DataLoader(
    testset,
    batch_size=100,
    shuffle=False
)

#print information so we know how the sets look training
print(f"Training images:    {len(trainset)}")
print(f"Validation images:  {len(validationset)}")
print(f"Test images:        {len(testset)}")
print(f"Training patients:  {len(train_patients)}")
print(f"Validation patients: {len(validation_patients)}")
print(f"Patient overlap:    {len(train_patients & validation_patients)}")
