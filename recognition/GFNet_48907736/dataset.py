"""
File structure is: ADNI_Processed/{train,val,test}/{0_NC,1_AD}/<scanID>_<slice>.jpeg

AD is positive class
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ["NC", "AD"]

def get_loader(data_root, split):
    """Data set loader for the model"""
    # The MRI scans are made grayscale and resized to 224x224
    steps = [transforms.Grayscale(1), transforms.CenterCrop(200), transforms.Resize((224, 224))]
    # Training images are augmented to prevent memorisation
    # Preprocessing and training steps justified in README.md
    if split == "train":
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1))]
    # Scale pixels from [0, 1] to [-1, 1] 
    steps += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]

    data = datasets.ImageFolder(f"{data_root}/{split}", transform=transforms.Compose(steps))
    return DataLoader(data, batch_size=64, shuffle=(split == "train"), num_workers=0)
