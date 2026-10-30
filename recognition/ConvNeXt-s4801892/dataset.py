# Contains data loader for loading & preprocessing data, including leakagefree train/validation/test splitting functions and data augmentations
from pathlib import Path
from sklearn.model_selection import train_test_split

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# root directory for ADNI dataset
root = Path("/home/groups/comp3710/ADNI/AD_NC")

# collect training images
images = [
    (f, label, f.stem.split("_")[0])
    for label, cls in [(1, "AD"), (0, "NC")] # label: AD = 1, NC = 0
    for f in (root / "train" / cls).glob("*.jpeg")
]

# split unique patients, stratified by diagnosis
patients = list({(pid, label) for _, label, pid in images})
patient_ids, labels = zip(*patients)

train_ids, val_ids = train_test_split(
    patient_ids,
    test_size=15/85,
    random_state=42,
    stratify=labels,
)

train_ids, val_ids = set(train_ids), set(val_ids)

train_images = [(f, y) for f, y, pid in images if pid in train_ids]
val_images = [(f, y) for f, y, pid in images if pid in val_ids]

# keep the original test set unchanged
test_images = [
    (f, label)
    for label, cls in [(1, "AD"), (0, "NC")]
    for f in (root / "test" / cls).glob("*.jpeg")
]

# data augmentation & transformation pipelines
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1156], std=[0.2229])
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1156], std=[0.2229])
])  

class MRIDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and transforming 2D brain MRI slices.
    """
    def __init__(self, images, transform=None):
        """
        Initialises the MRIDataset.
        
        Args:
            images (list): List of tuples containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Fetches and processes a single image and label by index.
        
        Args:
            idx (int): Index of the sample to fetch.
            
        Returns:
            tuple: (transformed_image_tensor, label_tensor)
        """
        path, label = self.images[idx]

        image = Image.open(path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# data loader instantiations
train_loader = DataLoader(
    MRIDataset(train_images, train_transform),
    batch_size=32,
    shuffle=True,
)

val_loader = DataLoader(
    MRIDataset(val_images, transform),
    batch_size=32,
    shuffle=False,
)

test_loader = DataLoader(
    MRIDataset(test_images, transform),
    batch_size=32,
    shuffle=False,
)
