"""
dataset.py
Authored by: Benigno Della Bosca

Provides an interface for loading ADNI MR images, as well as per epoch
transformations. Also implements Latent space pre-encoding for usage in unet
training
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

#NOTE: Previously the data was identified as possible at risk of data leakage,
# according to tutors the preprocessing supplied is sufficient
# comm -12 <(ls -R1 train | cut -d'_' -f1 | sort -u) <(ls -R1 test | cut -d'_' -f1 | sort -u)

# Parameters
#-------------------------------------------------------------------------------

train_path = "AD_NC/train"
test_path = "AD_NC/test"

rotation = (-5, 5)
translation = (0.05, 0.05)
brightness = 0.1
contrast = 0.1

CLASSES = {"NC": 0.0, "AD": 1.0}

# Transforms
#-------------------------------------------------------------------------------
# Move the slices around a bit to avoid memorisation
augment = transforms.Compose([
    transforms.Pad((0,8), fill=0), # dimension mismatch solution 240 actual
    transforms.RandomRotation(degrees=rotation, fill=0),
    transforms.RandomAffine(degrees=0, translate=translation, fill=0),
    transforms.ColorJitter(brightness=brightness, contrast=contrast),
    transforms.ToTensor()
])

# Just pad and turn to tensor
nil_transform = transforms.Compose([
    transforms.Pad((0,8), fill=0),
    transforms.ToTensor()
])

# Datasets
#-------------------------------------------------------------------------------
"""
ADNIDataset
the ADNI dataset requires single slice sagittal MRI with the classification
extracted from the file path
base: dir location
path: either train_path, or test_path from expected directory structure
transform: An augmentation transform to avoid aggressive learning over epochs
"""
class ADNIDataset(Dataset):
    def __init__(self, base, path, transform):
        self.transform = transform
        self.nc_dir =  os.path.join(base, path, "NC")
        self.ad_dir =  os.path.join(base, path, "AD")
        self.slices = []

        for file in os.listdir(self.nc_dir):
            filepath = os.path.join(self.nc_dir, file)
            self.slices.append((filepath,"NC"))

        for file in os.listdir(self.ad_dir):
            filepath = os.path.join(self.ad_dir, file)
            self.slices.append((filepath, "AD"))

    def __len__(self):
        return len(self.slices)

    # return slice and classification
    def __getitem__(self, ix):
        path, classification = self.slices[ix]
        slice = Image.open(path).convert("L")
        if self.transform: slice = self.transform(slice)

        label = torch.tensor(CLASSES[classification], dtype=torch.long)
        return slice, label

"""
ADNILatents
A small dataset implementation for returning the latent encoding and class for
UNet training
latents: latent space encodings
classes: classifications
"""
class ADNILatents(Dataset):
    def __init__(self, latents, classes):
        self.latents = latents
        self.classes = classes

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, ix):
        return self.latents[ix], self.classes[ix]

# Loaders
#-------------------------------------------------------------------------------
# test set
def get_test_ldr(base, batch_size=32, num_workers=4):
    # no need to augment
    test_ds = ADNIDataset(base, path=test_path, transform=nil_transform)
    test_ldr = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return test_ldr 

# train set
def get_train_ldr(base, batch_size=32, num_workers=4): 
    train_ds = ADNIDataset(base, path=train_path, transform=augment)
    train_ldr = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True, # random order per epoch
        num_workers=num_workers,
        pin_memory=True
    )
    return train_ldr

def get_latents_ldr(vae, ldr, dev, batch_size=32, shuffle=True):
    with torch.no_grad():
        # requires a pretrained VAE
        vae.eval()
        latents, classes = [], []
 
        for slices, classification in ldr:
            slices = slices.to(dev)
            latent = vae.get_latent(slices)
            latents.append(latent.cpu())
            classes.append(classification)

        latents = torch.cat(latents, dim=0)
        classes = torch.cat(classes, dim=0)

        # rescale latents depending on latent space distribution
        raw_std = latents.std().item()
        scale = 1.0 / raw_std
        print(f"[Latent Norm] Raw std: {raw_std:.4f} | Scale: {scale:.4f}")

        latents = latents * scale

        latent_ds = ADNILatents(latents, classes)
        latent_ldr = DataLoader(
            latent_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True
        )
    return latent_ldr
        
