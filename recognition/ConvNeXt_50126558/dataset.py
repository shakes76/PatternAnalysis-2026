# import files to perform data-preprocessing and augmentation
import os
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = "/home/groups/comp3710/ADNI/AD_NC"

# remove the black border around the brain scan (left, top, right, bottom)
def crop(im):
    return im.crop((32, 8, 232, 210))

class ADNIDataset(Dataset):
    def __init__(self, samples, transform):
        # samples is a list of (image path, label) pairs
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        # images are loaded and converted to single-channel greyscale
        return self.transform(Image.open(path).convert("L")), label

def get_dataloaders(batch_size=32):
    def load(split):
        # label: NC = 0, AD = 1, sorted listing keeps the order deterministic
        return [(os.path.join(ROOT, split, c, f), int(c == "AD"))
                for c in ("NC", "AD")
                for f in sorted(os.listdir(os.path.join(ROOT, split, c))) if f.endswith(".jpeg")]

    def pid(path):
         # patient ID is the filename minus the trailing slice number
        return os.path.basename(path).rsplit("_", 1)[0]

    train, test_all = load("train"), load("test")
    test_patients = sorted({pid(p) for p, _ in test_all})
    # shuffle the test patients
    random.Random(42).shuffle(test_patients)
    # split 30% of test patients for validation
    val_ids = set(test_patients[:int(len(test_patients) * 0.3)])
    val = [s for s in test_all if pid(s[0]) in val_ids]
    test = [s for s in test_all if pid(s[0]) not in val_ids]
    
    # crop away the black border, then resize to 224 x 224 as ConvNeXt expects
    plain = transforms.Compose([transforms.Lambda(crop), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # apply rotation, shifting, scaling, vary brightness to image
    aug = transforms.Compose([transforms.Lambda(crop), transforms.Resize((224, 224)), transforms.RandomAffine(10, (0.05, 0.05), (0.95, 1.05)),
                          transforms.ColorJitter(0.2, 0.2), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    def loader(samples, tf, shuffle):
        return DataLoader(ADNIDataset(samples, tf), batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    return loader(train, aug, True), loader(val, plain, False), loader(test, plain, False)