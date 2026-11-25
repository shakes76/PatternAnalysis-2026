from pathlib import Path
import random
import time

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "ADNI" / "AD_NC" / "train"
TEST_DIR = BASE_DIR / "ADNI" / "AD_NC" / "test"

def get_patient_id(image_path):
    return image_path.stem.split("_")[0]


def build_patient_split(data_dir, validation_ratio=0.2, seed=42):
    samples = []

    class_map = {"AD": 0, "NC": 1}

    for class_name, label in class_map.items():
        class_dir = data_dir / class_name

        # groups by patient id
        for image_path in class_dir.glob("*.jpeg"):
            patient_id = get_patient_id(image_path)

            samples.append({"path": image_path, "label": label, "patient_id": patient_id})

    patient_ids = sorted(set(sample["patient_id"] for sample in samples))

    rand = random.Random(seed)
    rand.shuffle(patient_ids)

    validation_count = int(len(patient_ids) * validation_ratio)

    validation_patients = set(patient_ids[:validation_count])
    train_patients = set(patient_ids[validation_count:])

    train_samples = [sample for sample in samples if sample["patient_id"] in train_patients]

    validation_samples = [sample for sample in samples if sample["patient_id"] in validation_patients]

    return train_samples, validation_samples, train_patients, validation_patients

def load_test_samples(test_dir):
    samples = []

    class_map = {"AD": 0,"NC": 1}

    for class_name, label in class_map.items():
        class_dir = test_dir / class_name

        for image_path in class_dir.glob("*.jpeg"):

            patient_id = get_patient_id(image_path)

            samples.append({"path": image_path, "label": label, "patient_id": patient_id})

    return samples

class ADNIDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # turns image into grey scale
        image = Image.open(sample["path"]).convert("L")
        label = sample["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

# previous testing to find best runs
"""class PerImageStandardise:
    Z-score each image using its own mean/std (operates on a tensor).
    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, x):
        return (x - x.mean()) / (x.std() + self.eps)

NORM = PerImageStandardise()"""

train_samples, val_samples, train_patients, val_patients = build_patient_split(DATA_DIR)

test_samples = load_test_samples(TEST_DIR)

"""train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.02, 0.02)
    ),
    transforms.ToTensor(),
    NORM
])"""


train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation(5), transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)), transforms.ToTensor()])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train_dataset = ADNIDataset(train_samples, transform=train_transform)

val_dataset = ADNIDataset(val_samples, transform=val_transform)

test_dataset = ADNIDataset(test_samples, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


assert train_patients.isdisjoint(val_patients), \
    "Patient leakage detected between training and validation sets"

"""test_patients = set(
    sample["patient_id"]
    for sample in test_samples
)

print("Train patients:", len(train_patients))
print("Validation patients:", len(val_patients))
print("Test patients:", len(test_patients))

print("Train images:", len(train_samples))
print("Validation images:", len(val_samples))
print("Test images:", len(test_samples))

assert train_patients.isdisjoint(test_patients), \
    "Patient leakage between training and test"

assert val_patients.isdisjoint(test_patients), \
    "Patient leakage between validation and test"

print("Test patients:", len(test_patients))
print("Test images:", len(test_samples))"""