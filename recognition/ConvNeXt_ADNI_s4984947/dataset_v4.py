"""Light anatomical augmentation for training MRI slices only."""
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dataset import ADNIDataset


class AugmentedADNIDataset(ADNIDataset):
    def __init__(self, entries):
        super().__init__(entries)
        self.augment = transforms.RandomAffine(
            degrees=8,
            translate=(0.04, 0.04),
            interpolation=InterpolationMode.BILINEAR,
            fill=0
        )

    def __getitem__(self, idx):
        path, label, patient, scan = self.entries[idx]

        with Image.open(path) as img:
            rgb = ImageOps.pad(
                img.convert('RGB'), (224, 224),
                method=Image.Resampling.BILINEAR,
                color=0
            )
            rgb = self.augment(rgb)
            image = self.to_tensor(rgb)

        return {
            'image': image,
            'label': label,
            'patient_id': patient,
            'scan_id': scan
        }
