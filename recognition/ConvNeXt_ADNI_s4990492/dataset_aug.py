"""Training-only spatial augmentation for the existing ADNI loader."""

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dataset import ADNISliceDataset


class AugmentedADNISliceDataset(ADNISliceDataset):
    """Apply small random rotations and translations to training images."""

    def __init__(self, manifest_path):
        super().__init__(manifest_path)
        self.augmentation = transforms.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return self.augmentation(image), label
