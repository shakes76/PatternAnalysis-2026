"""Train-only intensity augmentation on normalized tensors; no image I/O."""
import random

import torch
from torch.utils.data import Dataset


def intensity_protocol(enabled):
    if not enabled:
        return None
    return {"version": 1, "scale_range": [0.9, 1.1], "gamma_range": [0.9, 1.1],
            "probability": 1.0, "formula": "clip(scale * x**gamma, 0, 1)",
            "order": "intensity then original affine then original noise",
            "rng": "python random (saved by train.py)",
            "scope": "training only"}


def apply_intensity(image, scale, gamma):
    if not torch.is_floating_point(image) or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected floating point 3xHxW tensor")
    if not torch.isfinite(image).all() or image.min() < -1 or image.max() > 1:
        raise ValueError("Expected finite normalized pixels in [-1,1]")
    if not 0.9 <= scale <= 1.1 or not 0.9 <= gamma <= 1.1:
        raise ValueError("Intensity parameters outside fixed protocol")
    x = (image + 1.0) * 0.5
    return (scale * x.pow(gamma)).clamp(0.0, 1.0) * 2.0 - 1.0


class MRIIntensityTrainingSet(Dataset):
    def __init__(self, original):
        self.original = original

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        image, label = self.original[index]
        # Python RNG is independently seeded/saved/restored by train.py. This
        # does not consume torch RNG used by legacy affine/noise and DropPath.
        scale = random.uniform(0.9, 1.1)
        gamma = random.uniform(0.9, 1.1)
        return apply_intensity(image, scale, gamma), label


def make_training_input(original, augment, intensity, legacy_wrapper):
    if intensity and not augment:
        raise ValueError("Intensity experiment requires original --augment")
    source = MRIIntensityTrainingSet(original) if intensity else original
    return legacy_wrapper(source) if augment else source
