"""Prespecified image-only appearance experiments outside the frozen v1 core.

These are cache-space synthetic transforms, not physical MRI simulations.
Increasing signed gamma preserves intensity ordering; it cannot reproduce an
organ-order reversal. Polarity inversion is a separate, decreasing transform.
No transform uses a label to find support: preserved zeros are image == 0,
which is an input-value rule rather than an anatomical background definition.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


VARIANTS = ("gamma_only", "gamma_polarity")
STRESS_CONDITIONS = ("clean", "gamma_0.7", "gamma_1.5", "polarity", "spatial_gain")
POLICY = {
    "version": "cache_appearance_v1", "gamma_probability": 0.5,
    "gamma_range": [0.7, 1.5], "polarity_probability": 0.25,
    "clip_range": [-5.0, 5.0], "spatial_gain_h_endpoints": [0.7, 1.3],
}


def _keyed_rng(seed, epoch, case_key, stream):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer.")
    if not isinstance(case_key, str) or not case_key:
        raise ValueError("case_key must be a nonempty patient/visit identifier.")
    payload = json.dumps([POLICY["version"], seed, epoch, case_key, stream],
                         separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest(), "big"))


def deterministic_params(seed, epoch, case_key, variant):
    """Case/epoch keyed draws, independent of global NumPy/Torch RNG states.

    Both variants draw exactly the same gamma. Only gamma_polarity enables the
    independent polarity stream. Values do not depend on iteration order.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown appearance variant: {variant}.")
    rng = _keyed_rng(seed, epoch, case_key, "gamma")
    apply_gamma = bool(rng.random() < POLICY["gamma_probability"])
    gamma_draw = float(np.exp(rng.uniform(np.log(0.7), np.log(1.5))))
    polarity_draw = float(_keyed_rng(seed, epoch, case_key, "polarity").random())
    return {"gamma": gamma_draw if apply_gamma else 1.0,
            "gamma_applied": apply_gamma, "gamma_draw": gamma_draw,
            "polarity": bool(variant == "gamma_polarity" and polarity_draw < 0.25)}


def _image_backend(image):
    if isinstance(image, np.ndarray):
        if image.dtype != np.float32 or image.ndim not in (3, 4):
            raise ValueError("Expected a float32 (D,H,W) or (1,D,H,W) image.")
        if image.ndim == 4 and image.shape[0] != 1:
            raise ValueError("Expected one image channel.")
        if min(image.shape) < 1 or not np.isfinite(image).all():
            raise ValueError("Image must be nonempty and finite.")
        return None
    import torch  # Lazy: NumPy-only helpers remain usable without PyTorch.
    if (not torch.is_tensor(image) or image.dtype != torch.float32
            or image.ndim not in (3, 4) or (image.ndim == 4 and image.shape[0] != 1)):
        raise ValueError("Expected a float32 (D,H,W) or (1,D,H,W) image.")
    if min(image.shape) < 1 or not bool(torch.isfinite(image).all().item()):
        raise ValueError("Image must be nonempty and finite.")
    return torch


def augment_image(image, gamma=1.0, polarity=False):
    """Return a new image, preserving geometry and exact nonzero support.

    Signed power is followed by [-5,5] clipping; optional polarity multiplies
    the result by -1. Labels, spacing and geometry are never inputs.
    """
    if not np.isfinite(gamma) or gamma <= 0 or type(polarity) is not bool:
        raise ValueError("gamma must be finite/positive and polarity boolean.")
    torch = _image_backend(image)
    if torch is None:
        support = image != 0
        result = np.sign(image) * np.power(np.abs(image).astype(np.float64), gamma)
        result = np.clip(result, -5.0, 5.0).astype(np.float32)
        # Prevent float32 underflow from changing the declared nonzero support.
        tiny = np.nextafter(np.float32(0), np.float32(1))
        result = np.where(support & (result == 0), np.sign(image) * tiny, result)
        return np.where(support, -result if polarity else result, np.float32(0)).astype(np.float32)
    support = image != 0
    result = (image.sign() * image.abs().pow(gamma)).clamp(-5.0, 5.0)
    zero = torch.zeros((), dtype=image.dtype, device=image.device)
    tiny = torch.nextafter(zero, torch.ones_like(zero))
    result = torch.where(support & (result == 0), image.sign() * tiny, result)
    return torch.where(support, -result if polarity else result, zero)


def apply_stress(image, condition):
    """GT-free fixed stress; caller must enforce the validation-only scope.

    spatial_gain multiplies H positions by a linear 0.7..1.3 field. A singleton
    H receives gain 1.0. Each condition is applied separately to the clean image.
    """
    if condition not in STRESS_CONDITIONS:
        raise ValueError(f"Unknown synthetic stress condition: {condition}.")
    torch = _image_backend(image)
    if condition == "clean":
        return image.copy() if torch is None else image.clone()
    if condition.startswith("gamma_"):
        return augment_image(image, gamma=float(condition.split("_")[1]))
    if condition == "polarity":
        return augment_image(image, polarity=True)
    shape = [1] * image.ndim
    shape[-2] = image.shape[-2]
    if torch is None:
        gain = (np.linspace(0.7, 1.3, image.shape[-2], dtype=np.float32)
                if image.shape[-2] > 1 else np.ones(1, dtype=np.float32))
        scaled = (image.astype(np.float64) * gain.reshape(shape)).clip(-5.0, 5.0).astype(np.float32)
        tiny = np.nextafter(np.float32(0), np.float32(1))
        return np.where((image != 0) & (scaled == 0), np.sign(image) * tiny, scaled).astype(np.float32)
    gain = (torch.linspace(0.7, 1.3, image.shape[-2], dtype=image.dtype, device=image.device)
            if image.shape[-2] > 1 else torch.ones(1, dtype=image.dtype, device=image.device))
    scaled = (image * gain.reshape(shape)).clamp(-5.0, 5.0)
    zero = torch.zeros((), dtype=image.dtype, device=image.device)
    tiny = torch.nextafter(zero, torch.ones_like(zero))
    return torch.where((image != 0) & (scaled == 0), image.sign() * tiny, scaled)


class _DatasetView:
    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        # Keep records, manifest identity, grid and class metadata unchanged.
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


class AppearanceTrainingDataset(_DatasetView):
    """Original synchronized H/W flips followed by keyed image-only policy.

    Call set_epoch before each loader iteration. With worker processes, use
    fresh workers each epoch (no persistent_workers); the preregistered run uses
    workers=0. The independent appearance RNG does not consume original flips'
    Torch RNG or the training loader's shuffle generator.
    """
    def __init__(self, manifest_path, variant, seed=42, split="train"):
        if split != "train":
            raise ValueError("Appearance training augmentation is TRAIN-only.")
        deterministic_params(seed, 1, "parameter_validation", variant)
        from dataset import HipMRIVolumeDataset
        self.dataset = HipMRIVolumeDataset(manifest_path, split, augment=True)
        self.variant, self.seed, self.epoch = variant, seed, 1

    def set_epoch(self, epoch):
        _keyed_rng(self.seed, epoch, "parameter_validation", "gamma")
        self.epoch = epoch

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        params = deterministic_params(self.seed, self.epoch,
                                      f"{sample['patient']}/{sample['visit']}", self.variant)
        sample["image"] = augment_image(sample["image"], params["gamma"], params["polarity"])
        return sample


class StressValidationDataset(_DatasetView):
    """Explicit validation-only, unaugmented-base view; TEST is rejected."""
    def __init__(self, dataset, condition):
        if getattr(dataset, "split", None) != "validation" or getattr(dataset, "augment", None) is not False:
            raise ValueError("Synthetic stress requires an unaugmented validation dataset.")
        if condition not in STRESS_CONDITIONS:
            raise ValueError(f"Unknown synthetic stress condition: {condition}.")
        self.dataset, self.condition = dataset, condition

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["image"] = apply_stress(sample["image"], self.condition)
        return sample
