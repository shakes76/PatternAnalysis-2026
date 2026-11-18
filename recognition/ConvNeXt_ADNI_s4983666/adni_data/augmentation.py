"""Training-only augmentation for grayscale MRI slices.

Profiles:

* ``none``   - identity (used for every evaluation role).
* ``light``  - small rotation and translation; the original profile.
* ``strong`` - horizontal flip, random-resized crop, rotation, translation and a
  mild intensity change. It targets the memorisation seen when the scratch
  ConvNeXt reached ~100% training accuracy within a few epochs.
* ``scanner`` - ``strong`` plus acquisition-style perturbations: a smooth
  intensity inhomogeneity (bias field), Gaussian noise inside the head, random
  blur or sharpening, and wider gain/gamma. In this data the scanner, sequence
  and acquisition period are entangled with the diagnosis
  (docs/ACQUISITION_CONFOUND.md); randomising contrast, texture and noise is
  meant to make the acquisition style a less reliable cue than the anatomy.

Why each transform keeps the AD/NC label valid:

* Horizontal flip: the slices are sagittal, so this mirrors the head
  front-to-back. The atrophy patterns that separate AD from NC (enlarged
  ventricles, widened sulci) are still present in the mirrored slice, but the
  orientation itself never occurs in this data; the flip acts as a generic
  regulariser rather than a realistic variation.
* Crop, rotation and translation: head size, position and tilt vary between
  patients and scanners; the diagnosis does not depend on them.
* Intensity gain and gamma: brightness and contrast vary between acquisitions.
  Both map black to black, so the background stays zero.
* Bias field, noise, blur and sharpening: receive-coil inhomogeneity, noise
  level and effective resolution differ between scanners and sequences, not
  between diagnoses. The field only scales intensities down and noise is added
  inside the head only, so the background stays zero.

Vertical flips and large rotations are not used because they produce views that
never occur in this dataset.

When neighbouring slices are stacked as the bands of one PIL image, every band
receives exactly the same random transform, so the context stays aligned.

References: Pillow image API
https://pillow.readthedocs.io/en/stable/reference/Image.html ; random-resized
cropping follows Szegedy et al., "Going Deeper with Convolutions", CVPR 2015;
bias-field and noise augmentation as in TorchIO (Perez-Garcia et al., Comput.
Methods Programs Biomed. 2021, https://torchio.readthedocs.io/).
"""

from dataclasses import asdict, dataclass
import math
import random

from PIL import Image, ImageChops, ImageFilter
import torch


AUGMENTATION_NAMES = ("none", "light", "strong", "scanner")


@dataclass(frozen=True)
class AugmentationConfig:
    """Ranges of the random training transforms; a zero range disables one.

    ``intensity_gain=g`` samples a gain from [1 - g, 1 + g] and
    ``gamma_range=r`` samples a gamma from [1 - r, 1 + r]. ``bias_field=b``
    multiplies the image by a smooth random field in [1 - b, 1]; ``noise_std``
    is the largest Gaussian noise standard deviation as a fraction of 255;
    ``blur_sigma`` is the largest Gaussian blur radius in pixels, and
    ``sharpen_probability`` replaces the blur by an unsharp mask.
    """

    name: str = "none"
    flip_probability: float = 0.0
    min_crop_scale: float = 1.0
    rotation_degrees: float = 0.0
    translation_fraction: float = 0.0
    intensity_gain: float = 0.0
    gamma_range: float = 0.0
    bias_field: float = 0.0
    noise_std: float = 0.0
    blur_sigma: float = 0.0
    sharpen_probability: float = 0.0

    def __post_init__(self) -> None:
        """Reject unknown profiles and ranges that would distort the anatomy."""
        if self.name not in AUGMENTATION_NAMES:
            raise ValueError(f"Unsupported augmentation profile: {self.name!r}")
        limits = {"flip_probability": (0.0, 1.0), "min_crop_scale": (0.5, 1.0),
                  "rotation_degrees": (0.0, 15.0), "translation_fraction": (0.0, 0.1),
                  "intensity_gain": (0.0, 0.3), "gamma_range": (0.0, 0.3), "bias_field": (0.0, 0.4),
                  "noise_std": (0.0, 0.05), "blur_sigma": (0.0, 1.5), "sharpen_probability": (0.0, 1.0)}
        for field, (low, high) in limits.items():
            value = getattr(self, field)
            if not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError(f"{field} must be between {low} and {high}.")

    def to_dict(self) -> dict:
        """Serialize every range so a checkpoint records the exact transform."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "AugmentationConfig":
        """Rebuild a configuration saved by ``to_dict``."""
        return cls(**value)


PROFILES = {
    "none": AugmentationConfig(),
    "light": AugmentationConfig("light", rotation_degrees=5.0, translation_fraction=0.03),
    "strong": AugmentationConfig("strong", flip_probability=0.5, min_crop_scale=0.85,
                                 rotation_degrees=10.0, translation_fraction=0.05,
                                 intensity_gain=0.1, gamma_range=0.15),
    "scanner": AugmentationConfig("scanner", flip_probability=0.5, min_crop_scale=0.85,
                                  rotation_degrees=10.0, translation_fraction=0.05,
                                  intensity_gain=0.2, gamma_range=0.3, bias_field=0.3,
                                  noise_std=0.03, blur_sigma=1.0, sharpen_probability=0.3),
}


def make_augmentation(name: str = "none") -> AugmentationConfig:
    """Return the fixed configuration of a named profile."""
    if name not in PROFILES:
        raise ValueError(f"Unsupported augmentation profile: {name!r}")
    return PROFILES[name]


def _black(image: Image.Image) -> int | tuple[int, ...]:
    """Fill value for pixels exposed by rotation or translation."""
    bands = len(image.getbands())
    return 0 if bands == 1 else (0,) * bands


def _random_resized_crop(image: Image.Image, min_scale: float, rng: random.Random) -> Image.Image:
    """Crop a random region covering [min_scale, 1] of the area and resize it back."""
    width, height = image.size
    side = math.sqrt(rng.uniform(min_scale, 1.0))  # Same factor on both sides keeps the aspect ratio.
    crop_width, crop_height = max(1, round(width * side)), max(1, round(height * side))
    left = rng.randint(0, width - crop_width)
    top = rng.randint(0, height - crop_height)
    box = (left, top, left + crop_width, top + crop_height)
    return image.resize((width, height), Image.Resampling.BILINEAR, box=box)


def _bias_field(image: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    """Multiply by a smooth random field in [1 - strength, 1] (same field for every band)."""
    coarse = Image.new("L", (4, 4))
    coarse.putdata([round(255 * (1 - rng.uniform(0, strength))) for _ in range(16)])
    field = coarse.resize(image.size, Image.Resampling.BICUBIC)
    if image.mode != "L":
        field = Image.merge(image.mode, [field] * len(image.getbands()))
    return ImageChops.multiply(image, field)


def _head_noise(image: Image.Image, std: float, rng: random.Random) -> Image.Image:
    """Add zero-mean Gaussian noise inside the head; the black background is kept."""
    generator = torch.Generator().manual_seed(rng.getrandbits(63))
    bands = len(image.getbands())
    noise = torch.randn((image.height, image.width, bands), generator=generator) * std * 255 + 128
    noise = noise.round().clamp(0, 255).to(torch.uint8)
    noise_image = Image.frombytes(image.mode, image.size, noise.numpy().tobytes())
    noisy = ImageChops.add(image, noise_image, scale=1.0, offset=-128)
    head = image.convert("L").point(lambda value: 255 if value > 0 else 0)
    return Image.composite(noisy, image, head)


def augment_image(image: Image.Image, config: AugmentationConfig, rng: random.Random) -> Image.Image:
    """Apply one random draw of every enabled transform; the input is not modified."""
    if config.name == "none":
        return image
    if rng.random() < config.flip_probability:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if config.min_crop_scale < 1.0:
        image = _random_resized_crop(image, config.min_crop_scale, rng)
    if config.rotation_degrees or config.translation_fraction:
        angle = rng.uniform(-config.rotation_degrees, config.rotation_degrees)
        shift_x = rng.uniform(-config.translation_fraction, config.translation_fraction) * image.width
        shift_y = rng.uniform(-config.translation_fraction, config.translation_fraction) * image.height
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False,
                             translate=(shift_x, shift_y), fillcolor=_black(image))
    if config.intensity_gain or config.gamma_range:
        gain = rng.uniform(1 - config.intensity_gain, 1 + config.intensity_gain)
        gamma = rng.uniform(1 - config.gamma_range, 1 + config.gamma_range)
        table = [min(255, round(255 * gain * (value / 255) ** gamma)) for value in range(256)]
        image = image.point(table * len(image.getbands()))  # One table per band.
    if config.bias_field:
        image = _bias_field(image, config.bias_field, rng)
    if config.sharpen_probability and rng.random() < config.sharpen_probability:
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=rng.randint(50, 150), threshold=0))
    elif config.blur_sigma:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0, config.blur_sigma)))
    if config.noise_std:
        image = _head_noise(image, rng.uniform(0, config.noise_std), rng)
    return image
