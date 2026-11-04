"""PyTorch dataset of ADNI JPEG slices from the frozen patient manifests.

Each item is one slice, optionally stacked with its two neighbouring slices
from the same scan (``context_slices=3``). A 2D model then sees a little of
the through-plane anatomy while every scan still yields one prediction per
slice. At the first and last slice of a scan the missing neighbour is replaced
by the centre slice. Neighbours always come from the same scan, so context
never mixes patients or roles.

Pixel values are mapped from [0, 255] to [-1, 1] with fixed constants, so no
statistic is estimated from any data split.
"""

from pathlib import Path
import random

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, get_worker_info

from .augmentation import AugmentationConfig, augment_image
from .preprocessing import PreprocessingConfig, apply_preprocessing, prepare_scans, validate_scan_parameters


ROLES = ("train", "val", "calibration", "test", "evaluation")


class ADNISliceDataset(Dataset):
    """Return ``{"image", "label", "patient_id", "image_id", "slice_index", "relative_path"}``.

    ``image`` has shape [context_slices, height, width]. Augmentation is only
    allowed for the ``train`` role.
    """

    def __init__(self, rows: list[dict], data_root: Path, image_size: tuple[int, int] = (240, 256), *,
                 role: str = "evaluation", augmentation: AugmentationConfig | None = None,
                 augmentation_seed: int = 0, preprocessing: PreprocessingConfig | None = None,
                 scan_parameters: dict | None = None, context_slices: int = 1) -> None:
        if not rows:
            raise ValueError("A slice dataset cannot be empty.")
        if role not in ROLES:
            raise ValueError(f"Unsupported dataset role: {role!r}")
        if context_slices not in (1, 3):
            raise ValueError("context_slices must be 1 or 3.")
        self.rows = [dict(row) for row in rows]
        self.data_root = Path(data_root).resolve()
        self.image_size = tuple(image_size)
        self.role = role
        self.context_slices = context_slices
        self.augmentation = augmentation or AugmentationConfig()
        if self.augmentation.name != "none" and role != "train":
            raise ValueError("Augmentation is permitted only for the train role.")
        self.augmentation_seed = augmentation_seed
        self._rng = None
        self._rng_owner = None

        self.paths = []
        for row in self.rows:
            relative = Path(row["relative_path"])
            path = (self.data_root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(self.data_root) or not path.is_file():
                raise ValueError(f"Slice path is missing or outside the data directory: {relative}")
            if str(row["label"]) not in ("0", "1"):
                raise ValueError(f"Labels must be NC=0 or AD=1: {relative}")
            self.paths.append(path)
        # (scan, slice index) -> row position, used to find neighbouring slices.
        self.position = {(row["image_id"], int(row["slice_index"])): index
                         for index, row in enumerate(self.rows)}

        self.preprocessing = preprocessing or PreprocessingConfig()
        self.scan_parameters = {}
        if self.preprocessing.name != "none":
            self.scan_parameters = scan_parameters or prepare_scans(self.rows, self.data_root, self.preprocessing)
            validate_scan_parameters(self.rows, self.scan_parameters, self.preprocessing)

    def __len__(self) -> int:
        return len(self.rows)

    def _random(self) -> random.Random:
        """One augmentation RNG per DataLoader worker, seeded reproducibly."""
        worker = get_worker_info()
        owner = None if worker is None else worker.id
        if self._rng is None or self._rng_owner != owner:
            self._rng = random.Random(self.augmentation_seed if worker is None else worker.seed)
            self._rng_owner = owner
        return self._rng

    def _load_slice(self, index: int) -> Image.Image:
        """Read one grayscale slice and apply the deterministic preprocessing/resize."""
        row = self.rows[index]
        with Image.open(self.paths[index]) as source:
            image = ImageOps.exif_transpose(source).convert("L")
        if self.preprocessing.name != "none":
            return apply_preprocessing(image, self.scan_parameters[row["image_id"]],
                                       self.preprocessing, self.image_size)
        target = (self.image_size[1], self.image_size[0])
        return image if image.size == target else image.resize(target, Image.Resampling.BILINEAR)

    def _neighbour(self, index: int, offset: int) -> int:
        """Index of the slice ``offset`` positions away in the same scan, or the slice itself."""
        row = self.rows[index]
        return self.position.get((row["image_id"], int(row["slice_index"]) + offset), index)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        if self.context_slices == 1:
            image = self._load_slice(index)
        else:
            bands = [self._load_slice(self._neighbour(index, offset)) for offset in (-1, 0, 1)]
            image = Image.merge("RGB", bands)  # Bands = previous, centre, next slice.
        if self.augmentation.name != "none":
            image = augment_image(image, self.augmentation, self._random())
        height, width = self.image_size
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        # PIL stores bands interleaved (height, width, bands); convert to channels-first.
        tensor = pixels.reshape(height, width, self.context_slices).permute(2, 0, 1).float()
        tensor = tensor.div(255.0).sub(0.5).div(0.5)
        return {
            "image": tensor.contiguous(),
            "label": torch.tensor(float(row["label"])),
            "patient_id": row["patient_id"],
            "image_id": row["image_id"],
            "slice_index": int(row["slice_index"]),
            "relative_path": row["relative_path"],
        }
