"""Build shuffled batches of slices with matching spatial shapes."""

from collections import defaultdict

import nibabel as nib
import torch
from torch.utils.data import Sampler


class ShapeBatchSampler(Sampler):
    """Include every sample once per epoch without resizing or padding."""

    def __init__(self, dataset, batch_size, seed=42):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        groups = defaultdict(list)

        for index, record in enumerate(dataset.samples):
            image_shape = nib.load(record.image_path).shape
            mask_shape = nib.load(record.mask_path).shape
            if image_shape != mask_shape:
                raise ValueError(f"Shape mismatch: {record.image_path.name}")
            if (
                len(image_shape) != 2
                or min(image_shape) < 16
                or any(size % 16 for size in image_shape)
            ):
                raise ValueError(f"Unsupported shape: {image_shape}")
            groups[image_shape].append(index)

        self.groups = dict(sorted(groups.items()))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.groups.values()
        )

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches = []

        for indices in self.groups.values():
            order = torch.randperm(
                len(indices), generator=generator
            ).tolist()
            shuffled = [indices[position] for position in order]
            batches.extend(
                shuffled[start:start + self.batch_size]
                for start in range(0, len(shuffled), self.batch_size)
            )

        batch_order = torch.randperm(
            len(batches), generator=generator
        ).tolist()
        for position in batch_order:
            yield batches[position]
