"""Batch native 2D NIfTI slices by shape without resizing or dropping samples."""
from collections import defaultdict
from numbers import Integral

import nibabel as nib
import torch
from torch.utils.data import Sampler


class ShapeBatchSampler(Sampler):
    """Group HipMRI2DDataset indices by image header shape once at startup.

    Shuffle both the indices within each shape and the order of batches when
    training. Keep the final, possibly smaller batch of every shape group.
    Dataset.__getitem__ remains responsible for image/mask value validation.
    """

    def __init__(self, dataset, batch_size, shuffle=False, generator=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.generator = generator
        groups = defaultdict(list)
        for index, (image_path, _) in enumerate(dataset.pairs):
            shape = tuple(nib.load(str(image_path)).shape)
            if len(shape) != 2 or any(size <= 0 for size in shape):
                raise ValueError(f'Expected a non-empty 2D image, got {shape}: {image_path}')
            groups[shape].append(index)
        if not groups:
            raise ValueError('No image pairs available for shape batching')
        self._groups = {shape: tuple(indices) for shape, indices in sorted(groups.items())}

    @property
    def shape_counts(self):
        return [{'shape': list(shape), 'samples': len(indices)}
                for shape, indices in self._groups.items()]

    def __len__(self):
        return sum((len(indices) + self.batch_size - 1) // self.batch_size
                   for indices in self._groups.values())

    def __iter__(self):
        batches = []
        for group in self._groups.values():
            indices = list(group)
            if self.shuffle:
                order = torch.randperm(len(indices), generator=self.generator).tolist()
                indices = [indices[position] for position in order]
            batches.extend(indices[start:start + self.batch_size]
                           for start in range(0, len(indices), self.batch_size))
        if self.shuffle:
            order = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[position] for position in order]
        yield from batches
