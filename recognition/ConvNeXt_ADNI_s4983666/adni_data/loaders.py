"""Seeded DataLoaders; only the training loader shuffles, samples or augments."""

import os
import random
import sys

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .slices import ADNISliceDataset
from .sampling import sampling_weights


def seed_worker(worker_id: int) -> None:
    """Seed Python's RNG in each worker from the loader's generator.

    On macOS with Python 3.12+, an inherited resource-tracker pipe can block
    interpreter shutdown, so the worker marks it close-on-exec
    (https://github.com/pytorch/pytorch/issues/153050).
    """
    if sys.platform == "darwin" and sys.version_info >= (3, 12):
        from multiprocessing.resource_tracker import getfd
        os.set_inheritable(getfd(), False)
    random.seed(torch.initial_seed() % 2**32)


def make_loader(rows, data_root, image_size, batch_size, workers, seed, shuffle, device, *,
                role="evaluation", augmentation=None, sampling="slice_uniform",
                preprocessing=None, scan_parameters=None, context_slices=1) -> DataLoader:
    """Build a loader that keeps every sample of its role (drop_last=False)."""
    sampler = None
    if sampling != "slice_uniform":
        if role != "train" or not shuffle:
            raise ValueError("Balanced sampling is permitted only for shuffled training.")
        weights, _ = sampling_weights(rows, sampling)
        sampler = WeightedRandomSampler(weights, len(rows), replacement=True,
                                        generator=torch.Generator().manual_seed(seed + 1000003))
        shuffle = False
    dataset = ADNISliceDataset(rows, data_root, image_size=image_size, role=role,
                               augmentation=augmentation, augmentation_seed=seed,
                               preprocessing=preprocessing, scan_parameters=scan_parameters,
                               context_slices=context_slices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, drop_last=False,
                      num_workers=workers, pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed))
