"""Course-facing data interface: patient-level splits, transforms and loaders.

Training, prediction and the analysis scripts import their data loaders from
this module. The implementation lives in the ``adni_data/`` package; see
``adni_data/manifests.py`` for the leakage-free role definitions.
"""

from adni_data.augmentation import AugmentationConfig, make_augmentation
from adni_data.loaders import make_loader
from adni_data.manifests import load_fold, load_holdout
from adni_data.slices import ADNISliceDataset

__all__ = ["ADNISliceDataset", "AugmentationConfig", "make_augmentation", "make_loader",
           "load_fold", "load_holdout"]
