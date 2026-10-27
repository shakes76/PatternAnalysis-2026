"""TRAIN-only recovery policies; frozen A1 augmentation remains unchanged."""
from __future__ import annotations

from experiments.appearance import _DatasetView, _keyed_rng, augment_image, deterministic_params

VARIANTS = ("clean_adapt", "polarity_rehearsal")
POLARITY_PROBABILITIES = {"clean_adapt": 0.0, "polarity_rehearsal": 0.1}


def recovery_params(seed, epoch, case_key, variant):
    """Retain A1 gamma stream; change only the independent polarity threshold."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown recovery variant: {variant}.")
    values = deterministic_params(seed, epoch, case_key, "gamma_only")
    values["polarity"] = bool(_keyed_rng(seed, epoch, case_key, "polarity").random()
                              < POLARITY_PROBABILITIES[variant])
    return values


class RecoveryTrainingDataset(_DatasetView):
    """Original synchronized flips plus fixed gamma and recovery polarity.

    Workers must be zero for observable exposure counts. Counts describe input
    examples, including any example whose optimizer update was AMP-skipped.
    No geometry, mask, spacing or input-zero support is changed by appearance.
    """
    def __init__(self, manifest_path, variant, seed=42, split="train"):
        if split != "train":
            raise ValueError("Recovery augmentation is TRAIN-only.")
        recovery_params(seed, 1, "parameter_validation", variant)
        from dataset import HipMRIVolumeDataset
        self.dataset = HipMRIVolumeDataset(manifest_path, "train", augment=True)
        self.variant, self.seed, self.epoch = variant, seed, 1
        self.reset_counts()

    def set_epoch(self, epoch):
        _keyed_rng(self.seed, epoch, "parameter_validation", "gamma")
        self.epoch = epoch

    def reset_counts(self):
        self.policy_counts = {"noninverted": 0, "inverted": 0, "gamma_applied": 0}

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        params = recovery_params(self.seed, self.epoch, f"{sample['patient']}/{sample['visit']}", self.variant)
        sample["image"] = augment_image(sample["image"], params["gamma"], params["polarity"])
        self.policy_counts["inverted" if params["polarity"] else "noninverted"] += 1
        self.policy_counts["gamma_applied"] += int(params["gamma_applied"])
        return sample
