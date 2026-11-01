"""Check cache equivalence, invalidation and batched full-volume inference."""

import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from dataset import PatchDataset, prepare_cache, cache_path, clear_volume_cache
from modules import ImprovedUNet3D
from predict import sliding_window_predict


def main():
    """Check that caching and window batching preserve the sampled data and labels."""
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rng = np.random.default_rng(3)
        for shape in [(12, 19, 21), (19, 21, 23)]:
            image = rng.normal(size=shape).astype(np.float32)
            mask = rng.integers(0, 6, size=shape, dtype=np.int16)
            pair = (root / "image.nii.gz", root / "mask.nii.gz")
            for data, path in zip((image, mask), pair):
                nib.save(nib.Nifti1Image(data, np.eye(4)), path)
            first_key = cache_path(pair, 6, root / "cache")
            prepare_cache([pair], 6, root / "cache")
            for foreground in (0.0, 1.0):
                plain = PatchDataset([pair], (16,) * 3, foreground_probability=foreground)
                cached = PatchDataset(
                    [pair], (16,) * 3, foreground_probability=foreground, cache_dir=root / "cache"
                )
                # Replaying identical RNG seeds exposes any cache-induced sampling drift.
                for seed in range(10):
                    np.random.seed(seed)
                    expected = plain[0]
                    np.random.seed(seed)
                    actual = cached[0]
                    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
            # Editing the source label creates a new cache identity.
            mask.flat[0] = (int(mask.flat[0]) + 1) % 6
            nib.save(nib.Nifti1Image(mask, np.eye(4)), pair[1])
            assert first_key != cache_path(pair, 6, root / "cache")
            for architecture in ("baseline", "improved"):
                torch.manual_seed(5)
                model = ImprovedUNet3D(6, 4, architecture)
                single = sliding_window_predict(model, image, (16,) * 3, 6, torch.device("cpu"))
                batched = sliding_window_predict(
                    model, image, (16,) * 3, 6, torch.device("cpu"), window_batch_size=3
                )
                assert np.array_equal(single, batched)
            prepare_cache([pair], 6, root / "cache")
            loader = DataLoader(cached, batch_size=1, num_workers=2)
            assert sum(int(x.shape[0]) for x, _ in loader) == len(cached)
        clear_volume_cache()
    print("Cache sampling/padding/invalidation and batched inference equivalence passed")


if __name__ == "__main__":
    main()
