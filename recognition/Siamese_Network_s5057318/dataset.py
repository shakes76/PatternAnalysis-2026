"""Patient-level train/val/test split of the cleaned ISIC metadata."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 42
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_SIZE = 224


def extract_real_id(value):
    """Return the first non-dummy ID in a possibly comma-separated field, else None."""
    if pd.isna(value):
        return None
    for part in str(value).split(","):
        part = part.strip().strip("\"'").strip()
        if part and not part.lower().startswith("dummy"):
            return part
    return None


def load_clean_metadata(data_dir):
    """Load the csv and keep only rows with a real (non-dummy) ID in `real_id`."""
    df = pd.read_csv(os.path.join(data_dir, "cleaned_metadata.csv"),
                     dtype={"patient_id": str})
    df["real_id"] = df["patient_id"].map(extract_real_id)
    return df.dropna(subset=["real_id"]).reset_index(drop=True)


def split_by_id(df, fractions=FRACTIONS, seed=SEED):
    """Split rows into train/val/test so every real ID lands in exactly one split.

    IDs (not images) are assigned greedily, most-malignant first, to whichever
    split is furthest below its target share of both images and malignant
    images. This keeps split sizes and malignant % close to each other.
    Returns a dict {split_name: DataFrame}.
    """
    # Patient-ID split rule: count images (n) and malignant images (mal) per real ID,
    # shuffle the IDs with the seed, then sort most-malignant first. The sort is stable,
    # so IDs with the same malignant count keep their shuffled order.
    rng = np.random.default_rng(seed)
    per_id = df.groupby("real_id")["target"].agg(n="size", mal="sum")
    per_id = per_id.loc[rng.permutation(per_id.index)]  # seeded tie-breaking
    per_id = per_id.sort_values("mal", ascending=False, kind="stable")

    total_n, total_mal = per_id["n"].sum(), per_id["mal"].sum()
    names = list(fractions)
    count = {s: 0.0 for s in names}
    mal = {s: 0.0 for s in names}
    assign = {}
    # Greedy assignment: each ID (with all its images) goes to the split with the largest
    # deficit = relative shortfall of images + relative shortfall of malignant images
    # against that split's target fraction.
    for real_id, n, m in zip(per_id.index, per_id["n"], per_id["mal"]):
        def deficit(s):
            """Relative shortfall of images plus relative shortfall of malignant images for split s."""
            f = fractions[s]
            return ((f * total_n - count[s]) / (f * total_n)
                    + (f * total_mal - mal[s]) / max(f * total_mal, 1))
        best = max(names, key=deficit)
        assign[real_id] = best
        count[best] += n
        mal[best] += m

    split_col = df["real_id"].map(assign)
    return {s: df[split_col == s].reset_index(drop=True) for s in names}


def check_no_id_overlap(splits):
    """Raise AssertionError if any real ID appears in more than one split."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(splits[a]["real_id"]) & set(splits[b]["real_id"])
            if shared:
                raise AssertionError(
                    f"{len(shared)} real IDs in both {a} and {b}, e.g. {sorted(shared)[:5]}")


class GrayWorld:
    """Colour constancy: scale each channel so its mean equals the mean of all three.

    Works on a PIL image and returns a PIL image (values clamped to 0-255).
    """

    def __call__(self, img):
        """Return a colour-corrected copy of the PIL image `img`."""
        x = transforms.functional.pil_to_tensor(img).float()
        channel_mean = x.mean(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        x = (x * (x.mean() / channel_mean)).round().clamp(0, 255).to(torch.uint8)
        return transforms.functional.to_pil_image(x)


def _preprocess_steps(preprocess):
    """Transform steps for the `preprocess` option: [] for "none", [GrayWorld()] for "grayworld"."""
    if preprocess == "none":
        return []
    if preprocess == "grayworld":
        return [GrayWorld()]
    raise ValueError(f"unknown preprocess {preprocess!r}")


def compute_mean_std(df, image_dir, preprocess="none"):
    """Per-channel mean/std over the images in `df` (pass the train split only).

    The statistics are taken after `preprocess`, so they match what the model sees.
    """
    to_tensor = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                    *_preprocess_steps(preprocess), transforms.ToTensor()])
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    for isic_id in df["isic_id"]:
        img = to_tensor(_load_rgb(image_dir, isic_id)).double()
        total += img.sum(dim=(1, 2))
        total_sq += (img ** 2).sum(dim=(1, 2))
    n_pixels = len(df) * IMAGE_SIZE * IMAGE_SIZE
    mean = total / n_pixels
    std = (total_sq / n_pixels - mean ** 2).sqrt()
    return mean.float().tolist(), std.float().tolist()


def _load_rgb(image_dir, isic_id):
    """Open <image_dir>/<isic_id>.jpg as an RGB PIL image."""
    return Image.open(os.path.join(image_dir, f"{isic_id}.jpg")).convert("RGB")


def build_transform(mean, std, train, augment="basic", preprocess="none"):
    """Augmentation is applied for the train split only.

    "basic": flips and small rotation. "strong": basic plus colour jitter and a
    small random affine (translation, scale). `preprocess` ("none" or "grayworld")
    is applied to every split, right after the resize and before augmentation.
    """
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), *_preprocess_steps(preprocess)]
    if train:
        steps += [transforms.RandomHorizontalFlip(),
                  transforms.RandomVerticalFlip(),
                  transforms.RandomRotation(15)]
        if augment == "strong":
            steps += [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                      transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1))]
        elif augment != "basic":
            raise ValueError(f"unknown augment {augment!r}")
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


class ISICDataset(Dataset):
    """One split of the cleaned metadata; images are read from <image_dir>/<isic_id>.jpg.

    Items are (image tensor of shape 3x224x224, float label). The split's
    DataFrame (with `real_id`) is kept in `self.df`.
    """

    def __init__(self, split_df, image_dir, transform):
        self.df = split_df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.transform(_load_rgb(self.image_dir, row["isic_id"]))
        return image, torch.tensor(row["target"], dtype=torch.float32)


def get_dataloaders(splits, image_dir, batch_size=32, num_workers=0, seed=SEED, augment="basic",
                    preprocess="none"):
    """Return (train_loader, val_loader, test_loader) from the dict of split_by_id.

    Normalisation statistics come from the train split only; only the train
    loader augments and shuffles.
    """
    mean, std = compute_mean_std(splits["train"], image_dir, preprocess)
    generator = torch.Generator().manual_seed(seed)
    loaders = []
    for name in ("train", "val", "test"):
        train = name == "train"
        dataset = ISICDataset(splits[name], image_dir, build_transform(mean, std, train, augment, preprocess))
        dataset.mean, dataset.std = mean, std
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=train,
                                  num_workers=num_workers,
                                  generator=generator if train else None))
    return tuple(loaders)


class PairDataset(Dataset):
    """Image pairs (img1, img2, label) built from ONE split only.

    Label convention (matches ContrastiveLoss): 0 = similar pair (same class),
    1 = dissimilar pair (different class). Pair kind is set by `idx % 4`:
    0 = benign-benign, 1 = malignant-malignant, 2 and 3 = dissimilar
    (one benign, one malignant). So pairs are exactly 50% similar / 50%
    dissimilar, and malignant images are deliberately oversampled. An image is
    never paired with itself.

    With `fixed_seed=None` (train) pairs are drawn fresh on every access, so
    they change every epoch. With an int `fixed_seed` (val/test) all pairs are
    drawn once at construction and are identical on every run.

    With `pair_source="same"` both images of a pair come from one source (the
    IP_ / HAM_ prefix of `real_id`): the first image (the malignant one for
    malignant-malignant and dissimilar pairs) is drawn as with "any", then the
    second is drawn from that image's source only.
    """

    def __init__(self, split_df, image_dir, transform, num_pairs=None, fixed_seed=None,
                 pair_source="any"):
        self.df = split_df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        targets = self.df["target"].to_numpy()
        self.benign = torch.from_numpy(np.flatnonzero(targets == 0))
        self.malignant = torch.from_numpy(np.flatnonzero(targets == 1))
        if len(self.benign) < 2 or len(self.malignant) < 2:
            raise ValueError("each class needs at least 2 images to form same-class pairs")
        if pair_source not in ("any", "same"):
            raise ValueError(f"unknown pair_source {pair_source!r}")
        self.pair_source = pair_source
        if pair_source == "same":
            self.source = self.df["real_id"].str.split("_").str[0].to_numpy()
            self.by_source = {}  # (source, class) -> row indices
            for src in np.unique(self.source):
                for cls in (0, 1):
                    rows = np.flatnonzero((self.source == src) & (targets == cls))
                    self.by_source[src, cls] = torch.from_numpy(rows)
                n_ben, n_mal = len(self.by_source[src, 0]), len(self.by_source[src, 1])
                if n_ben == 1 or n_mal == 1 or (n_mal and not n_ben):
                    raise ValueError(f"source {src}: {n_ben} benign / {n_mal} malignant images "
                                     "is too few for same-source pairs")
        # Default length: one pair per image in the split, rounded down to a multiple of 4.
        n = num_pairs if num_pairs is not None else len(self.df)
        self.num_pairs = n - n % 4  # whole number of (bb, mm, dis, dis) groups
        self.fixed = fixed_seed is not None
        # Val/test: every pair is drawn once here from a seeded generator. Train: pairs are
        # drawn on each access from the global torch RNG (see draw()), so they change every epoch.
        if self.fixed:
            gen = torch.Generator().manual_seed(fixed_seed)
            self._pairs = [self._draw(k, gen) for k in range(self.num_pairs)]

    @staticmethod
    def _randint(n, gen):
        """Random int in [0, n) from `gen` (the global torch RNG when `gen` is None)."""
        return int(torch.randint(n, (1,), generator=gen))

    def _same_class(self, pool, gen):
        """Two different row indices drawn uniformly from `pool`."""
        i = self._randint(len(pool), gen)
        j = self._randint(len(pool) - 1, gen)
        if j >= i:  # skip i itself, so the two images are never the same
            j += 1
        return int(pool[i]), int(pool[j])

    def _from_source(self, first, cls, gen):
        """A row of class `cls` from the same source as row `first`, never `first` itself."""
        pool = self.by_source[self.source[first], cls]
        if int(self.df.at[first, "target"]) != cls:
            return int(pool[self._randint(len(pool), gen)])
        pos = int((pool == first).nonzero())
        j = self._randint(len(pool) - 1, gen)
        return int(pool[j + 1 if j >= pos else j])  # skip `first` itself

    def _draw_same_source(self, kind, gen):
        """Pair of kind `kind` (0 bb, 1 mm, 2/3 dissimilar) with both images from one source."""
        if kind in (0, 1):
            pool = self.benign if kind == 0 else self.malignant
            first = int(pool[self._randint(len(pool), gen)])
            return first, self._from_source(first, kind, gen), 0
        b = int(self.malignant[self._randint(len(self.malignant), gen)])
        # Dissimilar: the malignant image b was drawn first; the benign image a comes from b's source.
        a = self._from_source(b, 0, gen)
        if self._randint(2, gen):  # random order within the dissimilar pair
            a, b = b, a
        return a, b, 1

    def _draw(self, idx, gen):
        """Draw pair number `idx` as (row_1, row_2, label); `idx % 4` sets the pair kind."""
        kind = idx % 4
        if self.pair_source == "same":
            return self._draw_same_source(kind, gen)
        # pair_source "any": kind 0 = two different benign images, kind 1 = two different
        # malignant images (label 0, similar); kinds 2 and 3 = one random benign and one random
        # malignant image in random order (label 1, dissimilar).
        if kind == 0:
            return (*self._same_class(self.benign, gen), 0)
        if kind == 1:
            return (*self._same_class(self.malignant, gen), 0)
        a = int(self.benign[self._randint(len(self.benign), gen)])
        b = int(self.malignant[self._randint(len(self.malignant), gen)])
        if self._randint(2, gen):  # random order within the dissimilar pair
            a, b = b, a
        return a, b, 1

    def draw(self, idx):
        """Return (row_index_1, row_index_2, label) for pair `idx`, without loading images."""
        return self._pairs[idx] if self.fixed else self._draw(idx, None)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        i, j, label = self.draw(idx)
        img1 = self.transform(_load_rgb(self.image_dir, self.df.at[i, "isic_id"]))
        img2 = self.transform(_load_rgb(self.image_dir, self.df.at[j, "isic_id"]))
        return img1, img2, torch.tensor(label, dtype=torch.float32)


def get_pair_dataloaders(splits, image_dir, batch_size=32, num_workers=0,
                         seed=SEED, num_pairs=None, split_seed=None, augment="basic",
                         preprocess="none", pair_source="any"):
    """Return (train_loader, val_loader, test_loader) of image pairs.

    Same normalisation (train mean/std) and augmentation policy as
    get_dataloaders: only train augments. Train pairs are re-drawn every epoch
    and shuffled; val/test pairs are drawn once from fixed seeds
    (split_seed + 1 and split_seed + 2; split_seed defaults to seed).
    `pair_source` ("any" or "same") applies to the train pairs only.
    """
    mean, std = compute_mean_std(splits["train"], image_dir, preprocess)
    generator = torch.Generator().manual_seed(seed)
    split_seed = seed if split_seed is None else split_seed
    # train pairs are re-drawn each epoch (no fixed seed); val and test pairs use fixed seeds so they are identical on every run
    fixed_seeds = {"train": None, "val": split_seed + 1, "test": split_seed + 2}
    loaders = []
    for name in ("train", "val", "test"):
        train = name == "train"
        # pair_source only applies to the train pairs; val and test pairs always use "any"
        dataset = PairDataset(splits[name], image_dir, build_transform(mean, std, train, augment, preprocess),
                              num_pairs=num_pairs, fixed_seed=fixed_seeds[name],
                              pair_source=pair_source if train else "any")
        # keep the train mean/std on the dataset so callers can un-normalise images for display
        dataset.mean, dataset.std = mean, std
        # shuffle only the train pairs; the seeded generator makes that shuffle reproducible
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=train,
                                  num_workers=num_workers,
                                  generator=generator if train else None))
    return tuple(loaders)


def main():
    """Print images, IDs and malignant count per split and run the ID-overlap check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder containing cleaned_metadata.csv")
    args = parser.parse_args()

    df = load_clean_metadata(args.data_dir)
    splits = split_by_id(df)
    check_no_id_overlap(splits)

    print(f"{'split':<6}{'images':>8}{'IDs':>7}{'malignant':>11}{'malignant %':>13}")
    for name, part in splits.items():
        mal = int((part["target"] == 1).sum())
        print(f"{name:<6}{len(part):>8}{part['real_id'].nunique():>7}"
              f"{mal:>11}{100 * mal / len(part):>12.2f}%")
    assert sum(len(p) for p in splits.values()) == len(df), "rows lost in split"
    print("Overlap check: PASSED (no real ID appears in more than one split)")


if __name__ == "__main__":
    main()
