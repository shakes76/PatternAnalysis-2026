"""Check PairDataset: label balance, pair kinds, self-pairs, val determinism; save a pair grid."""
import argparse
import hashlib
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import dataset as d

HERE = os.path.dirname(os.path.abspath(__file__))
N_PAIRS = 2000
KINDS = {(0, 0): "benign-benign", (1, 1): "malignant-malignant"}


def kind_of(ds, i, j):
    """Pair kind of rows i and j from their targets: benign-benign, malignant-malignant or dissimilar."""
    return KINDS.get((int(ds.df.at[i, "target"]), int(ds.df.at[j, "target"])), "dissimilar")


def main():
    """Check 2000 pairs per split, check val/test reproducibility and save a grid of 6 val pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder containing cleaned_metadata.csv")
    parser.add_argument("--image_dir", help="default: <data_dir>/combined_images")
    parser.add_argument("--out", default=os.path.join(HERE, "outputs", "pairs.png"))
    args = parser.parse_args()
    image_dir = args.image_dir or os.path.join(args.data_dir, "combined_images")

    splits = d.split_by_id(d.load_clean_metadata(args.data_dir))
    mean, std = d.compute_mean_std(splits["train"], image_dir)
    seeds = {"train": None, "val": d.SEED + 1, "test": d.SEED + 2}
    datasets = {
        name: d.PairDataset(splits[name], image_dir,
                            d.build_transform(mean, std, name == "train"),
                            num_pairs=N_PAIRS, fixed_seed=seeds[name])
        for name in splits
    }

    for name, ds in datasets.items():
        pairs = [ds.draw(k) for k in range(len(ds))]
        kinds = [kind_of(ds, i, j) for i, j, _ in pairs]
        labels = [label for _, _, label in pairs]
        same_file = sum(ds.df.at[i, "isic_id"] == ds.df.at[j, "isic_id"] for i, j, _ in pairs)
        label_ok = all((label == 1) == (kind == "dissimilar")
                       for (_, _, label), kind in zip(pairs, kinds))
        print(f"[{name}] {len(pairs)} pairs from {len(ds.df)} images")
        print(f"  labels: similar(0)={labels.count(0)}  dissimilar(1)={labels.count(1)}")
        print(f"  benign-benign={kinds.count('benign-benign')}  "
              f"malignant-malignant={kinds.count('malignant-malignant')}  "
              f"dissimilar={kinds.count('dissimilar')}")
        print(f"  pairs with both images the same file: {same_file}")
        print(f"  labels consistent with pair kinds: {label_ok}")
        assert same_file == 0 and label_ok

    # Train pairs are re-drawn on every access; val/test must be reproducible.
    train = datasets["train"]
    first = [train.draw(k) for k in range(len(train))]
    second = [train.draw(k) for k in range(len(train))]
    print(f"\ntrain pairs differ between two draws (re-drawn each epoch): {first != second}")
    assert first != second
    for name in ("val", "test"):
        again = d.PairDataset(splits[name], image_dir, datasets[name].transform,
                              num_pairs=N_PAIRS, fixed_seed=seeds[name])
        same = again._pairs == datasets[name]._pairs
        digest = hashlib.sha1(repr(datasets[name]._pairs).encode()).hexdigest()[:12]
        print(f"{name} pairs identical across two builds: {same}   sha1={digest}")
        assert same

    # Grid of 6 val pairs (no augmentation): image 1 on top, image 2 below.
    val = datasets["val"]
    m, s = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    for col in range(6):
        img1, img2, label = val[col]
        i, j, _ = val.draw(col)
        axes[0, col].set_title(f"label {int(label)}\n{kind_of(val, i, j)}", fontsize=8)
        for row, img in enumerate((img1, img2)):
            axes[row, col].imshow((img * s + m).clamp(0, 1).permute(1, 2, 0).numpy())
            axes[row, col].axis("off")
    fig.suptitle("Val pairs: label 0 = similar, 1 = dissimilar")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
