"""Checks that --seed and --split_seed behave as documented (usage: check_seeds.py DATA_DIR)."""
import argparse
import os

import torch

import dataset as d

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("data_dir", help="folder with cleaned_metadata.csv and combined_images/")
args = parser.parse_args()

failures = 0


def check(name, got, expected):
    """Print PASS if `got == expected`, else FAIL, and count failures."""
    global failures
    ok = got == expected
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name} -> {got} (expected {expected})")


def summary(splits):
    """([images per split], [malignant per split]) in train, val, test order."""
    names = ("train", "val", "test")
    return ([len(splits[n]) for n in names], [int(splits[n]["target"].sum()) for n in names])


def id_sets(splits):
    """The set of isic_id values in each split."""
    return {n: set(splits[n]["isic_id"]) for n in ("train", "val", "test")}


def resolve(seed, split_seed):
    """Same rule as train.py / predict.py: split_seed defaults to seed."""
    return seed if split_seed is None else split_seed


df = d.load_clean_metadata(args.data_dir)
default = d.split_by_id(df, seed=resolve(42, None))
imgs, mal = summary(default)
check("a) default split images (train/val/test)", imgs, [9881, 2029, 2151])
check("a) default split malignant (train/val/test)", mal, [833, 186, 176])

decoupled = d.split_by_id(df, seed=resolve(7, 42))
check("b) seed=7, split_seed=42 gives the same isic_id sets as the default",
      id_sets(decoupled) == id_sets(default), True)

seven = d.split_by_id(df, seed=resolve(7, None))
imgs, mal = summary(seven)
check("c) seed 7 as split seed: images", imgs, [9858, 2124, 2079])
check("c) seed 7 as split seed: malignant", mal, [835, 178, 182])

# The val/test pairs must follow split_seed too, not --seed.
image_dir = os.path.join(args.data_dir, "combined_images")
val_batches = {}
for key, (seed, split_seed) in {"default": (42, None), "decoupled": (7, 42)}.items():
    _, val_loader, _ = d.get_pair_dataloaders(default, image_dir, batch_size=8, seed=seed,
                                              num_pairs=16, split_seed=split_seed)
    val_batches[key] = next(iter(val_loader))
check("d) val pairs for seed=7, split_seed=42 equal the default's",
      all(torch.equal(a, b) for a, b in zip(val_batches["default"], val_batches["decoupled"])), True)

print("\nALL PASSED" if not failures else f"\n{failures} FAILED")
raise SystemExit(1 if failures else 0)
