"""Sanity checks for dataset.get_dataloaders on the real data.

Prints the shape, dtype and value range of one train batch, checks that val/test
images load identically twice while train images differ (augmentation), counts
labels over 20 train batches and prints the train mean/std. Run from the
repository root (the data path DATA is hard-coded at the top of this file):
    python recognition/Siamese_Network_s5057318/check_loader.py
"""
import sys
import torch

sys.path.insert(0, "recognition/Siamese_Network_s5057318")
import dataset as d

DATA = r"C:\isic_data"
IMG = DATA + r"\combined_images"

df = d.load_clean_metadata(DATA)
splits = d.split_by_id(df)
out = d.get_dataloaders(splits, IMG, batch_size=32, num_workers=0)
print("get_dataloaders returns:", type(out).__name__)
loaders = out if isinstance(out, dict) else dict(zip(["train", "val", "test"], out))
print("loader names:", list(loaders))

# Check 1: shape, dtype and value range of one train batch
batch = next(iter(loaders["train"]))
x, y = batch[0], batch[1]
print("CHECK 1 batch shape:", tuple(x.shape), "dtype:", x.dtype)
print("CHECK 1 min/max: %.2f / %.2f" % (x.min().item(), x.max().item()))
print("CHECK 1 labels in batch:", y.tolist()[:12], "... malignant:", int(y.sum().item()))

# Check 3: the same val and test image loaded twice must be identical
for name in ["val", "test"]:
    ds = loaders[name].dataset
    same = all(torch.equal(ds[i][0], ds[i][0]) for i in range(5))
    print("CHECK 3", name, "same image twice identical:", same)

# Sanity: train augmentation should make repeated loads differ
ds = loaders["train"].dataset
differs = any(not torch.equal(ds[0][0], ds[0][0]) for _ in range(10))
print("CHECK 3 train repeated loads differ (augmentation on):", differs)

# Label balance over 20 train batches
total = mal = 0
for i, b in enumerate(loaders["train"]):
    total += len(b[1])
    mal += int(b[1].sum().item())
    if i == 19:
        break
print("LABELS train images seen:", total, "malignant:", mal, "pct: %.1f" % (100 * mal / total))

# Value range of a val batch (no rotation applied)
vb = next(iter(loaders["val"]))[0]
print("VAL batch min/max: %.2f / %.2f" % (vb.min().item(), vb.max().item()))

ds = loaders["val"].dataset
print("MEAN:", [round(float(m), 3) for m in ds.mean])
print("STD:", [round(float(s), 3) for s in ds.std])
print("Black pixel would normalise to:", [round(float(-m / s), 2) for m, s in zip(ds.mean, ds.std)])