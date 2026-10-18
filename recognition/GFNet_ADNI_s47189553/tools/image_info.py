"""Show the image dimensions (width x height) in each AD_NC folder."""
import os
from collections import Counter

from PIL import Image
# DATA_DIR = "/home/groups/comp3710/ADNI"
DATA_DIR = "/Users/ryanhuang/Documents/UQ/Sem 2 2026/COMP3710/ADNI"
ROOT = os.path.join(DATA_DIR, "AD_NC")


def main():
    for split in ("train", "test"):
        for cls in ("AD", "NC"):
            folder = os.path.join(ROOT, split, cls)
            sizes = Counter()
            for f in os.listdir(folder):
                if f.endswith(".jpeg"):
                    with Image.open(os.path.join(folder, f)) as img:
                        sizes[img.size] += 1          # (width, height)
            for (w, h), n in sizes.items():
                print(f"{split}/{cls}: {w} x {h} (W x H) -> {n} images")


if __name__ == "__main__":
    main()