"""
Count unique patients in each AD_NC folder.

Usage:
    python tools/count_ids.py                                  # Rangpur default
    python tools/count_ids.py "/Users/ryanhuang/Documents/UQ/Sem 2 2026/COMP3710/ADNI"                             # folder containing AD_NC
    python tools/count_ids.py "path/to/AD_NC"                  # or AD_NC itself
"""
import argparse
import os

DEFAULT_DIR = "/home/groups/comp3710/ADNI"


def find_root(path):
    """Accept either the AD_NC folder or the folder that contains it."""
    if os.path.isdir(os.path.join(path, "train")):
        return path
    if os.path.isdir(os.path.join(path, "AD_NC", "train")):
        return os.path.join(path, "AD_NC")
    raise SystemExit(f"No AD_NC/train found under: {path}")


def ids_in(folder):
    """Return the set of unique IDs from the .jpeg filenames in a folder."""
    return {f.rsplit("_", 1)[0]
            for f in os.listdir(folder) if f.endswith(".jpeg")}


def main():
    ap = argparse.ArgumentParser(description="Count unique patient IDs in ADNI AD_NC")
    ap.add_argument("path", nargs="?", default=DEFAULT_DIR,
                    help="AD_NC folder, or the folder containing it (default: Rangpur)")
    root = find_root(os.path.expanduser(ap.parse_args().path))
    print(f"Root: {root}\n")

    ids = {}
    print(f"{'folder':<12}{'images':>8}{'unique IDs':>12}")
    for split in ("train", "test"):
        for cls in ("AD", "NC"):
            folder = os.path.join(root, split, cls)
            n_images = sum(f.endswith(".jpeg") for f in os.listdir(folder))
            ids[(split, cls)] = ids_in(folder)
            print(f"{split + '/' + cls:<12}{n_images:>8}{len(ids[(split, cls)]):>12}")

    train = ids[("train", "AD")] | ids[("train", "NC")]
    test = ids[("test", "AD")] | ids[("test", "NC")]
    print()
    print(f"train unique IDs: {len(train)}")
    print(f"test unique IDs:  {len(test)}")
    print(f"IDs in both train and test: {len(train & test)}")


if __name__ == "__main__":
    main()