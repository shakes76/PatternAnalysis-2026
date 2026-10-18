"""
Count unique patients in each AD_NC folder.

"""
import os
DATA_DIR = "/home/groups/comp3710/ADNI"
# DATA_DIR = "/Users/ryanhuang/Documents/UQ/Sem 2 2026/COMP3710/ADNI"
ROOT = os.path.join(DATA_DIR, "AD_NC")
 
 
def ids_in(folder):
    """Return the set of unique IDs from the .jpeg filenames in a folder."""
    return {f.rsplit("_", 1)[0]
            for f in os.listdir(folder) if f.endswith(".jpeg")}
 
 
def main():
    ids = {}
    print(f"{'folder':<12}{'images':>8}{'unique IDs':>12}")
    for split in ("train", "test"):
        for cls in ("AD", "NC"):
            folder = os.path.join(ROOT, split, cls)
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