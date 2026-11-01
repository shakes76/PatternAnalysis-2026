"""Independent check of dataset.split_by_id.

Rebuilds every real ID straight from the raw CSV (without dataset.py's real_id
column) and reports images without an ID, IDs found in more than one split and
the image totals. Run from the repository root (the data path DATA is hard-coded
at the top of this file):
    python recognition/Siamese_Network_s5057318/verify_split.py
"""
import sys
import pandas as pd

sys.path.insert(0, "recognition/Siamese_Network_s5057318")
import dataset as d

DATA = r"C:\isic_data"

df = d.load_clean_metadata(DATA)
splits = d.split_by_id(df)

# Normalise whatever came back into {name: DataFrame}
if isinstance(splits, dict):
    named = dict(splits)
else:
    named = dict(zip(["train", "val", "test"], splits))
print("split return type:", type(splits).__name__, "| names:", list(named))
for name, part in named.items():
    print(name, "columns:", list(part.columns), "rows:", len(part))

# Rebuild real IDs straight from the raw CSV, ignoring dataset.py's own ID column
raw = pd.read_csv(DATA + r"\cleaned_metadata.csv")

def real_id(s):
    """First non-dummy ID in a comma-separated patient_id field (written separately from dataset.py)."""
    for x in str(s).split(","):
        x = x.strip().strip('"')
        if x and not x.startswith("dummy"):
            return x
    return None

lookup = raw.set_index("isic_id")["patient_id"].map(real_id)

parts = [pd.DataFrame({"isic_id": p["isic_id"].values, "split": n})
         for n, p in named.items()]
all_df = pd.concat(parts)
all_df["real_id"] = all_df["isic_id"].map(lookup)

print("images with no real id:", int(all_df.real_id.isna().sum()))
n = all_df.groupby("real_id")["split"].nunique()
print("IDs in more than one split:", int((n > 1).sum()))
print("total images:", len(all_df), "unique image ids:", all_df.isic_id.nunique())