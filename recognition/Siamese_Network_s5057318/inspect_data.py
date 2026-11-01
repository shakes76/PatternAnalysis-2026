"""Inspect the ISIC metadata: keep only rows with a real (non-dummy) patient ID."""
import argparse
import os

import pandas as pd

from dataset import extract_real_id


def main():
    """Print rows kept, malignant count, unique real IDs and malignant percentage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder containing cleaned_metadata.csv")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.data_dir, "cleaned_metadata.csv"),
                     dtype={"patient_id": str})
    df["real_id"] = df["patient_id"].map(extract_real_id)
    kept = df.dropna(subset=["real_id"])

    malignant = int((kept["target"] == 1).sum())
    print(f"Rows kept: {len(kept)} (of {len(df)})")
    print(f"Malignant count: {malignant}")
    print(f"Unique real IDs: {kept['real_id'].nunique()}")
    print(f"Malignant percentage: {100 * malignant / len(kept):.2f}%")


if __name__ == "__main__":
    main()
