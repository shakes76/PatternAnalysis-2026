import numpy as np
from dataset_v4 import prepare_dataset_v4

data = prepare_dataset_v4("data")
norm = data["normalizer"]

real = norm.inverse_transform(
    np.asarray(data["val"][:512], dtype=np.float64)
)

generated = np.load(
    "outputs_v5_38_integrated/generated_lob.npz"
)

book = generated["relative_orderbook"].astype(
    np.float64
)

print("=== V5.40 PRICE REFERENCE AUDIT ===")

# LOBSTER original CSV price columns are
# Ask1, Bid1, Ask2, Bid2, ...
raw_csv = np.loadtxt(
    "data/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv",
    delimiter=",",
    max_rows=1000
)

# LOBSTER prices are stored as integer units
# of 1/10000 dollars.
ask_raw = raw_csv[:, 0] / 10000
bid_raw = raw_csv[:, 2] / 10000

print("Raw CSV Ask1 mean:", ask_raw.mean())
print("Raw CSV Bid1 mean:", bid_raw.mean())
print("Raw CSV Midprice mean:",
      ((ask_raw + bid_raw) / 2).mean())

print("\nNORMALIZER OUTPUT")
print("Real Ask1 mean:", real[..., 0].mean())
print("Real Bid1 mean:", real[..., 2].mean())
print("Real Mid mean:",
      ((real[..., 0] + real[..., 2]) / 2).mean())

print("\nGENERATED OUTPUT")
print("Generated Ask1 mean:", book[..., 0].mean())
print("Generated Bid1 mean:", book[..., 2].mean())
print("Generated relative Mid mean:",
      ((book[..., 0] + book[..., 2]) / 2).mean())

print("\nDATASET STRUCTURE")
print("Train shape:", data["train"].shape)
print("Validation shape:", data["val"].shape)
print("Generated shape:", book.shape)

print("\nV5.40 price reference audit complete!")
