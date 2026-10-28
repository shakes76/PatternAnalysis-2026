
from pathlib import Path
import numpy as np

# Location of the dataset
dataset_path = Path(__file__).resolve().parent / "raw" / "facebook.npz"

# Load the downloaded dataset
with np.load(dataset_path, allow_pickle=False) as data:
    features = data["X"]
    edges_x = data["edges_x"]
    edges_y = data["edges_y"]
    target = data["y"]

# Combine source and destination node indices
edges = np.column_stack((edges_x, edges_y))

# Verify the dataset dimensions
assert features.shape == (22470, 128)
assert edges.shape == (342004, 2)
assert target.shape == (22470,)

# Write the converted dataset to a temporary file
converted_path = dataset_path.with_name("facebook_converted.npz")

np.savez(
    converted_path,
    features=features,
    edges=edges,
    target=target
)

# Replace the original dataset
converted_path.replace(dataset_path)

print("Dataset converted successfully!")
print("Features:", features.shape)
print("Edges:", edges.shape)
print("Labels:", target.shape)

