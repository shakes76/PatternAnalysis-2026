from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "facebook.npz"


def load_facebook(path=DATA_PATH):
    with np.load(path, allow_pickle=False) as npz:
        x = torch.tensor(npz["X"], dtype=torch.float32)

        edge_index = torch.tensor(
            np.stack([npz["edges_x"], npz["edges_y"]]),
            dtype=torch.long
        )

        y = torch.tensor(npz["y"], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

def split_nodes(data, seed=42):
    indices = np.arange(data.num_nodes)
    labels = data.y.numpy()

    # First: 60% train, 40% remaining
    train_idx, remaining_idx = train_test_split(
        indices,
        test_size=0.4,
        stratify=labels,
        random_state=seed
    )

    # Second: 20% validation, 20% test
    val_idx, test_idx = train_test_split(
        remaining_idx,
        test_size=0.5,
        stratify=labels[remaining_idx],
        random_state=seed
    )

    # Create boolean masks
    for name, selected in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx)
    ]:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[selected] = True
        setattr(data, f"{name}_mask", mask)

    return data


if __name__ == "__main__":
    data = load_facebook()
    data = split_nodes(data)

    print(data)
    print("Training:", data.train_mask.sum().item())
    print("Validation:", data.val_mask.sum().item())
    print("Testing:", data.test_mask.sum().item())