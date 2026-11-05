"""
dataset.py
Author: Derek McKinney (dm-uq), SN: 45506053

Contains utilities for loading the FacebookPagePage dataset and for
constructing reproducible train/validation/test node masks used by the
project experiments.

References:

    Facebook Page-Page dataset:
    https://github.com/gayanku/facebook_page-page_dataset

    PyG FacebookPagePage:
    https://pytorch-geometric.readthedocs.io/en/latest/generated/
    torch_geometric.datasets.FacebookPagePage.html

"""
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import FacebookPagePage

DEFAULT_DATA_ROOT: Final[Path] = Path("data/FacebookPagePage")

# mapping from original dataset numerical class labels to descriptive labels
CLASS_NAMES: Final[dict[int, str]] = {
    0: "tvshow",
    1: "government",
    2: "company",
    3: "politician",
}


def load_facebook_data(
        root: str | Path = DEFAULT_DATA_ROOT,
) -> FacebookPagePage:
    """
    Load the Facebook Page-Page dataset using PyTorch Geometric.

    Args:
        root: Directory containing the raw and processed dataset files.

    Returns:
        The PyG FacebookPagePage dataset.
    """
    return FacebookPagePage(root=str(root))


def add_stratified_masks(
        data: Data,
        train_pct: int = 60,
        val_pct: int = 20,
        test_pct: int = 20,
        seed: int = 808,
) -> Data:
    """
    Add reproducible stratified train, validation, and test masks to graph
    data schema.

    Class proportions are approximately preserved across all three splits.
    The masks partition the graph nodes only, with the graph structure and node
    features remaining available for transductive message passing.

    Args:
        data: PyG graph containing node labels in `data.y`.
        train_pct: Percentage of nodes assigned to training.
        val_pct: Percentage of nodes assigned to validation.
        test_pct: Percentage of nodes assigned to testing. The three split
            percentages must sum to 100.
        seed: Random seed used for the stratified splits.

    Returns:
        The graph with `train_mask`, `val_mask`, and `test_mask` added.
    """
    # validate user-provided data split args
    if any(
            pct <= 0
            for pct in (train_pct, val_pct, test_pct)
    ):
        raise ValueError('All split percentages must be greater than zero.')

    if train_pct + val_pct + test_pct != 100:
        raise ValueError(
            "Train, val and test percentages must sum to 100."
        )

    # validate node count and construct indices for splitting
    num_nodes: int | None = data.num_nodes

    if num_nodes is None:
        raise ValueError("Graph does not define the number of nodes.")

    indices: NDArray[np.int64] = cast(
        NDArray[np.int64],
        np.arange(
            num_nodes,
            dtype=np.int64,
        ),
    )

    # validate graph labels and convert to NumPy
    labels_tensor = data.y

    if not isinstance(labels_tensor, Tensor):
        raise TypeError("Graph labels must be stored as a PyTorch Tensor.")

    if labels_tensor.dtype != torch.int64:
        raise TypeError("Graph labels must have dtype torch.int64.")

    labels: NDArray[np.int64] = labels_tensor.cpu().numpy()

    # define fractions used by sklearn
    train_fraction: float = train_pct / 100
    val_fraction: float = val_pct / (val_pct + test_pct)

    # create training split
    train_idx, remaining_idx = train_test_split(
        indices,
        train_size=train_fraction,
        stratify=labels,
        random_state=seed,
    )

    # split remaining nodes into validation and test sets
    val_idx, test_idx = train_test_split(
        remaining_idx,
        train_size=val_fraction,
        stratify=labels[remaining_idx],
        random_state=seed,
    )

    # initialise data split masks
    train_mask: Tensor = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )
    val_mask: Tensor = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )
    test_mask: Tensor = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )

    # set data split masks
    train_mask[torch.from_numpy(train_idx)] = True
    val_mask[torch.from_numpy(val_idx)] = True
    test_mask[torch.from_numpy(test_idx)] = True

    # add data split masks to graph data schema
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data
