"""Load and split the modified Facebook Page-Page graph dataset.

The course dataset is distributed as an .npz archive containing X, edges_x,
edges_y, y, and page_name. This module validates those arrays, converts them
to PyTorch tensors, and creates deterministic class-stratified splits.

References:
    Course-linked modified Facebook Page-Page archive (128 input features):
    https://github.com/gayanku/facebook_page-page_dataset
    Original node-ID/page-type labels used to verify the class-name mapping:
    https://github.com/gayanku/facebook_page-page_dataset/blob/main/facebook_large/musae_facebook_target.csv
    The supplied feature matrix is loaded directly; this module does not
    reconstruct or refit the publisher's feature-compression procedure.

AI assistance:
    OpenAI Codex assisted in generating, revising and debugging this code.
    The README section "Artificial Intelligence Usage Disclosure" documents
    the scope of assistance and the data audits, smoke runs, complete training
    runs and saved-output checks used to verify the implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data


DEFAULT_SPLIT_RATIOS = (0.60, 0.20, 0.20)
REQUIRED_NPZ_KEYS = {"X", "edges_x", "edges_y", "y", "page_name"}


def _as_one_dimensional(
    array: np.ndarray,
    name: str,
) -> np.ndarray:
    """Return a one-dimensional array or raise an informative error."""

    squeezed = np.asarray(array).squeeze()

    if squeezed.ndim != 1:
        raise ValueError(
            f"{name!r} must be one-dimensional after squeezing; "
            f"received shape {np.asarray(array).shape}"
        )

    return squeezed


def _normalise_labels(
    raw_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the original labels into consecutive integer classes."""

    labels = _as_one_dimensional(raw_labels, "y")

    if labels.dtype.kind in {"f", "c"}:
        if not np.isfinite(labels).all():
            raise ValueError(
                "The label array contains NaN or infinite values"
            )

    class_values, encoded_labels = np.unique(
        labels,
        return_inverse=True,
    )

    if class_values.size < 2:
        raise ValueError(
            "Node classification requires at least two classes"
        )

    return (
        encoded_labels.astype(np.int64, copy=False),
        class_values,
    )


def _validate_ratios(
    ratios: Sequence[float],
) -> tuple[float, float, float]:
    """Check the train, validation, and test proportions."""

    if len(ratios) != 3:
        raise ValueError(
            "ratios must contain train, validation, and test values"
        )

    values = tuple(float(value) for value in ratios)

    if any(value <= 0.0 for value in values):
        raise ValueError("all split ratios must be positive")

    if not np.isclose(
        sum(values),
        1.0,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            f"split ratios must add up to 1.0; received {values}"
        )

    return values


def _class_split_counts(
    class_size: int,
    ratios: tuple[float, float, float],
) -> np.ndarray:
    """Calculate how many members of one class enter each split."""

    if class_size < 5:
        raise ValueError(
            "Each class needs at least five nodes for a 60/20/20 "
            f"split; encountered a class with {class_size} nodes"
        )

    exact_counts = np.asarray(ratios) * class_size
    split_counts = np.floor(exact_counts).astype(np.int64)

    remaining = class_size - int(split_counts.sum())
    fractional_parts = exact_counts - split_counts

    # Stable sorting makes tied results reproducible.
    order = np.argsort(-fractional_parts, kind="stable")
    split_counts[order[:remaining]] += 1

    if np.any(split_counts == 0):
        raise ValueError(
            f"A class produced an empty split: {split_counts}"
        )

    return split_counts


def assert_split_integrity(
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
    labels: Tensor,
) -> None:
    """Check that the three masks are disjoint and cover every node."""

    masks = (train_mask, val_mask, test_mask)
    mask_names = ("train", "validation", "test")
    num_nodes = int(labels.numel())

    for name, mask in zip(mask_names, masks):
        if mask.dtype != torch.bool:
            raise AssertionError(
                f"{name}_mask must have torch.bool type"
            )

        if mask.ndim != 1 or mask.numel() != num_nodes:
            raise AssertionError(
                f"{name}_mask must have shape ({num_nodes},), "
                f"received {tuple(mask.shape)}"
            )

    overlap = (
        (train_mask & val_mask)
        | (train_mask & test_mask)
        | (val_mask & test_mask)
    )

    if bool(overlap.any()):
        overlapping_nodes = (
            overlap.nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )

        raise AssertionError(
            "Split masks overlap at node indices "
            f"{overlapping_nodes[:10]}"
        )

    assigned_nodes = train_mask | val_mask | test_mask

    if not bool(assigned_nodes.all()):
        missing_nodes = (
            (~assigned_nodes)
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )

        raise AssertionError(
            "Some nodes were not assigned to a split: "
            f"{missing_nodes[:10]}"
        )

    # Confirm that every class appears in every split.
    for class_index in torch.unique(labels, sorted=True):
        class_nodes = labels == class_index

        class_counts = [
            int((class_nodes & mask).sum())
            for mask in masks
        ]

        if any(count == 0 for count in class_counts):
            raise AssertionError(
                f"Class {int(class_index)} is absent from a split: "
                f"{class_counts}"
            )


def stratified_split_masks(
    labels: np.ndarray | Tensor | Sequence[int],
    seed: int = 42,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create deterministic class-stratified Boolean masks.

    Args:
        labels: One class index for every graph node.
        seed: Seed controlling the random ordering within each class.
        ratios: Train, validation, and test proportions.

    Returns:
        Training, validation, and testing Boolean masks.
    """

    split_ratios = _validate_ratios(ratios)

    if isinstance(labels, Tensor):
        labels_array = labels.detach().cpu().numpy()
    else:
        labels_array = np.asarray(labels)

    labels_array = _as_one_dimensional(
        labels_array,
        "labels",
    )

    if labels_array.size == 0:
        raise ValueError("Cannot split an empty label array")

    random_generator = np.random.default_rng(seed)

    train_mask_array = np.zeros(
        labels_array.size,
        dtype=np.bool_,
    )
    val_mask_array = np.zeros(
        labels_array.size,
        dtype=np.bool_,
    )
    test_mask_array = np.zeros(
        labels_array.size,
        dtype=np.bool_,
    )

    for class_value in np.unique(labels_array):
        class_indices = np.flatnonzero(
            labels_array == class_value
        )

        random_generator.shuffle(class_indices)

        train_count, val_count, test_count = (
            _class_split_counts(
                int(class_indices.size),
                split_ratios,
            )
        )

        train_end = int(train_count)
        val_end = train_end + int(val_count)
        test_end = val_end + int(test_count)

        train_indices = class_indices[:train_end]
        val_indices = class_indices[train_end:val_end]
        test_indices = class_indices[val_end:test_end]

        train_mask_array[train_indices] = True
        val_mask_array[val_indices] = True
        test_mask_array[test_indices] = True

    train_mask = torch.from_numpy(train_mask_array)
    val_mask = torch.from_numpy(val_mask_array)
    test_mask = torch.from_numpy(test_mask_array)

    labels_tensor = torch.as_tensor(
        labels_array,
        dtype=torch.long,
    )

    assert_split_integrity(
        train_mask,
        val_mask,
        test_mask,
        labels_tensor,
    )

    return train_mask, val_mask, test_mask


def _load_archive(
    npz_path: Path,
) -> dict[str, np.ndarray]:
    """Read the required arrays from the local NPZ archive."""

    try:
        with np.load(
            npz_path,
            allow_pickle=False,
        ) as archive:
            missing_keys = REQUIRED_NPZ_KEYS.difference(
                archive.files
            )

            if missing_keys:
                raise KeyError(
                    "Dataset is missing required keys: "
                    f"{sorted(missing_keys)}"
                )

            arrays = {
                key: np.asarray(archive[key])
                for key in REQUIRED_NPZ_KEYS
            }

    except ValueError as exc:
        # page_name may be stored as an object array and therefore
        # require allow_pickle=True.
        if "Object arrays cannot be loaded" not in str(exc):
            raise

        with np.load(
            npz_path,
            allow_pickle=True,
        ) as archive:
            missing_keys = REQUIRED_NPZ_KEYS.difference(
                archive.files
            )

            if missing_keys:
                raise KeyError(
                    "Dataset is missing required keys: "
                    f"{sorted(missing_keys)}"
                )

            arrays = {
                key: np.asarray(archive[key])
                for key in REQUIRED_NPZ_KEYS
            }

    return arrays


def load_facebook_page_page(
    npz_path: str | Path,
    seed: int = 42,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> Data:
    """Load the Facebook Page-Page dataset as a PyG Data object.

    Args:
        npz_path: Location of the local facebook.npz archive.
        seed: Seed used to generate the deterministic split.
        ratios: Train, validation, and test proportions.

    Returns:
        A graph containing node features, edges, labels, split masks,
        page names, and class information.
    """

    path = Path(npz_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"Facebook Page-Page dataset was not found: {path}"
        )

    if path.suffix.lower() != ".npz":
        raise ValueError(
            f"Expected an .npz dataset, received {path.name!r}"
        )

    arrays = _load_archive(path)

    # Validate the node-feature matrix.
    features = np.asarray(arrays["X"])

    if features.ndim != 2:
        raise ValueError(
            "X must have shape [nodes, features], "
            f"received {features.shape}"
        )

    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(
            "X must contain at least one node and one feature"
        )

    if not np.issubdtype(features.dtype, np.number):
        raise ValueError("X must contain numerical values")

    if not np.isfinite(features).all():
        raise ValueError(
            "X contains NaN or infinite values"
        )

    # Validate and encode the class labels.
    labels, class_values = _normalise_labels(
        arrays["y"]
    )

    page_names = _as_one_dimensional(
        arrays["page_name"],
        "page_name",
    )

    num_nodes = int(features.shape[0])

    if labels.size != num_nodes:
        raise ValueError(
            f"y contains {labels.size} labels for "
            f"{num_nodes} nodes"
        )

    if page_names.size != num_nodes:
        raise ValueError(
            f"page_name contains {page_names.size} names "
            f"for {num_nodes} nodes"
        )

    # Validate the graph edges.
    edge_sources = _as_one_dimensional(
        arrays["edges_x"],
        "edges_x",
    )
    edge_targets = _as_one_dimensional(
        arrays["edges_y"],
        "edges_y",
    )

    if edge_sources.size != edge_targets.size:
        raise ValueError(
            "edges_x and edges_y must have the same length"
        )

    if edge_sources.size == 0:
        raise ValueError(
            "The graph must contain at least one edge"
        )

    for values, name in (
        (edge_sources, "edges_x"),
        (edge_targets, "edges_y"),
    ):
        if not np.issubdtype(values.dtype, np.integer):
            valid_float_indices = (
                np.issubdtype(values.dtype, np.floating)
                and np.isfinite(values).all()
                and np.equal(values, np.floor(values)).all()
            )

            if not valid_float_indices:
                raise ValueError(
                    f"{name} must contain integer node indices"
                )

    edge_index_array = np.stack(
        (
            edge_sources.astype(np.int64),
            edge_targets.astype(np.int64),
        ),
        axis=0,
    )

    minimum_index = int(edge_index_array.min())
    maximum_index = int(edge_index_array.max())

    if minimum_index < 0 or maximum_index >= num_nodes:
        raise ValueError(
            "Edge index is outside the valid range. "
            f"Expected 0 to {num_nodes - 1}, received "
            f"{minimum_index} to {maximum_index}"
        )

    # Create leakage-free split masks.
    label_tensor = torch.from_numpy(labels)

    train_mask, val_mask, test_mask = (
        stratified_split_masks(
            label_tensor,
            seed=seed,
            ratios=ratios,
        )
    )

    # Construct the PyTorch Geometric graph.
    data = Data(
        x=torch.as_tensor(
            features,
            dtype=torch.float32,
        ),
        edge_index=torch.from_numpy(
            edge_index_array
        ),
        y=label_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # Store metadata used by prediction and reporting.
    data.page_name = [
        str(name)
        for name in page_names.tolist()
    ]

    data.class_values = [
        str(value)
        for value in class_values.tolist()
    ]

    data.num_classes = int(class_values.size)
    data.split_seed = int(seed)
    data.split_ratios = tuple(
        float(value)
        for value in ratios
    )

    if data.num_nodes != num_nodes:
        raise AssertionError(
            f"Constructed graph has {data.num_nodes} nodes; "
            f"expected {num_nodes}"
        )

    assert_split_integrity(
        data.train_mask,
        data.val_mask,
        data.test_mask,
        data.y,
    )

    return data
