"""Acquire, validate and split the course Facebook Page-Page graph.

The course mirror is linked in Appendix A.6 of the assessment specification.
Its numeric keys differ from the upstream FacebookPagePage loader, so we
construct a PyG Data object explicitly. Optional pickled page-name metadata
is never loaded. No label is included in node features or message passing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = PROJECT_DIR / "data" / "FacebookPagePage" / "raw" / "facebook.npz"
DATA_COMMIT = "ec01d02c39da4385052023ed28e36f0794f61dc8"
DATA_URL = (
    "https://raw.githubusercontent.com/gayanku/facebook_page-page_dataset/"
    f"{DATA_COMMIT}/facebook.npz"
)
DATA_SHA256 = "7a1b8a0687515e9859d1ca649dc276825f5db44568894dfbb6b55ebc1cf6a9c1"
CLASS_NAMES = ("tvshow", "government", "company", "politician")


def sha256(path: Path) -> str:
    """Return a streaming checksum without loading a whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(path: Path = DEFAULT_PATH) -> Path:
    """Download the pinned course asset; preserve any unexpected existing file."""
    path = Path(path)
    if path.exists():
        if sha256(path) != DATA_SHA256:
            raise ValueError(f"Existing dataset has an unexpected checksum: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.part")
    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        with temporary.open("wb") as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
    if sha256(temporary) != DATA_SHA256:
        raise ValueError(f"Download checksum mismatch; preserved for inspection: {temporary}")
    temporary.replace(path)
    return path


def validate_arrays(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    expected_nodes: int | None = 22470,
    expected_features: int = 128,
) -> None:
    """Reject malformed or mismatched data before any training can start."""
    if x.ndim != 2 or x.shape[1] != expected_features:
        raise ValueError(f"Expected a node-by-{expected_features} feature matrix, got {x.shape}")
    if expected_nodes is not None and x.shape[0] != expected_nodes:
        raise ValueError(f"Expected {expected_nodes} nodes, got {x.shape[0]}")
    if x.dtype.kind not in "fiu" or not np.isfinite(x).all():
        raise ValueError("Features must be finite numeric values")
    if y.ndim != 1 or len(y) != len(x) or y.dtype.kind not in "iu":
        raise ValueError("Labels must be one aligned integer value per node")
    if not np.array_equal(np.unique(y), np.arange(len(CLASS_NAMES))):
        raise ValueError("Expected exactly the integer class labels 0, 1, 2 and 3")
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("Edges must be a 2-by-E integer array")
    if edges.shape[1] == 0 or edges.min() < 0 or edges.max() >= len(x):
        raise ValueError("Edge endpoints must reference existing nodes")


def load_arrays(path: Path = DEFAULT_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read numeric arrays only, supporting both course and original PyG keys."""
    with np.load(Path(path), allow_pickle=False) as archive:
        keys = set(archive.files)
        if {"X", "y", "edges_x", "edges_y"} <= keys:
            x = archive["X"].copy()
            y = archive["y"].copy()
            edges = np.stack((archive["edges_x"], archive["edges_y"]))
        elif {"features", "target", "edges"} <= keys:
            x = archive["features"].copy()
            y = archive["target"].copy()
            raw_edges = archive["edges"].copy()
            if raw_edges.ndim != 2 or raw_edges.shape[1] != 2:
                raise ValueError("Original PyG edge array must have shape E-by-2")
            edges = raw_edges.T
        else:
            raise ValueError(f"Unrecognized numeric dataset keys: {sorted(keys)}")
    validate_arrays(x, y, edges)
    return x.astype(np.float32), y.astype(np.int64), edges.astype(np.int64)


def stratified_split(y: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]:
    """Make an approximately 60/20/20 split, with per-class rounding."""
    random = np.random.default_rng(seed)
    groups: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    for label in np.unique(y):
        indices = random.permutation(np.flatnonzero(y == label))
        if len(indices) < 5:
            raise ValueError("Each class needs at least five nodes for this split")
        train_end = int(0.6 * len(indices))
        val_end = train_end + int(0.2 * len(indices))
        groups["train"].append(indices[:train_end])
        groups["validation"].append(indices[train_end:val_end])
        groups["test"].append(indices[val_end:])
    splits = {name: np.sort(np.concatenate(parts)) for name, parts in groups.items()}
    verify_split(splits, len(y))
    return splits


def verify_split(splits: dict[str, np.ndarray], num_nodes: int) -> None:
    """Require every node to occur in exactly one nonempty split."""
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test split names")
    for indices in splits.values():
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
            raise ValueError("Every split must be a nonempty integer vector")
    joined = np.concatenate(list(splits.values()))
    if len(joined) != num_nodes or not np.array_equal(np.sort(joined), np.arange(num_nodes)):
        raise ValueError("Splits must cover all nodes exactly once without overlap")


def standardize_features(
    x: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit feature scaling on training nodes, then apply the same transform to all."""
    mean = x[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x[train_indices].std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)
    return ((x - mean) / scale).astype(np.float32), mean, scale


def canonical_edges(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    """Make unique, bidirectional non-self edges for adding one GCN self-loop later."""
    source, target = edges
    nonself = source != target
    forward = source[nonself] * num_nodes + target[nonself]
    backward = target[nonself] * num_nodes + source[nonself]
    codes = np.unique(np.concatenate((forward, backward)))
    return np.stack((codes // num_nodes, codes % num_nodes)).astype(np.int64)


def make_pyg_graph(path: Path = DEFAULT_PATH, split_seed: int = 42):
    """Convert the course archive to native PyG data without its incompatible loader.

    Torch/PyG are imported only here so data acquisition and auditing can also
    run on a small CPU-only machine with NumPy. Model forward methods must
    accept only x and edge_index; y is reserved for explicitly masked losses
    and evaluation. This is a transductive, not an inductive, protocol.
    """
    import torch
    from torch_geometric.data import Data

    x, y, edges = load_arrays(path)
    splits = stratified_split(y, split_seed)
    x, mean, scale = standardize_features(x, splits["train"])
    edges = canonical_edges(edges, len(y))
    graph = Data(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        edge_index=torch.from_numpy(edges),
    )
    for name, indices in splits.items():
        mask = torch.zeros(len(y), dtype=torch.bool)
        mask[torch.from_numpy(indices)] = True
        setattr(graph, f"{name}_mask", mask)
    graph.feature_mean = torch.from_numpy(mean)
    graph.feature_scale = torch.from_numpy(scale)
    return graph
