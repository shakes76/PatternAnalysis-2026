"""Load the approved specification-linked Facebook graph and create fixed splits.

References:
https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.FacebookPagePage.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
https://github.com/gayanku/facebook_page-page_dataset

This is transductive node classification: all features and edges are visible,
but labels are never placed on the graph supplied to a model. Test targets have
an explicit access guard; this is an accident-prevention API, not a security
boundary. Stratification and descriptive audits use labels before training.
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected


DEFAULT_ROOT = Path(__file__).resolve().parent / "data" / "facebook"
SOURCE_REVISION = "ec01d02c39da4385052023ed28e36f0794f61dc8"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gayanku/facebook_page-page_dataset/"
    f"{SOURCE_REVISION}/facebook.npz"
)
SOURCE_SHA256 = "7a1b8a0687515e9859d1ca649dc276825f5db44568894dfbb6b55ebc1cf6a9c1"
CLASS_NAMES = ("company", "government", "politician", "tvshow")
EXPECTED_NODES = 22_470
EXPECTED_FEATURES = 128
EXPECTED_EDGE_ENTRIES = 342_004
PARTITIONS = ("train", "selection", "calibration", "policy", "test")


class DatasetUnavailableError(RuntimeError):
    """The prescribed artifact could not be obtained; do not substitute data."""


def sha256_file(path: Path) -> str:
    """Hash bytes in bounded memory for provenance and corruption checks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_graph(data: Data) -> None:
    """Reject malformed numeric tensors without changing features or edges."""
    if data.x is None or data.x.ndim != 2 or not data.x.is_floating_point():
        raise ValueError("Features must be a two-dimensional floating tensor")
    if not torch.isfinite(data.x).all():
        raise ValueError("Features contain NaN or infinity")
    if data.y is None or data.y.dtype != torch.long or data.y.shape != (data.x.size(0),):
        raise ValueError("Labels must be one int64 target per node")
    classes = torch.unique(data.y)
    if not torch.equal(classes, torch.arange(classes.numel(), device=classes.device)):
        raise ValueError("Labels must be contiguous class IDs starting at zero")
    if data.edge_index is None or data.edge_index.dtype != torch.long:
        raise ValueError("edge_index must contain int64 node indices")
    data.validate(raise_on_error=True)


def read_prescribed_npz(path: Path) -> Data:
    """Adapt the approved numeric arrays to PyG Data without enabling pickle.

    The file contains existing SVD features, not original GraphMining features.
    No embedding is fitted here. The page_name object array is never accessed.
    load_facebook separately verifies the pinned hash before calling this reader.
    """
    with np.load(path, allow_pickle=False) as archive:
        required = {"X", "y", "edges_x", "edges_y"}
        if not required.issubset(archive.files):
            raise ValueError(f"Expected approved specification-linked NPZ keys {sorted(required)}")
        features = archive["X"]
        targets = archive["y"]
        source_nodes = archive["edges_x"]
        destination_nodes = archive["edges_y"]
        if source_nodes.ndim != 1 or destination_nodes.ndim != 1:
            raise ValueError("Expected one-dimensional edge endpoint arrays")
        if source_nodes.dtype.kind not in "iu" or destination_nodes.dtype.kind not in "iu":
            raise ValueError("Edge endpoint arrays must be integer-valued")
        edges = np.stack((source_nodes, destination_nodes), axis=1)
        if features.shape != (EXPECTED_NODES, EXPECTED_FEATURES):
            raise ValueError(f"Expected features ({EXPECTED_NODES}, {EXPECTED_FEATURES}); got {features.shape}")
        if targets.shape != (EXPECTED_NODES,) or targets.dtype.kind not in "iu":
            raise ValueError("Expected one integer target for each of 22,470 nodes")
        if edges.shape != (EXPECTED_EDGE_ENTRIES, 2) or edges.dtype.kind not in "iu":
            raise ValueError("Expected 342,004 integer edge pairs in the approved NPZ")
        if features.dtype.kind != "f":
            raise ValueError("Expected floating-point descriptive features")
        data = Data(
            x=torch.from_numpy(features).float(),
            y=torch.from_numpy(targets).long(),
            edge_index=torch.from_numpy(edges).long().t().contiguous(),
        )
    validate_graph(data)
    if torch.unique(data.y).numel() != 4:
        raise ValueError("Expected exactly four classes")
    if not is_undirected(data.edge_index, num_nodes=data.num_nodes):
        raise ValueError("Expected reciprocal edges for mutual page likes")
    return data


def stratified_masks(labels: torch.Tensor, seed: int = 42) -> dict[str, torch.Tensor]:
    """Make reproducible 60/20/20 node splits, with separate validation roles.

    Validation is subdivided into selection/calibration/policy in a 2:1:1 ratio.
    Integer rounding gives one extra policy node for the full Facebook graph.
    A local sklearn random_state avoids changing NumPy or torch global RNG state.
    """
    if labels.ndim != 1 or labels.dtype != torch.long or labels.device.type != "cpu":
        raise ValueError("Splitting expects a one-dimensional CPU int64 label tensor")
    indices = np.arange(labels.numel())
    y = labels.numpy()

    def divide(nodes: np.ndarray, first_fraction: float):
        return train_test_split(
            nodes, train_size=first_fraction, stratify=y[nodes], random_state=seed
        )

    train, heldout = divide(indices, 0.6)
    validation, test = divide(heldout, 0.5)
    selection, remaining_validation = divide(validation, 0.5)
    calibration, policy = divide(remaining_validation, 0.5)
    masks = {}
    for name, nodes in zip(PARTITIONS, (train, selection, calibration, policy, test)):
        mask = torch.zeros(labels.numel(), dtype=torch.bool)
        mask[torch.from_numpy(nodes)] = True
        masks[name] = mask
    masks["val"] = masks["selection"] | masks["calibration"] | masks["policy"]
    validate_masks(masks, labels.numel())
    return masks


def validate_masks(masks: dict[str, torch.Tensor], num_nodes: int) -> None:
    """Check shapes, nonempty partitions, exclusivity and exhaustive coverage."""
    for name in (*PARTITIONS, "val"):
        mask = masks[name]
        if mask.dtype != torch.bool or mask.shape != (num_nodes,) or not mask.any():
            raise ValueError(f"Invalid or empty {name} mask")
    assignments = torch.stack([masks[name] for name in PARTITIONS]).sum(dim=0)
    if not torch.all(assignments == 1):
        raise ValueError("Every node must belong to exactly one disjoint partition")
    validation = masks["selection"] | masks["calibration"] | masks["policy"]
    if not torch.equal(masks["val"], validation):
        raise ValueError("Validation mask does not equal its three role masks")


def audit_graph(data: Data, masks: dict[str, torch.Tensor]) -> dict:
    """Describe raw contents, including duplicates/self-loops, without repairs."""
    validate_graph(data)
    validate_masks(masks, data.num_nodes)
    source, destination = data.edge_index
    keys = source * data.num_nodes + destination
    unique_keys = torch.unique(keys)
    incident_nodes = torch.unique(data.edge_index)
    unique_pairs = torch.unique(
        torch.minimum(source, destination) * data.num_nodes
        + torch.maximum(source, destination)
    )
    nclasses = int(torch.unique(data.y).numel())
    return {
        "nodes": data.num_nodes,
        "features": data.num_node_features,
        "feature_dtype": str(data.x.dtype),
        "feature_min": float(data.x.min()),
        "feature_max": float(data.x.max()),
        "nonfinite_feature_values": int((~torch.isfinite(data.x)).sum()),
        "classes": nclasses,
        "class_counts": torch.bincount(data.y, minlength=nclasses).tolist(),
        "edge_entries": data.num_edges,
        "duplicate_edge_entries": data.num_edges - unique_keys.numel(),
        "self_loop_entries": int((source == destination).sum()),
        "unique_unordered_pairs_including_loops": unique_pairs.numel(),
        "isolated_nodes": data.num_nodes - incident_nodes.numel(),
        "undirected": is_undirected(data.edge_index, num_nodes=data.num_nodes),
        "splits": {
            name: {
                "nodes": int(mask.sum()),
                "class_counts": torch.bincount(data.y[mask], minlength=nclasses).tolist(),
                "mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
            }
            for name, mask in masks.items()
        },
    }


@dataclass
class NodeClassificationDataset:
    """Feature/edge-only graph, node masks and separately held label vectors."""

    graph: Data
    masks: dict[str, torch.Tensor]
    audit: dict
    provenance: dict
    _targets: dict[str, torch.Tensor] = field(repr=False)

    def targets(self, split: str, *, allow_test: bool = False) -> torch.Tensor:
        """Return labels in boolean-mask order; test access must be explicit.

        Use selection, not the combined val mask, for checkpoint selection.
        Calibration and policy targets are reserved for their later stages.
        Returned copies prevent callers from overwriting the stored targets.
        """
        if split == "test" and not allow_test:
            raise PermissionError("Test labels are reserved for final evaluation")
        if split not in PARTITIONS:
            raise ValueError(f"Choose a specific label role from {PARTITIONS}")
        return self._targets[split].clone()


def prepare_dataset(data: Data, seed: int = 42, provenance: dict | None = None):
    """Split a validated graph and remove all labels from the model input."""
    validate_graph(data)
    masks = stratified_masks(data.y, seed)
    audit = audit_graph(data, masks)
    audit["split_seed"] = seed
    graph = Data(x=data.x.clone(), edge_index=data.edge_index.clone(), num_nodes=data.num_nodes)
    return NodeClassificationDataset(
        graph=graph,
        masks=masks,
        audit=audit,
        provenance=provenance or {},
        _targets={name: data.y[masks[name]].clone() for name in PARTITIONS},
    )


def load_facebook(
    root: Path = DEFAULT_ROOT,
    seed: int = 42,
    *,
    raw_file: Path | None = None,
) -> NodeClassificationDataset:
    """Download or import the exact approved file and verify its pinned checksum.

    The original FacebookPagePage loader expects a different schema and its
    download host fails DNS. The student explicitly approved this alternative
    linked in Appendix A of the specification. A small adapter constructs native
    PyG Data directly. No processed pickle cache or alternative-source fallback
    is used. Every load rechecks the pinned raw-file SHA256.
    """
    root = Path(root).resolve()
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / "facebook.npz"
    manifest_path = raw_dir / "provenance.json"
    if raw_file is not None and destination.exists():
        if sha256_file(Path(raw_file)) != sha256_file(destination):
            raise ValueError("A different raw file already exists; choose a fresh --root")
    if not destination.exists():
        with tempfile.NamedTemporaryFile(dir=raw_dir, suffix=".part", delete=False) as temp:
            temporary_path = Path(temp.name)
        try:
            if raw_file is not None:
                shutil.copyfile(raw_file, temporary_path)
            else:
                try:
                    with urlopen(SOURCE_URL, timeout=30) as response:
                        with temporary_path.open("wb") as output:
                            shutil.copyfileobj(response, output)
                except (URLError, TimeoutError, OSError) as error:
                    raise DatasetUnavailableError(
                        f"Cannot obtain approved dataset from {SOURCE_URL}: {error}. "
                        "Use --raw-file with a copy of the same pinned artifact. "
                        "No substitute dataset was loaded."
                    ) from error
            checksum = sha256_file(temporary_path)
            if checksum != SOURCE_SHA256:
                raise ValueError("Raw-file SHA256 does not match the pinned approved artifact")
            read_prescribed_npz(temporary_path)
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    checksum = sha256_file(destination)
    if checksum != SOURCE_SHA256:
        raise ValueError("Raw-file SHA256 does not match the pinned approved artifact")
    raw_data = read_prescribed_npz(destination)
    provenance = {
        "source": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "sha256": checksum,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "loader": "numeric NPZ adapter to torch_geometric.data.Data",
        "features": "source-provided 128-dimensional TruncatedSVD features; not original PyG features",
        "source_decision": "specification-linked file explicitly approved by student on 2026-10-17",
        "class_names": list(CLASS_NAMES),
    }
    manifest_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return prepare_dataset(raw_data, seed, provenance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-file", type=Path, help="Offline copy of the exact pinned specification-linked NPZ")
    parser.add_argument("--audit-output", type=Path, help="Save generated audit JSON under an ignored outputs/ folder")
    args = parser.parse_args()
    try:
        dataset = load_facebook(args.root, args.seed, raw_file=args.raw_file)
    except (DatasetUnavailableError, ValueError, OSError) as error:
        parser.exit(1, f"DATASET NOT READY: {error}\n")
    report = {"provenance": dataset.provenance, "audit": dataset.audit}
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
