"""Synthetic unit tests plus an explicitly gated real-artifact integration test.

Run from the repository root:
    .venv/Scripts/python.exe recognition/Facebook_GCN_Jonte/test_dataset.py -v

Synthetic passes do not establish validity of the Facebook dataset. Real
integration tests run against the approved specification-linked file, with a
separate raw-UCI comparison where that reference archive is available locally.
"""

import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError
import zipfile

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import coalesce

from dataset import (
    DEFAULT_ROOT,
    PARTITIONS,
    SOURCE_SHA256,
    DatasetUnavailableError,
    audit_graph,
    load_facebook,
    prepare_dataset,
    read_prescribed_npz,
    sha256_file,
    stratified_masks,
    validate_graph,
    validate_masks,
)


def synthetic_graph() -> Data:
    """An imbalanced four-class ring used only to test software invariants."""
    generator = torch.Generator().manual_seed(7)
    labels = torch.repeat_interleave(torch.arange(4), torch.tensor([240, 160, 120, 80]))
    nodes = torch.arange(labels.numel())
    forward = torch.stack((nodes, (nodes + 1) % labels.numel()))
    edges = torch.cat((forward, forward.flip(0)), dim=1)
    return Data(x=torch.randn((labels.numel(), 8), generator=generator), y=labels, edge_index=edges)


def write_fixture(path: Path, data: Data) -> None:
    """Write artificial arrays in the approved schema, never as real data."""
    np.savez(
        path, X=data.x.numpy(), y=data.y.numpy(),
        edges_x=data.edge_index[0].numpy(), edges_y=data.edge_index[1].numpy(),
        page_name=np.array(["synthetic metadata"], dtype=object),
    )


class SyntheticDatasetTests(unittest.TestCase):
    """Every test in this class uses artificial fixtures, never Facebook data."""

    def setUp(self):
        self.data = synthetic_graph()
        self.masks = stratified_masks(self.data.y)

    def test_split_sizes_and_disjoint_exhaustive_coverage(self):
        expected = {"train": 360, "val": 120, "test": 120, "selection": 60, "calibration": 30, "policy": 30}
        self.assertEqual({name: int(mask.sum()) for name, mask in self.masks.items()}, expected)
        validate_masks(self.masks, self.data.num_nodes)
        self.assertTrue(torch.all(torch.stack([self.masks[name] for name in PARTITIONS]).sum(0) == 1))

    def test_each_partition_preserves_class_proportions(self):
        proportions = torch.bincount(self.data.y) / self.data.num_nodes
        for name, mask in self.masks.items():
            counts = torch.bincount(self.data.y[mask], minlength=4)
            self.assertTrue(torch.all(counts > 0), name)
            self.assertTrue(torch.all(torch.abs(counts - mask.sum() * proportions) <= 1), name)

    def test_same_seed_is_repeatable_different_seed_changes_membership(self):
        repeated = stratified_masks(self.data.y, seed=42)
        different = stratified_masks(self.data.y, seed=43)
        for name in self.masks:
            self.assertTrue(torch.equal(self.masks[name], repeated[name]), name)
        self.assertFalse(torch.equal(self.masks["train"], different["train"]))

    def test_splitting_does_not_change_global_rng_states(self):
        torch_state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()
        stratified_masks(self.data.y)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        for before, after in zip(numpy_state, np.random.get_state()):
            np.testing.assert_equal(before, after)

    def test_overlap_is_rejected(self):
        masks = {name: mask.clone() for name, mask in self.masks.items()}
        masks["train"][torch.where(masks["test"])[0][0]] = True
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_masks(masks, self.data.num_nodes)

    def test_wrong_validation_union_is_rejected(self):
        masks = {name: mask.clone() for name, mask in self.masks.items()}
        masks["val"][torch.where(masks["train"])[0][0]] = True
        with self.assertRaisesRegex(ValueError, "Validation mask"):
            validate_masks(masks, self.data.num_nodes)

    def test_model_graph_contains_no_labels_and_preserves_all_nodes_and_edges(self):
        prepared = prepare_dataset(self.data)
        self.assertIsNone(prepared.graph.y)
        self.assertEqual(set(prepared.graph.keys()), {"x", "edge_index", "num_nodes"})
        self.assertTrue(torch.equal(prepared.graph.x, self.data.x))
        self.assertTrue(torch.equal(prepared.graph.edge_index, self.data.edge_index))
        # Cross-split edges are deliberately retained in the transductive graph.
        source, target = prepared.graph.edge_index
        self.assertTrue((prepared.masks["train"][source] & prepared.masks["test"][target]).any())

    def test_targets_match_mask_order_and_test_access_is_explicit(self):
        prepared = prepare_dataset(self.data)
        for name in PARTITIONS[:-1]:
            self.assertTrue(torch.equal(prepared.targets(name), self.data.y[prepared.masks[name]]))
        with self.assertRaises(PermissionError):
            prepared.targets("test")
        self.assertTrue(torch.equal(prepared.targets("test", allow_test=True), self.data.y[prepared.masks["test"]]))
        with self.assertRaises(ValueError):
            prepared.targets("val")

    def test_returned_targets_are_copies(self):
        prepared = prepare_dataset(self.data)
        returned = prepared.targets("train")
        returned.fill_(-1)
        self.assertTrue(torch.all(prepared.targets("train") >= 0))

    def test_heldout_target_changes_do_not_change_model_inputs(self):
        prepared = prepare_dataset(self.data)
        features = prepared.graph.x.clone()
        edges = prepared.graph.edge_index.clone()
        prepared._targets["test"].fill_(-1)
        self.assertTrue(torch.equal(prepared.graph.x, features))
        self.assertTrue(torch.equal(prepared.graph.edge_index, edges))

    def test_nonfinite_features_and_out_of_range_edges_are_rejected(self):
        invalid = self.data.clone()
        invalid.x[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN"):
            validate_graph(invalid)
        invalid = self.data.clone()
        invalid.edge_index[0, 0] = invalid.num_nodes
        with self.assertRaises(ValueError):
            validate_graph(invalid)

    def test_bad_labels_are_rejected(self):
        invalid = self.data.clone()
        invalid.y[0] = -1
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_graph(invalid)

    def test_audit_reports_duplicates_loops_and_isolation_without_repairs(self):
        data = self.data.clone()
        # Remove all edges touching one node, duplicate a surviving edge and add a loop.
        data.edge_index = data.edge_index[:, (data.edge_index != 599).all(dim=0)]
        data.edge_index = torch.cat((data.edge_index, data.edge_index[:, :1], torch.tensor([[0], [0]])), dim=1)
        original_edges = data.edge_index.clone()
        report = audit_graph(data, self.masks)
        self.assertEqual(report["duplicate_edge_entries"], 1)
        self.assertEqual(report["self_loop_entries"], 1)
        self.assertEqual(report["isolated_nodes"], 1)
        self.assertTrue(torch.equal(original_edges, data.edge_index))

    def test_incomplete_npz_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "facebook.npz"
            np.savez(path, X=self.data.x.numpy(), y=self.data.y.numpy())
            with self.assertRaisesRegex(ValueError, "specification-linked NPZ keys"):
                read_prescribed_npz(path)

    def test_wrong_feature_dimensions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "facebook.npz"
            write_fixture(path, self.data)
            with self.assertRaisesRegex(ValueError, "Expected features"):
                read_prescribed_npz(path)

    def test_object_arrays_are_not_unpickled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "facebook.npz"
            np.savez(path, X=np.array(["not numeric"], dtype=object), y=np.array([0]), edges_x=np.array([0]), edges_y=np.array([0]))
            with self.assertRaisesRegex(ValueError, "allow_pickle=False"):
                read_prescribed_npz(path)

    def test_download_failure_is_explicit_and_leaves_no_partial_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("dataset.urlopen", side_effect=URLError("synthetic DNS failure")):
                with self.assertRaisesRegex(DatasetUnavailableError, "No substitute dataset"):
                    load_facebook(Path(directory))
            self.assertEqual(list((Path(directory) / "raw").iterdir()), [])

    def test_unverified_offline_copy_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "unknown.npz"
            source.write_bytes(b"not the approved dataset")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_facebook(Path(directory) / "cache", raw_file=source)

    def test_synthetic_offline_pyg_round_trip_and_cached_reload(self):
        # Only this synthetic test changes expected sizes/hash; production
        # constants remain pinned. No artificial file is stored in project data.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.npz"
            write_fixture(source, self.data)
            with patch.multiple("dataset", EXPECTED_NODES=600, EXPECTED_FEATURES=8, EXPECTED_EDGE_ENTRIES=1200, SOURCE_SHA256=sha256_file(source)):
                root = Path(directory) / "cache"
                loaded = load_facebook(root, raw_file=source)
                repeated = load_facebook(root)
            self.assertEqual(loaded.provenance["sha256"], sha256_file(source))
            self.assertTrue(torch.equal(loaded.graph.x, self.data.x))
            self.assertTrue(torch.equal(loaded.graph.edge_index, self.data.edge_index))
            self.assertIsNone(loaded.graph.y)
            for name in loaded.masks:
                self.assertTrue(torch.equal(loaded.masks[name], repeated.masks[name]))
            self.assertTrue((root / "raw" / "provenance.json").is_file())
            self.assertFalse((root / "processed").exists())
            # Corruption must be detected on reload, not hidden by a cache.
            (root / "raw" / "facebook.npz").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_facebook(root)

    def test_checksum_mismatch_does_not_install_an_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.npz"
            write_fixture(source, self.data)
            root = Path(directory) / "cache"
            with patch.multiple("dataset", EXPECTED_NODES=600, EXPECTED_FEATURES=8, EXPECTED_EDGE_ENTRIES=1200):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    load_facebook(root, raw_file=source)
            self.assertEqual(list((root / "raw").iterdir()), [])

    def test_shape_correct_but_nonreciprocal_edges_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.npz"
            edges = self.data.edge_index.t().numpy().copy()
            edges[0] = [0, 4]
            np.savez(source, X=self.data.x.numpy(), y=self.data.y.numpy(), edges_x=edges[:, 0], edges_y=edges[:, 1])
            with patch.multiple("dataset", EXPECTED_NODES=600, EXPECTED_FEATURES=8, EXPECTED_EDGE_ENTRIES=1200):
                with self.assertRaisesRegex(ValueError, "reciprocal"):
                    read_prescribed_npz(source)

    def test_sha256_known_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hash-check.txt"
            path.write_bytes(b"abc")
            self.assertEqual(sha256_file(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


class RealFacebookIntegrationTests(unittest.TestCase):
    """Runs against the exact approved real artifact; no task models involved."""

    @classmethod
    def setUpClass(cls):
        if not (DEFAULT_ROOT / "raw" / "facebook.npz").is_file():
            raise unittest.SkipTest("Download the approved dataset with dataset.py first")
        cls.prepared = load_facebook()

    def test_real_artifact_dimensions_numeric_values_and_checksum(self):
        prepared = self.prepared
        self.assertEqual(prepared.graph.x.shape, (22_470, 128))
        self.assertEqual(prepared.graph.edge_index.shape, (2, 342_004))
        self.assertEqual(prepared.audit["classes"], 4)
        self.assertTrue(prepared.audit["undirected"])
        self.assertEqual(prepared.audit["nonfinite_feature_values"], 0)
        self.assertEqual(prepared.provenance["sha256"], SOURCE_SHA256)
        with np.load(DEFAULT_ROOT / "raw" / "facebook.npz", allow_pickle=False) as raw:
            self.assertTrue(torch.equal(prepared.graph.x, torch.from_numpy(raw["X"])))
            edges = torch.from_numpy(np.stack((raw["edges_x"], raw["edges_y"])))
            self.assertTrue(torch.equal(prepared.graph.edge_index, edges))

    def test_real_graph_integrity(self):
        audit = self.prepared.audit
        self.assertEqual(audit["duplicate_edge_entries"], 179)
        self.assertEqual(audit["self_loop_entries"], 358)
        self.assertEqual(audit["unique_unordered_pairs_including_loops"], 171002)
        self.assertEqual(audit["isolated_nodes"], 0)

    def test_default_gcn_normalization_does_not_double_self_messages(self):
        # Operator check only: no GCN model or training is implemented here.
        graph = self.prepared.graph
        normalized, weights = gcn_norm(
            graph.edge_index, num_nodes=graph.num_nodes, dtype=torch.float32,
        )
        loops = normalized[0] == normalized[1]
        self.assertEqual(int(loops.sum()), graph.num_nodes)
        self.assertEqual(normalized.size(1), 364116)
        unique_edges = coalesce(graph.edge_index, num_nodes=graph.num_nodes)
        reference_edges, reference_weights = gcn_norm(
            unique_edges, num_nodes=graph.num_nodes, dtype=torch.float32,
        )
        actual = torch.sparse_coo_tensor(
            normalized, weights, (graph.num_nodes, graph.num_nodes),
        ).coalesce()
        reference = torch.sparse_coo_tensor(
            reference_edges, reference_weights, actual.shape,
        ).coalesce()
        self.assertTrue(torch.equal(actual.indices(), reference.indices()))
        self.assertTrue(torch.allclose(actual.values(), reference.values()))

    def test_real_split_coverage_class_balance_and_repeatability(self):
        prepared = self.prepared
        validate_masks(prepared.masks, 22_470)
        self.assertEqual(prepared.audit["class_counts"], [6495, 6880, 5768, 3327])
        proportions = torch.tensor(prepared.audit["class_counts"]) / 22470
        for report in prepared.audit["splits"].values():
            self.assertTrue(all(count > 0 for count in report["class_counts"]))
            deviations = torch.tensor(report["class_counts"]) - report["nodes"] * proportions
            self.assertTrue(torch.all(torch.abs(deviations) <= 1))
        repeated = load_facebook()
        for name in prepared.masks:
            self.assertTrue(torch.equal(prepared.masks[name], repeated.masks[name]), name)

    def test_real_labels_are_separate_and_match_raw_mask_order(self):
        prepared = self.prepared
        self.assertIsNone(prepared.graph.y)
        with self.assertRaises(PermissionError):
            prepared.targets("test")
        with np.load(DEFAULT_ROOT / "raw" / "facebook.npz", allow_pickle=False) as raw:
            labels = torch.from_numpy(raw["y"]).long()
        for name in PARTITIONS:
            self.assertTrue(torch.equal(prepared.targets(name, allow_test=True), labels[prepared.masks[name]]))

    @unittest.skipUnless((DEFAULT_ROOT.parent / "reference" / "uci-facebook.zip").is_file(), "Optional raw-UCI reference archive is not downloaded")
    def test_real_labels_and_unique_edges_match_independent_uci_source(self):
        reference = DEFAULT_ROOT.parent / "reference" / "uci-facebook.zip"
        self.assertEqual(sha256_file(reference), "fa2f5c4bcb806695873acaba2769c318aa41085417074ae0406fe7c450a4e0d8")
        with zipfile.ZipFile(reference) as archive:
            label_rows = list(csv.DictReader(io.StringIO(archive.read("facebook_large/musae_facebook_target.csv").decode())))
            edge_rows = list(csv.DictReader(io.StringIO(archive.read("facebook_large/musae_facebook_edges.csv").decode())))
        with np.load(DEFAULT_ROOT / "raw" / "facebook.npz", allow_pickle=False) as raw:
            labels = raw["y"]
        class_names = ("company", "government", "politician", "tvshow")
        for row in label_rows:
            self.assertEqual(class_names[int(labels[int(row["id"])])], row["page_type"])
        edges = {(int(row["id_1"]), int(row["id_2"])) for row in edge_rows}
        self.assertEqual(set(map(tuple, self.prepared.graph.edge_index.t().tolist())), edges | {(b, a) for a, b in edges})


if __name__ == "__main__":
    unittest.main()
