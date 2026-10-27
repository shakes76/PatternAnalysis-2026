"""Meaningful v2 guards/geometry/sampling/inference fixtures; no training runs.

Array/transaction fixtures run without Torch. Torch inference and deep-head
gradient fixtures must also pass in the actual execution environment; skipped
local cases are not counted as verified model execution.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v2_pipeline", HERE / "v2_pipeline.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


class ControlledRng:
    def __init__(self, selected_class, voxel_index=0):
        self.selected_class, self.voxel_index, self.calls = selected_class, voxel_index, 0

    def random(self):
        return 0.0

    def integers(self, low, high=None):
        self.calls += 1
        if high == 6:
            return self.selected_class
        return self.voxel_index if high is None and self.calls == 2 else 0


class ArrayFixtures(unittest.TestCase):
    def test_each_uniformly_selected_organ_is_centered_not_union_weighted(self):
        mask = np.zeros((3, 4, 5), dtype=np.uint8)
        for label in range(1, 6):
            mask[0, 0, label - 1] = label
        mask[2] = 1  # Body has many more voxels but is not reweighted by size.
        for label in range(1, 6):
            center, requested, chosen, fallback = v2.sample_center(mask, ControlledRng(label), 1.0)
            self.assertEqual(int(mask[center]), label)
            self.assertEqual(chosen, label)
            self.assertTrue(requested)
            self.assertFalse(fallback)

    def test_absent_selected_class_uses_explicit_spatial_fallback(self):
        mask = np.ones((3, 4, 5), dtype=np.uint8)
        center, requested, chosen, fallback = v2.sample_center(mask, ControlledRng(5), 1.0)
        self.assertEqual(center, (0, 0, 0))
        self.assertEqual(chosen, 5)
        self.assertTrue(requested and fallback)

    def test_uniform_branch_does_not_consult_organ_locations(self):
        first = np.zeros((7, 8, 9), dtype=np.uint8)
        second = np.full_like(first, 5)
        a = v2.sample_center(first, np.random.default_rng(817), 0.0)
        b = v2.sample_center(second, np.random.default_rng(817), 0.0)
        self.assertEqual(a, b)
        self.assertFalse(a[1])

    def test_patch_edges_are_legal_full_size_without_padding(self):
        shape, size = (96, 192, 192), (64, 128, 128)
        for center, starts in [((0, 0, 0), (0, 0, 0)), ((95, 191, 191), (32, 64, 64))]:
            slices = v2.patch_slices(center, shape, size)
            self.assertEqual(tuple(s.start for s in slices), starts)
            self.assertEqual(tuple(s.stop - s.start for s in slices), size)
        with self.assertRaises(ValueError):
            v2.patch_slices((0, 0, 0), (8, 8, 8), (9, 4, 4))

    def test_window_boundaries_cover_nondivisible_grid(self):
        starts = v2.window_starts(17, 8, .5)
        self.assertEqual(starts, [0, 4, 8, 9])
        covered = np.zeros(17)
        for start in starts:
            covered[start:start + 8] += 1
        self.assertTrue(np.all(covered > 0))

    def test_center_aware_grid_preserves_world_edge_extent(self):
        affine = np.diag([2., 3., 4., 1.])
        record = {key: {"shape_xyz": [100, 80, 40], "affine_xyz": affine.tolist()}
                  for key in ("original_image", "original_mask", "canonical_image", "canonical_mask")}
        geometry = v2.expected_geometry(record, [20, 40, 50])
        new = np.array(geometry["resampled_affine_xyz"])
        np.testing.assert_allclose(new @ [-.5, -.5, -.5, 1], affine @ [-.5, -.5, -.5, 1])
        np.testing.assert_allclose(new @ [49.5, 39.5, 19.5, 1], affine @ [99.5, 79.5, 39.5, 1])
        np.testing.assert_allclose(geometry["spacing_dhw_mm"], [8., 6., 4.])

    def test_path_escape_and_existing_output_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                v2._safe_relative(directory, "../old_evidence.npz")
            with self.assertRaises(FileExistsError):
                v2._fresh_directory(directory)

    def test_changed_scientific_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan = {"experiment_id": "fixture", "configuration": copy.deepcopy(v2.DEFAULT_CONFIGURATION)}
            v2.write_json(path, plan)
            self.assertEqual(v2.load_protocol(path)["_sha256"], v2.sha256(path))
            plan["configuration"]["seed"] += 1
            v2.write_json(path, plan)
            with self.assertRaises(ValueError):
                v2.load_protocol(path)

    def test_new_cache_digest_and_actual_mask_counts_are_both_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "case.npz"
            shape = (2, 3, 4)
            np.savez_compressed(path, image=np.zeros(shape, np.float32),
                                mask=np.ones(shape, np.uint8), spacing=np.ones(3, np.float32))
            record = {"patient": "T", "visit": "0", "split": "train", "cache_path": path.name,
                      "pending": False, "cache_sha256": v2.sha256(path), "spacing_dhw_mm": [1., 1., 1.],
                      "class_voxel_counts": [0, 24, 0, 0, 0, 0]}
            manifest = {"target_shape_dhw": list(shape), "records": [record]}
            store = v2.CachedVolumes(root / "manifest.json", manifest, "train")
            _, mask, _ = store.load(0)
            self.assertTrue(np.all(mask == 1))
            wrong = copy.deepcopy(manifest)
            wrong["records"][0]["class_voxel_counts"] = [24, 0, 0, 0, 0, 0]
            with self.assertRaises(ValueError):
                v2.CachedVolumes(root / "manifest.json", wrong, "train").load(0)
            path.write_bytes(b"changed cached bytes")
            with self.assertRaises(ValueError):
                v2.CachedVolumes(root / "manifest.json", manifest, "train").load(0)

    def test_qa_gate_requires_final_fixed_fit_and_full_pipeline_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fit").mkdir()
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            protocol = {"_sha256": "a" * 64, "qa": {"fit_seed": 3711, "fit_steps": 400,
                "fit_case": ["T", "0"], "fit_amp": False, "minimum_fit_prostate_dice": .9,
                "minimum_fit_foreground_dice": .9, "maximum_final_initial_loss_ratio": .25}}
            identity = {"manifest_sha256": v2.sha256(manifest), "protocol_sha256": protocol["_sha256"]}
            fit = {"status": "passed", **identity, "case": ["T", "0"], "amp": False,
                   "seed": 3711, "steps": 400, "models": {name: {"passed": True,
                   "history": [{"step": 0, "loss": 2.}, {"step": 400, "loss": .2,
                       "prostate_dice": .95, "foreground_dice": .96}]} for name in v2.MODELS}}
            fit_path, qa_path = root / "fit/fit_receipt.json", root / "qa_receipt.json"
            v2.write_json(fit_path, fit)
            report = {"status": "passed", **identity, "fit_receipt_sha256": v2.sha256(fit_path),
                      "models": {name: {"passed": True, "full_image_only_inference_passed": True,
                        "native_geometry_serialisation_passed": True, "independent_metric_check_passed": True,
                        "final_fit_prostate_dice": .95, "final_fit_foreground_dice": .96} for name in v2.MODELS}}
            v2.write_json(qa_path, report)
            v2.validate_qa_receipt(qa_path, manifest, protocol)
            report["models"]["improved3d"]["full_image_only_inference_passed"] = False
            v2.write_json(qa_path, report)
            with self.assertRaises(ValueError):
                v2.validate_qa_receipt(qa_path, manifest, protocol)
            report["models"]["improved3d"]["full_image_only_inference_passed"] = True
            # A forged passed flag and matching new digest still cannot turn
            # an early 300-step fit into the required final 400-step gate.
            fit["models"]["improved3d"]["history"][-1]["step"] = 300
            v2.write_json(fit_path, fit)
            report["fit_receipt_sha256"] = v2.sha256(fit_path)
            v2.write_json(qa_path, report)
            with self.assertRaises(ValueError):
                v2.validate_qa_receipt(qa_path, manifest, protocol)


class ImmutableCacheTransactionFixture(unittest.TestCase):
    def test_two_stages_keep_manifest_identical_and_stage1_never_reads_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.json"
            raw = root / "raw"
            raw.mkdir()
            source = {"num_classes": 6, "label_ids": list(range(6)),
                      "raw_label_to_index": {str(i): i for i in range(6)},
                      "patient_splits": {"train": ["train"], "validation": ["validation"], "test": ["test"]},
                      "audit_sha256": "1" * 64, "records": []}
            geometry = {"shape_xyz": [2, 2, 2], "affine_xyz": np.eye(4).tolist()}
            for split, count in [("train", 143), ("validation", 38), ("test", 30)]:
                for index in range(count):
                    record = {"patient": split, "visit": str(index), "split": split,
                              "source_image": f"{split}_{index}_image.nii", "source_mask": f"{split}_{index}_mask.nii"}
                    record.update({key: copy.deepcopy(geometry) for key in
                                   ("original_image", "original_mask", "canonical_image", "canonical_mask")})
                    # Deliberately stale old-grid counts must not be copied.
                    record["class_voxel_counts"] = [8, 0, 0, 0, 0, 0]
                    source["records"].append(record)
                    for role in ("source_image", "source_mask"):
                        (raw / record[role]).write_bytes(b"fixture source identity")
            v2.write_json(original, source)
            plan_path = root / "protocol.json"
            plan = {"experiment_id": "fixture", "configuration": copy.deepcopy(v2.DEFAULT_CONFIGURATION),
                    "required_original_sha256": {"artifacts/cache64/manifest.json": v2.sha256(original)}}
            v2.write_json(plan_path, plan)
            plan = v2.load_protocol(plan_path)
            opened = []

            def preprocess(image, mask, shape, labels):
                opened.append(Path(image).name)
                record = next(r for r in source["records"] if r["source_image"] == Path(image).name)
                metadata = v2.expected_geometry(record, shape)
                metadata["class_voxel_counts"] = [int(np.prod(shape)), 0, 0, 0, 0, 0]
                return {}, metadata

            def write_fixture_cache(destination, arrays):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"transaction fixture, not a real cache")
                return v2.sha256(destination)

            dependencies = SimpleNamespace(dataset=SimpleNamespace(load_manifest=v2.read_json),
                prepare=SimpleNamespace(preprocess_pair=preprocess),
                utils=SimpleNamespace(safe_case_name=lambda p, v: p + "_" + v),
                source_hashes={"tools/prepare_cache.py": "a" * 64})
            with patch.object(v2, "_write_cache", write_fixture_cache), redirect_stdout(StringIO()):
                result = v2.prepare(original, root / "new", raw, plan, plan_path, dependencies)
                self.assertEqual(len(opened), 181)
                self.assertFalse(any(name.startswith("test_") for name in opened))
                manifest = v2.load_new_manifest(result, plan, dependencies)
                test_records = [r for r in manifest["records"] if r["split"] == "test"]
                self.assertTrue(all(r["pending"] and "class_voxel_counts" not in r for r in test_records))
                digest = v2.sha256(result)
                with self.assertRaises(ValueError):
                    v2.prepare_test(result, raw, plan, dependencies)
                freeze_path = root / "selection_freeze.json"
                freeze = {"status": "frozen", "manifest_sha256": digest, "protocol_sha256": plan["_sha256"],
                          "models": {name: {"completed_epochs": 30, "checkpoint_sha256": "a" * 64} for name in v2.MODELS}}
                v2.write_json(freeze_path, freeze)
                v2.prepare_test(result, raw, plan, dependencies, freeze_path)
                self.assertEqual(v2.sha256(result), digest)
                self.assertEqual(len(opened), 211)
                receipt = v2.read_json(result.parent / "test_completion_receipt.json")
                self.assertEqual(receipt["status"], "completed")
                self.assertEqual(len(receipt["records"]), 30)
                with self.assertRaises(FileExistsError):
                    v2.prepare_test(result, raw, plan, dependencies, freeze_path)


@unittest.skipIf(v2.torch is None, "Requires real Torch execution environment")
class TorchFixtures(unittest.TestCase):
    def test_sliding_prediction_matches_pointwise_rule_including_boundaries(self):
        torch = v2.torch

        class Pointwise(torch.nn.Module):
            def forward(self, image):
                return {"logits": torch.cat([-image, image], dim=1)}

        config = dict(v2.DEFAULT_CONFIGURATION, patch_shape_dhw=[4, 4, 4], num_classes=2, amp=False)
        image = torch.linspace(-2, 2, 7 * 9 * 11).reshape(1, 7, 9, 11)
        prediction = v2.predict_image(Pointwise(), image, "improved3d", torch.device("cpu"), config)
        torch.testing.assert_close(prediction, (image[0] > 0).long())
        self.assertEqual(prediction.shape, (7, 9, 11))

    def test_averages_probabilities_not_votes_or_logits(self):
        torch = v2.torch

        class PatchDependent(torch.nn.Module):
            def __init__(self, first_delta, remaining_delta):
                super().__init__()
                self.first_delta, self.remaining_delta = first_delta, remaining_delta

            def forward(self, image):
                start = int(image[0, 0, 0, 0, 0])
                delta = self.first_delta if start == 0 else self.remaining_delta
                logits = torch.zeros((1, 2, *image.shape[2:]))
                logits[:, 1] = delta
                return {"logits": logits}

        config = dict(v2.DEFAULT_CONFIGURATION, patch_shape_dhw=[4, 4, 4], num_classes=2, amp=False)
        # starts [0,2,3]; voxel z3 belongs to three windows.
        image = torch.arange(7.).reshape(1, 7, 1, 1).expand(1, 7, 4, 4).clone()
        # First scenario: probabilities select 1 while majority vote selects 0.
        # Second scenario: probabilities select 1 while averaged logits select 0.
        for first, remaining in [(4., -1.), (-4., 1.5)]:
            prediction = v2.predict_image(PatchDependent(first, remaining), image, "improved3d", torch.device("cpu"), config)
            self.assertEqual(int(prediction[3, 0, 0]), 1)

    def test_frozen_deep_supervision_heads_receive_nonzero_gradients(self):
        torch = v2.torch
        project = Path(os.environ.get("HIPMRI_V1_PROJECT_ROOT",
            str(HERE.parent / "PatternAnalysis-2026/recognition/ImprovedUNet3D_s4927782")))
        dependencies = v2.load_dependencies(project)
        model = dependencies.modules.ImprovedUNet3D(num_classes=6, base_channels=2)
        torch.manual_seed(3711)
        image = torch.randn(1, 1, 8, 16, 16)
        target = torch.randint(0, 6, (1, 8, 16, 16))
        loss = dependencies.modules.segmentation_loss(model(image), target)
        loss.backward()
        for head in model.aux_heads:
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(float(head.weight.grad.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
