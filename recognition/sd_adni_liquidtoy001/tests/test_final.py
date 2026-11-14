"""CPU tests of the final-evaluation tools with tiny synthetic inputs.

The distribution and membership measures need SciPy (the analysis environment,
requirements-evaluation.txt); the model-side pieces need diffusers and peft
(the training environment). Each class skips when its library is missing, so
run the suite in both environments.
"""

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_common  # noqa: E402

HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_DIFFUSERS = importlib.util.find_spec("diffusers") is not None


def write_cohort(
    folder: Path, count: int = 4, size: int = 32, first: int = 37200
) -> list[dict]:
    folder.mkdir(parents=True)
    random = np.random.default_rng(0)
    records = []
    for i in range(count):
        label = ("AD", "NC")[i % 2]
        name = f"{label}-{first + i}.png"
        array = random.integers(0, 256, (size, size, 3), dtype=np.uint8)
        Image.fromarray(array).save(folder / name)
        digest = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        records.append(
            {"file": name, "label": label, "seed": first + i, "png_sha256": digest}
        )
    (folder / "report.json").write_text(json.dumps({"records": records}))
    return records


class RolesAndSamplingTest(unittest.TestCase):
    def test_the_rehearsal_never_uses_the_test_split(self):
        self.assertNotIn("test", final_common.roles("rehearsal").values())
        self.assertEqual(final_common.roles("test")["heldout"], "test")
        self.assertEqual(final_common.roles("test")["calibration"], "validation")
        with self.assertRaises(ValueError):
            final_common.roles("final")

    def test_balanced_sample_is_deterministic_and_balanced(self):
        labels = ["AD"] * 10 + ["NC"] * 7
        first = final_common.balanced_sample(labels, 5, seed=3710)
        self.assertEqual(first, final_common.balanced_sample(labels, 5, seed=3710))
        self.assertEqual(sum(labels[i] == "AD" for i in first), 5)
        self.assertEqual(len(set(first)), 10)
        with self.assertRaises(ValueError):
            final_common.balanced_sample(labels, 8, seed=3710)

    def test_subject_codes_replace_patient_ids(self):
        rows = [{"patient_id": p} for p in ("b", "a", "b", "c")]
        self.assertEqual(final_common.subject_codes(rows).tolist(), [1, 0, 1, 2])


class NearestTest(unittest.TestCase):
    def test_exact_copy_is_zero_and_own_subject_is_skipped(self):
        generator = torch.Generator().manual_seed(1)
        references = torch.rand((6, 16), generator=generator)
        distances, indices = final_common.nearest(references[:3], references, chunk=2)
        self.assertEqual(distances.max().item(), 0.0)
        self.assertEqual(indices.tolist(), [0, 1, 2])
        groups = torch.tensor([0, 0, 1, 1, 2, 2])
        distances, indices = final_common.nearest(
            references[:3], references, groups[:3], groups
        )
        self.assertTrue((distances > 0).all())
        for query, index in enumerate(indices.tolist()):
            self.assertNotEqual(groups[query], groups[index])

    def test_rmse_scale(self):
        query = torch.zeros((1, 4))
        reference = torch.full((1, 4), 0.5)
        distance, _ = final_common.nearest(query, reference)
        self.assertAlmostEqual(distance.item(), 0.5, places=6)


class CohortReadingTest(unittest.TestCase):
    def test_views_and_hash_check(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder) / "cohort"
            records = write_cohort(folder)
            self.assertEqual(len(final_common.cohort_records(folder)), 4)
            views = final_common.generated_views(folder / records[0]["file"])
            self.assertEqual(tuple(views["gray64"].shape), (4096,))
            self.assertEqual(tuple(views["rgb256"].shape), (3, 256, 256))
            self.assertEqual(tuple(views["rgb"].shape), (3, 32, 32))
            self.assertLessEqual(views["rgb256"].abs().max().item(), 1.0)
            Image.new("RGB", (32, 32)).save(folder / records[1]["file"])
            with self.assertRaises(ValueError):
                final_common.cohort_records(folder)

    def test_inception_input_size_and_normalisation(self):
        grey = torch.full((2, 3, 128, 128), 0.5)
        result = final_common.inception_input(grey)
        self.assertEqual(tuple(result.shape), (2, 3, 299, 299))
        expected = (0.5 - 0.485) / 0.229
        self.assertAlmostEqual(result[0, 0, 150, 150].item(), expected, places=5)


class MergeTest(unittest.TestCase):
    def test_chunks_join_with_hashes_and_the_first_grid(self):
        from merge_cohorts import merge

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_cohort(folder / "a", count=4, first=37200)
            write_cohort(folder / "b", count=4, first=37204)
            Image.new("RGB", (8, 8)).save(folder / "a" / "grid-cfg7.5.png")
            merged = merge([folder / "a", folder / "b"], folder / "merged")
            self.assertEqual(len(merged["records"]), 8)
            self.assertEqual(len(final_common.cohort_records(folder / "merged")), 8)
            self.assertTrue((folder / "merged" / "grid-cfg7.5.png").exists())
            with self.assertRaises(ValueError):
                merge([folder / "a", folder / "a"], folder / "twice")


@unittest.skipUnless(HAS_SCIPY, "needs SciPy (analysis environment)")
class DistributionMeasuresTest(unittest.TestCase):
    def setUp(self):
        import final_metrics

        self.m = final_metrics

    def test_frechet_distance_closed_form_for_diagonal_gaussians(self):
        mean_a, mean_b = np.array([0.0, 1.0]), np.array([2.0, 1.0])
        var_a, var_b = np.array([1.0, 4.0]), np.array([9.0, 1.0])
        value = self.m.frechet_distance(mean_a, np.diag(var_a), mean_b, np.diag(var_b))
        expected = 4.0 + np.sum((np.sqrt(var_a) - np.sqrt(var_b)) ** 2)
        self.assertAlmostEqual(value, expected, places=6)

    def test_fid_and_kid_separate_shifted_sets(self):
        random = np.random.default_rng(0)
        a = random.normal(size=(600, 8))
        b = random.normal(size=(600, 8))
        shifted = b + 1.0
        self.assertLess(self.m.fid(a, b), 0.2)
        self.assertGreater(self.m.fid(a, shifted), 7.0)
        same, apart = self.m.kid(a, b), self.m.kid(a, shifted)
        self.assertLess(abs(same["mean"]), 0.02)
        self.assertGreater(apart["mean"], 10 * abs(same["mean"]) + 0.05)

    def test_prdc_extremes(self):
        random = np.random.default_rng(1)
        real = random.normal(size=(200, 4))
        same = self.m.prdc(real, real.copy())
        for key in ("precision", "recall", "coverage"):
            self.assertEqual(same[key], 1.0)
        self.assertAlmostEqual(same["density"], 1.0 + 1.0 / self.m.PRDC_K, delta=0.25)
        far = self.m.prdc(real, real + 100)
        self.assertEqual(
            (far["precision"], far["recall"], far["density"], far["coverage"]),
            (0.0, 0.0, 0.0, 0.0),
        )

    def test_a_one_class_cohort_is_compared_with_that_class_only(self):
        random = np.random.default_rng(6)
        real = np.concatenate(
            [random.normal(size=(300, 4)), random.normal(size=(300, 4)) + 5]
        )
        real_is_nc = np.array([0] * 300 + [1] * 300, dtype=np.int8)
        fake = random.normal(size=(64, 4)) + 5  # like the NC slices
        block = self.m.distribution_block(
            fake, np.ones(64, dtype=np.int8), real, real_is_nc
        )
        self.assertEqual(block["classes"], ["NC"])
        self.assertLess(block["fid"], 1.0)
        self.assertGreater(block["prdc"]["coverage"]["mean"], 0.5)
        self.assertEqual(list(block["per_class_fid"]), ["NC"])

    def test_training_preference(self):
        random = np.random.default_rng(2)
        heldout = random.uniform(0.05, 0.1, 500)
        calibration_heldout = random.uniform(0.05, 0.1, 500)
        neutral = self.m.training_preference(
            heldout * 0.9, heldout, calibration_heldout * 0.9, calibration_heldout
        )
        self.assertAlmostEqual(neutral["difference"], 0.0, places=10)
        self.assertLess(neutral["difference_ci95"][0], 0.0)
        self.assertGreater(neutral["difference_ci95"][1], 0.0)
        copying = self.m.training_preference(
            heldout * 0.45, heldout, calibration_heldout * 0.9, calibration_heldout
        )
        self.assertAlmostEqual(copying["closer_to_training_percent"], 50.0, places=6)
        self.assertLess(copying["difference_ci95"][1], 0.0)

    def test_auc_and_tpr(self):
        members, nonmembers = np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])
        self.assertEqual(self.m.auc(members, nonmembers), 1.0)
        self.assertEqual(self.m.auc(nonmembers, members), 0.0)
        self.assertEqual(self.m.auc(np.ones(4), np.ones(5)), 0.5)
        # Two of the 4 x 4 pairs have the member lower; one more is a tie.
        self.assertEqual(
            self.m.auc(np.array([1, 2, 5, 6.0]), np.array([1.5, 2, 9, 0.5]) + 2.5),
            self.m.auc(np.array([1, 2, 5, 6.0]), np.array([4.0, 4.5, 11.5, 3.0])),
        )
        random = np.random.default_rng(3)
        nonmember = random.normal(size=10_000)
        self.assertAlmostEqual(self.m.tpr_at_fpr(nonmember, nonmember), 0.01, places=3)
        self.assertEqual(self.m.tpr_at_fpr(nonmember - 100, nonmember), 1.0)

    def test_bootstrap_is_seeded(self):
        values = np.random.default_rng(4).normal(size=50)
        self.assertEqual(
            self.m.bootstrap_mean_ci(values), self.m.bootstrap_mean_ci(values)
        )


@unittest.skipUnless(HAS_SCIPY, "needs SciPy (analysis environment)")
class BlindSortTest(unittest.TestCase):
    def test_build_hides_names_and_score_counts(self):
        import argparse

        import blind_sort

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            np.savez(
                folder / "real.npz",
                images=np.random.default_rng(5).integers(
                    0, 256, (32, 64, 64), dtype=np.uint8
                ),
                is_nc=np.array([0] * 16 + [1] * 16, dtype=np.int8),
            )
            write_cohort(folder / "cohort", count=40)
            blind_sort.build(
                argparse.Namespace(
                    real=folder / "real.npz",
                    cohort=folder / "cohort",
                    output=folder / "out",
                )
            )
            html = (folder / "out" / "private-blind-sort.html").read_text()
            key = json.loads((folder / "out" / "private-key.json").read_text())
            self.assertEqual(len(key), 64)
            self.assertEqual(sum(k["truth"] == "real" for k in key), 32)
            self.assertEqual(html.count("data:image/png;base64"), 64)
            self.assertNotIn(".png", html)
            answers = folder / "answers.json"
            answers.write_text(json.dumps([k["truth"] for k in key]))
            output = folder / "score.json"
            blind_sort.score(
                argparse.Namespace(
                    key=folder / "out" / "private-key.json",
                    answers=answers,
                    output=output,
                )
            )
            self.assertEqual(json.loads(output.read_text())["accuracy"], 1.0)

    def test_reference_page_shows_every_real_slice_by_class(self):
        import argparse

        import blind_sort

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            np.savez(
                folder / "reference.npz",
                images=np.zeros((16, 64, 64), dtype=np.uint8),
                is_nc=np.array([0] * 8 + [1] * 8, dtype=np.int8),
            )
            output = folder / "page" / "private-reference.html"
            blind_sort.reference(
                argparse.Namespace(real=folder / "reference.npz", output=output)
            )
            html = output.read_text()
            self.assertEqual(html.count("data:image/png;base64"), 16)
            self.assertEqual(html.count("real, AD"), 8)
            self.assertEqual(html.count("real, NC"), 8)
            self.assertNotIn("synthetic", html)


@unittest.skipUnless(HAS_DIFFUSERS, "needs diffusers and peft (training environment)")
class ModelSideTest(unittest.TestCase):
    def test_e0_adapter_export_round_trip_keeps_float32(self):
        from export_e0_adapter import export
        from modules import add_lora, load_weights
        from test_modules import tiny_unet

        trained = tiny_unet()
        add_lora(trained, 4)
        with torch.no_grad():
            for name, parameter in trained.named_parameters():
                if "lora_" in name:
                    parameter.normal_()
        adapter = {
            n: p.detach().clone() for n, p in trained.named_parameters() if "lora_" in n
        }
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "checkpoint.pt"
            torch.save({"adapter": adapter, "step": 20000}, checkpoint)
            output = Path(folder) / "e0.safetensors"
            report = export(checkpoint, output)
            self.assertEqual(report["step"], 20000)
            fresh = tiny_unet()
            self.assertEqual(load_weights(fresh, output), "lora")
            restored = dict(fresh.named_parameters())
            for name, value in adapter.items():
                self.assertTrue(torch.equal(restored[name], value))

    def test_membership_noise_is_fixed_and_rows_exclude_defective_scans(self):
        from membership import fixed_noise, member_rows

        rows = [{"scan_id": s} for s in ("1", "2", "2", "3")]
        self.assertEqual(member_rows(rows, {"2"}), [0, 3])
        cpu = torch.device("cpu")
        a = fixed_noise("member", 7, 1, (1, 4, 4, 4), cpu)
        self.assertTrue(torch.equal(a, fixed_noise("member", 7, 1, (1, 4, 4, 4), cpu)))
        for other in (
            fixed_noise("nonmember", 7, 1, (1, 4, 4, 4), cpu),
            fixed_noise("member", 8, 1, (1, 4, 4, 4), cpu),
            fixed_noise("member", 7, 2, (1, 4, 4, 4), cpu),
        ):
            self.assertFalse(torch.equal(a, other))

    def test_per_image_loss_matches_the_training_loss_on_average(self):
        from membership import per_image_loss
        from modules import diffusion_loss
        from test_modules import sd_schedule, tiny_embeddings, tiny_unet

        unet, schedule = tiny_unet().eval(), sd_schedule()
        generator = torch.Generator().manual_seed(8)
        latents = torch.randn((3, 4, 8, 8), generator=generator)
        noise = torch.randn((3, 4, 8, 8), generator=generator)
        conditioning = tiny_embeddings()["AD"].expand(3, -1, -1)
        each = per_image_loss(unet, latents, conditioning, schedule, noise, 200)
        with torch.no_grad():
            mean = diffusion_loss(
                unet, latents, conditioning, schedule, noise, torch.full((3,), 200)
            )
        self.assertEqual(tuple(each.shape), (3,))
        self.assertAlmostEqual(each.mean().item(), mean.item(), places=5)

    def test_vae_cohort_extends_without_changing_the_first_seeds(self):
        from baseline_vae import SmallVAE, generate

        torch.manual_seed(0)
        model = SmallVAE()
        with tempfile.TemporaryDirectory() as folder:
            short, long = Path(folder) / "short", Path(folder) / "long"
            short.mkdir()
            long.mkdir()
            first = generate(model, short, count=2, device="cpu")
            extended = generate(model, long, count=3, device="cpu")
            self.assertEqual(len(extended), 6)
            hashes = {r["file"]: r["png_sha256"] for r in extended}
            for record in first:
                self.assertEqual(hashes[record["file"]], record["png_sha256"])


if __name__ == "__main__":
    unittest.main()
