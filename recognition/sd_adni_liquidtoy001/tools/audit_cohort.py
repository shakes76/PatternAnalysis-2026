"""Private low-resolution nearest-neighbor and diversity diagnostics, not privacy proof."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import ADNISliceDataset


def nearest(query, references):
    # Direct cdist avoids negative cancellation artifacts for exact-copy controls.
    distances = (
        torch.cdist(query, references, compute_mode="donot_use_mm_for_euclid_dist")
        / query.shape[1] ** 0.5
    )
    return distances.min(dim=1)


def summary(values):
    return {
        "mean": values.mean().item(),
        "median": values.median().item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }


def patient_selection(rows, counts):
    selected = []
    for label, count in counts.items():
        patients = sorted(
            {r["patient_id"] for r in rows if r["class_name"] == label},
            key=lambda p: hashlib.sha256(f"3710:{p}".encode()).hexdigest(),
        )[:count]
        for patient in patients:
            indices = sorted(
                [i for i, r in enumerate(rows) if r["patient_id"] == patient],
                key=lambda i: (int(rows[i]["scan_id"]), int(rows[i]["slice_index"])),
            )
            selected.append(indices[len(indices) // 2])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohorts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    datasets = {
        s: ADNISliceDataset(
            Path("/home/groups/comp3710/ADNI"),
            Path("data/manifest-seed3710.csv"),
            s,
            64,
        )
        for s in ("train", "validation")
    }
    counts = {
        c: len(
            {
                r["patient_id"]
                for r in datasets["validation"].rows
                if r["class_name"] == c
            }
        )
        for c in ("AD", "NC")
    }
    references = {}
    for split, dataset in datasets.items():
        vectors = []
        for i in range(len(dataset)):
            item = dataset[i]
            if (
                hashlib.sha256(
                    (dataset.root / item["relative_path"]).read_bytes()
                ).hexdigest()
                != dataset.rows[i]["sha256"]
            ):
                raise ValueError("Reference source hash changed")
            vectors.append((item["pixel_values"][0].flatten() + 1) / 2)
        full = torch.stack(vectors)
        indices = patient_selection(dataset.rows, counts)
        references[split] = (full, indices)
        del vectors
        print(f"Loaded {split}: {len(full)} images", flush=True)
    control, _ = nearest(references["train"][0][:2], references["train"][0])
    if control.max() > 1e-7:
        raise AssertionError("Exact reference-copy positive control failed")
    reports = []
    for cohort in args.cohorts:
        manifest = json.loads((cohort / "report.json").read_text())
        samples = []
        for record in manifest["records"]:
            path = cohort / record["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["png_sha256"]:
                raise ValueError("Generated image hash mismatch")
            with Image.open(path) as image:
                array = (
                    np.asarray(
                        image.convert("L").resize((64, 64), Image.Resampling.BILINEAR),
                        dtype=np.float32,
                    )
                    / 255
                )
            samples.append(torch.from_numpy(array.copy()).flatten())
        query = torch.stack(samples)
        report = {
            "cohort": str(cohort),
            "samples": len(samples),
            "metric": "grayscale64 pixel RMSE, not a semantic feature metric",
            "matched_patients_per_class": counts,
            "neighbors": {},
            "individual": [],
        }
        records = []
        for split, (full, indices) in references.items():
            for scope, reference in (
                ("full", full),
                ("patient_matched", full[indices]),
            ):
                distances, index = nearest(query, reference)
                key = f"{split}_{scope}"
                report["neighbors"][key] = {
                    "reference_count": len(reference),
                    **summary(distances),
                }
                for i, (distance, idx) in enumerate(
                    zip(distances.tolist(), index.tolist())
                ):
                    actual = idx if scope == "full" else indices[idx]
                    records.append(
                        {
                            "sample": manifest["records"][i]["file"],
                            "reference_set": key,
                            "rmse": distance,
                            "reference": datasets[split].rows[actual],
                        }
                    )
        pairs = torch.pdist(query) / query.shape[1] ** 0.5
        report["pairwise_generated_rmse"] = summary(pairs)
        for label in ("AD", "NC"):
            subset = query[
                [
                    i
                    for i, r in enumerate(manifest["records"])
                    if r.get("label") == label
                ]
            ]
            if len(subset) > 1:
                report[f"{label}_pairwise_rmse"] = summary(
                    torch.pdist(subset) / query.shape[1] ** 0.5
                )
        report["individual"] = records
        reports.append(report)
    (args.output / "private-neighbors.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    for report in reports:
        del report["individual"]
    (args.output / "aggregate.json").write_text(
        json.dumps(
            {
                "reports": reports,
                "exact_copy_control_passed": True,
                "limits": [
                    "Unequal full reference sizes bias nearest distances",
                    "Black background and alignment affect pixel RMSE",
                    "No privacy or anatomical validity guarantee",
                    "Validation only; test sealed",
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
