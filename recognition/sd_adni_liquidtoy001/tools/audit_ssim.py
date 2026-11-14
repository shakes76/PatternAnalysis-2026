"""Explicit SSIM pairing for generated images; neither pair is ground truth."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_appearance import verify
from PIL import Image
from skimage.metrics import structural_similarity


def score(a, b):
    return float(
        structural_similarity(
            a,
            b,
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def summarize(values):
    x = np.array(values)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
        "fraction_above_0_6": float((x > 0.6).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("outputs/features-639765")
    )
    parser.add_argument(
        "--references", type=Path, default=Path("outputs/encoded-639777")
    )
    parser.add_argument("--output", type=Path, help="default: the references folder")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="additional generated cohort folder containing report.json",
    )
    args = parser.parse_args()
    output = args.output or args.references
    output.mkdir(parents=True, exist_ok=True)
    refs = np.load(args.references / "private-reference-images.npz")
    images = {
        int(i): x.astype(np.float64) / 255
        for i, x in zip(refs["indices"], refs["images"])
    }
    features = np.load(args.features / "private-features.npz")["features"]
    rows = json.loads((args.features / "private-rows.json").read_text())
    # Reference images are indexed by row; a newer feature run must keep the
    # same real-image rows in the same order for those indices to stay valid.
    reference_rows = json.loads(
        Path("outputs/features-639765/private-rows.json").read_text()
    )
    for index in refs["indices"]:
        if rows[int(index)]["file"] != reference_rows[int(index)]["file"]:
            raise ValueError("Feature rows no longer align with reference images")
    # Actual reference identity control; do not report it as model quality.
    if abs(score(images[0], images[0]) - 1) > 1e-12:
        raise ValueError("SSIM identity control failed")
    private = []
    groups = [
        ("base", "outputs/cohort-632557/base"),
        ("adapted", "outputs/cohort-632557/adapted"),
        ("baseline", "outputs/baseline-vae-632570"),
    ] + [tuple(item.split("=", 1)) for item in args.extra]
    for group, folder in groups:
        folder = Path(folder)
        records = json.loads((folder / "report.json").read_text())["records"]
        for r in records:
            verify(folder / r["file"], r["png_sha256"])
            with Image.open(folder / r["file"]) as im:
                image = (
                    np.asarray(
                        im.convert("L").resize((256, 256), Image.Resampling.BILINEAR),
                        dtype=np.float64,
                    )
                    / 255
                )
            qi = next(
                i
                for i, row in enumerate(rows)
                if row["group"] == group and row["file"] == r["file"]
            )
            candidates = [
                i
                for i, row in enumerate(rows)
                if row["group"] == "validation" and row["label"] == r["label"]
            ]
            nearest = candidates[
                int(
                    np.argmin(
                        np.mean((features[candidates] - features[qi]) ** 2, axis=1)
                    )
                )
            ]
            choice = int(
                hashlib.sha256(f"{r['label']}:{r['seed']}".encode()).hexdigest(), 16
            ) % len(candidates)
            fixed = candidates[choice]
            private.append(
                {
                    "group": group,
                    "label": r["label"],
                    "file": r["file"],
                    "nearest_index": nearest,
                    "fixed_index": fixed,
                    "nearest_ssim": score(image, images[nearest]),
                    "fixed_ssim": score(image, images[fixed]),
                }
            )
    real = []
    for qi, row in enumerate(rows):
        if row["group"] != "validation":
            continue
        candidates = [
            i
            for i, r in enumerate(rows)
            if r["group"] == "train" and r["label"] == row["label"]
        ]
        ni = candidates[
            int(np.argmin(np.mean((features[candidates] - features[qi]) ** 2, axis=1)))
        ]
        real.append(score(images[qi], images[ni]))
    report = {
        "metric": "SSIM256 grayscale gaussian sigma1.5 population covariance",
        "real_validation_to_training_feature_neighbor": summarize(real),
        "groups": {},
        "identity_control": True,
        "limits": "Unpaired generated cohorts; nearest selection optimistic; fixed reference arbitrary; neither score is clinical validation",
    }
    for group, _ in groups:
        report["groups"][group] = {}
        for label in ("all", "AD", "NC"):
            records = [
                r
                for r in private
                if r["group"] == group and label in ("all", r["label"])
            ]
            report["groups"][group][label] = {
                metric: summarize([r[metric] for r in records])
                for metric in ("nearest_ssim", "fixed_ssim")
            }
    (output / "ssim.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "private-ssim.json").write_text(json.dumps(private, indent=2) + "\n")
    print(json.dumps({g: v["all"] for g, v in report["groups"].items()}, indent=2))
    print("real reference", report["real_validation_to_training_feature_neighbor"])


if __name__ == "__main__":
    main()
