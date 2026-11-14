"""Descriptive appearance diagnostics; neither anatomical scores nor pass gates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_cohort import ADNISliceDataset, patient_selection
from PIL import Image, ImageOps


def measure(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    gray = (
        np.asarray(
            image.convert("L").resize((128, 128), Image.Resampling.BILINEAR),
            dtype=np.float64,
        )
        / 255
    )
    border = np.ones(gray.shape, dtype=bool)
    border[8:-8, 8:-8] = False
    return {
        "color": float(np.ptp(rgb, axis=2).mean()),
        "border": float(gray[border].mean()),
        "edge": float(
            (
                np.abs(np.diff(gray, axis=0)).mean()
                + np.abs(np.diff(gray, axis=1)).mean()
            )
            / 2
        ),
    }


def summarize(records):
    result = {}
    for label in ("all", "AD", "NC"):
        group = [r for r in records if label == "all" or r["label"] == label]
        if not group:
            continue
        result[label] = {"n": len(group), "metrics": {}}
        for metric in ("color", "border", "edge"):
            values = np.array([r[metric] for r in group])
            result[label]["metrics"][metric] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "q05": float(np.quantile(values, 0.05)),
                "q95": float(np.quantile(values, 0.95)),
            }
    return result


def verify(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Source hash changed: {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--baseline", type=Path, default=Path("outputs/baseline-vae-632570")
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="additional generated cohort folder containing report.json",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "protocol": "appearance-v1",
        "test_images_read": False,
        "real_reference_run": args.real,
        "groups": {},
    }
    private = {}
    for name, root in [
        ("base", "outputs/cohort-632557/base"),
        ("adapted", "outputs/cohort-632557/adapted"),
        ("vae", args.baseline),
    ] + [tuple(item.split("=", 1)) for item in args.extra]:
        root = Path(root)
        source = json.loads((root / "report.json").read_text())
        records = []
        for row in source["records"]:
            path = root / row["file"]
            verify(path, row["png_sha256"])
            with Image.open(path) as image:
                records.append(
                    {"file": row["file"], "label": row["label"], **measure(image)}
                )
        private[name] = records
        report["groups"][name] = summarize(records)
    if args.real:
        root = Path("/home/groups/comp3710/ADNI")
        dataset = ADNISliceDataset(
            root, Path("data/manifest-seed3710.csv"), "validation", 128
        )
        counts = {
            c: len({r["patient_id"] for r in dataset.rows if r["class_name"] == c})
            for c in ("AD", "NC")
        }
        records = []
        for index in patient_selection(dataset.rows, counts):
            row = dataset.rows[index]
            path = root / row["relative_path"]
            verify(path, row["sha256"])
            with Image.open(path) as source:
                image = source.convert("L")
                w, h = image.size
                side = max(w, h)
                left, top = (side - w) // 2, (side - h) // 2
                image = ImageOps.expand(
                    image, border=(left, top, side - w - left, side - h - top), fill=0
                )
                records.append(
                    {
                        "file": row["relative_path"],
                        "label": row["class_name"],
                        **measure(image),
                    }
                )
        private["validation"] = records
        report["groups"]["validation"] = summarize(records)
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / "aggregate.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "private-records.json").write_text(
        json.dumps(private, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
