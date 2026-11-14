"""Real-patient distance calibration and private generated/neighbor panels."""

import json
from pathlib import Path

import torch
from audit_cohort import ADNISliceDataset, patient_selection, summary
from PIL import Image, ImageDraw, ImageOps


def main():
    torch.set_num_threads(2)
    output = Path("outputs/calibration-20261006")
    output.mkdir(parents=True, exist_ok=False)
    root = Path("/home/groups/comp3710/ADNI")
    datasets = {
        s: ADNISliceDataset(root, Path("data/manifest-seed3710.csv"), s, 64)
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
    selected = {s: patient_selection(d.rows, counts) for s, d in datasets.items()}
    vectors = {
        s: torch.stack(
            [(d[i]["pixel_values"][0].flatten() + 1) / 2 for i in range(len(d))]
        )
        for s, d in datasets.items()
    }
    report = {
        "metric": "grayscale64 pixel RMSE",
        "matched_counts": counts,
        "self_patient_excluded": True,
        "calibration": {},
    }
    for split in datasets:
        query = vectors[split][selected[split]]
        report["calibration"][f"{split}_matched_pairwise"] = summary(
            torch.pdist(query) / 64
        )
        for scope, indices in (
            ("full", list(range(len(datasets["train"])))),
            ("matched", selected["train"]),
        ):
            distance = (
                torch.cdist(
                    query,
                    vectors["train"][indices],
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
                / 64
            )
            for i, query_index in enumerate(selected[split]):
                patient = datasets[split].rows[query_index]["patient_id"]
                exclude = torch.tensor(
                    [
                        datasets["train"].rows[j]["patient_id"] == patient
                        for j in indices
                    ]
                )
                distance[i, exclude] = float("inf")
            report["calibration"][f"{split}_to_train_{scope}"] = summary(
                distance.min(1).values
            )
    audit = json.loads(
        Path("outputs/cohort-632557/audit/private-neighbors.json").read_text()
    )
    adapted = audit[1]
    baseline = json.loads(
        Path("outputs/vae-audit-635709/private-neighbors.json").read_text()
    )[0]
    nearest = min(
        (r for r in adapted["individual"] if r["reference_set"] == "train_full"),
        key=lambda r: r["rmse"],
    )["sample"]
    cases = [
        (
            "F1-background",
            "official-256-cfg7.5-AD-37220.png",
            adapted,
            "gray background",
        ),
        ("F2-color", "official-256-cfg7.5-NC-37217.png", adapted, "colored patch"),
        ("F3-closest", nearest, adapted, "closest training neighbor; risk probe"),
        ("F4-blur", "vae-AD-37200.png", baseline, "blurred baseline"),
        (
            "F5-repeat",
            "official-256-cfg7.5-AD-37201.png",
            adapted,
            "similar gross shape across seeds; hypothesis",
        ),
    ]
    private = []
    for case_id, sample, source, description in cases:
        matches = [
            next(
                r
                for r in source["individual"]
                if r["sample"] == sample and r["reference_set"] == split
            )
            for split in ("train_full", "validation_full")
        ]
        canvas = Image.new("RGB", (768, 300), "white")
        draw = ImageDraw.Draw(canvas)
        with Image.open(Path(source["cohort"]) / sample) as image:
            canvas.paste(image.convert("RGB").resize((256, 256)), (0, 0))
        draw.text((4, 260), case_id + " generated", fill="black")
        for col, match in enumerate(matches, 1):
            with Image.open(root / match["reference"]["relative_path"]) as image:
                image = image.convert("L")
                side = max(image.size)
                image = ImageOps.pad(image, (side, side), color=0).resize(
                    (256, 256), Image.Resampling.BILINEAR
                )
                canvas.paste(image.convert("RGB"), (col * 256, 0))
            draw.text(
                (col * 256 + 4, 260),
                f"{match['reference_set']} RMSE {match['rmse']:.4f}",
                fill="black",
            )
        canvas.save(output / f"{case_id}.png")
        private.append(
            {
                "id": case_id,
                "sample": sample,
                "description": description,
                "neighbors": matches,
            }
        )
    (output / "private-cases.json").write_text(json.dumps(private, indent=2) + "\n")
    report["cases"] = [
        {
            "id": c["id"],
            "sample": c["sample"],
            "description": c["description"],
            "train_rmse": c["neighbors"][0]["rmse"],
            "validation_rmse": c["neighbors"][1]["rmse"],
        }
        for c in private
    ]
    (output / "aggregate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
