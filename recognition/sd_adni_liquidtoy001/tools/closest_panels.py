"""Private side-by-side panels of the generated images closest to training slices.

For the visual part of the memorisation audit in docs/e1-protocol.md: take the
k generated images with the smallest grayscale64 RMSE to any training slice
(from audit_cohort.py's private-neighbors.json) and place each beside that
slice. Panels contain real patient images and stay private.

    python tools/closest_panels.py outputs/e1-neighbours-<job>/private-neighbors.json \
        --output outputs/e1-closest-<job> --count 5
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path("/home/groups/comp3710/ADNI")
SIZE = 256


def real_slice(relative: str) -> Image.Image:
    """Load a training slice with the loader's square black padding."""
    with Image.open(ROOT / relative) as source:
        image = source.convert("L")
    width, height = image.size
    side = max(width, height)
    left, top = (side - width) // 2, (side - height) // 2
    padded = ImageOps.expand(
        image, border=(left, top, side - width - left, side - height - top), fill=0
    )
    return padded.resize((SIZE, SIZE), Image.Resampling.BILINEAR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("neighbors", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for report in json.loads(args.neighbors.read_text()):
        cohort = Path(report["cohort"])
        closest = sorted(
            (r for r in report["individual"] if r["reference_set"] == "train_full"),
            key=lambda r: r["rmse"],
        )[: args.count]
        panel = Image.new("L", (2 * SIZE, len(closest) * (SIZE + 20)), 255)
        draw = ImageDraw.Draw(panel)
        for row, record in enumerate(closest):
            with Image.open(cohort / record["sample"]) as generated:
                panel.paste(
                    generated.convert("L").resize((SIZE, SIZE)), (0, row * (SIZE + 20))
                )
            panel.paste(
                real_slice(record["reference"]["relative_path"]),
                (SIZE, row * (SIZE + 20)),
            )
            draw.text(
                (4, row * (SIZE + 20) + SIZE + 4),
                f"{record['sample']}  |  nearest training slice, RMSE {record['rmse']:.4f}",
                fill=0,
            )
        name = f"closest-{cohort.name}.png"
        panel.save(args.output / name)
        summary.append(
            {
                "cohort": str(cohort),
                "panel": name,
                "pairs": [{"sample": r["sample"], "rmse": r["rmse"]} for r in closest],
            }
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
