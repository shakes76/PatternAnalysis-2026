"""
Inference driver: images in, boxes and routes out. Needs no masks or labels.

Usage:
    uv run predict --checkpoint runs/<run>/weights/best.pt --images <dir> --output <dir>
"""

import argparse
import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from yolo_lesion_47041585.dataset import NormalizedBox
from yolo_lesion_47041585.helpers.visualisation import draw_detections
from yolo_lesion_47041585.modules import LesionPredictor, validate_model_contract

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Detection:
    """A predicted box in the original image's frame.

    Attributes:
        image_id (str): Image identifier (file stem).
        box (NormalizedBox): The box, normalised by the original image size.
        class_id (int): 0 Benign, 1 Malignant.
        confidence (float): Class confidence.
    """

    image_id: str
    box: NormalizedBox
    class_id: int
    confidence: float


def detections_from_result(result: Results) -> list[Detection]:
    """Convert one Ultralytics `Results` into Detections in the original image's frame.

    Ultralytics has already undone the resize and padding, and `boxes.xywhn` is
    normalised by the original image size, so no further geometry is needed here.

    Args:
        result (ultralytics.engine.results.Results): Prediction for a single image.

    Returns:
        list[Detection]: One entry per predicted box, with the image ID taken from the
            file name.
    """
    image_id = Path(result.path).stem
    boxes = result.boxes
    if boxes is None:
        return []

    # tolist() is untyped in torch, cast the expected types
    xywhn = cast(list[list[float]], boxes.xywhn.tolist())  # pyright: ignore[reportUnknownMemberType]
    class_ids = cast(list[float], boxes.cls.tolist())  # pyright: ignore[reportUnknownMemberType]
    confidences = cast(list[float], boxes.conf.tolist())  # pyright: ignore[reportUnknownMemberType]
    return [
        Detection(image_id, NormalizedBox(x, y, w, h), int(class_id), conf)
        for (x, y, w, h), class_id, conf in zip(xywhn, class_ids, confidences)
    ]


@dataclass(frozen=True)
class InferenceSettings:
    """How a run's model is applied to images, taken from the run's saved config.

    Attributes:
        class_names (tuple[str, ...]): Class names in class ID order.
        imgsz (int): Square input size the image is resized and padded to.
        conf (float): Confidence floor below which boxes are dropped.
        iou (float): IoU above which class-aware NMS suppresses a box.
        max_det (int): Maximum number of boxes kept per image.
    """

    class_names: tuple[str, ...]
    imgsz: int
    conf: float
    iou: float
    max_det: int


def load_run(checkpoint: Path) -> tuple[YOLO, InferenceSettings]:
    """Load a trained model and the inference settings saved with its run.

    Args:
        checkpoint (Path): A run's weights, such as `runs/<run>/weights/best.pt`. The
            run's `config.resolved.yaml` is expected two folders above it.

    Returns:
        tuple[YOLO, InferenceSettings]: The model and how to apply it.

    Raises:
        FileNotFoundError: If the checkpoint or the run's saved config is missing.
        ValueError: If the model is not a detector with the config's classes.
    """
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No checkpoint at {checkpoint}")

    config_path = checkpoint.resolve().parents[1] / "config.resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"No saved run config at {config_path}")

    config = yaml.safe_load(config_path.read_text())
    settings = InferenceSettings(
        class_names=tuple(config["classes"]),
        imgsz=int(config["train"]["imgsz"]),
        conf=float(config["inference"]["conf"]),
        iou=float(config["inference"]["iou"]),
        max_det=int(config["inference"]["max_det"]),
    )
    model = YOLO(str(checkpoint))
    validate_model_contract(model, settings.class_names)

    return model, settings


def predict_cases(
    model: YOLO,
    images: Sequence[Path],
    settings: InferenceSettings,
    device: str | None = None,
) -> list[Detection]:
    """Detect lesions, undoing resize/padding so boxes are normalised by the original
    image size. This is the single inference path shared by validation, test and demo.

    Args:
        model (YOLO): A model loaded with `load_run`.
        images (Sequence[Path]): Image files. No masks or labels are read.
        settings (InferenceSettings): The run's inference settings.
        device (str | None): Ultralytics device, or None to let it choose.

    Returns:
        list[Detection]: Every box kept, across all images, highest confidence first
            within each image.
    """
    # stream=True yields results one at a time instead of holding them all in memory
    results = cast(
        Iterable[Results],
        model.predict(  # pyright: ignore[reportUnknownMemberType]
            [str(image) for image in images],
            imgsz=settings.imgsz,
            conf=settings.conf,
            iou=settings.iou,
            max_det=settings.max_det,
            device=device,
            stream=True,
            verbose=False,
            predictor=LesionPredictor,
        ),
    )

    return [d for result in results for d in detections_from_result(result)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="folder of images")
    parser.add_argument(
        "--output", type=Path, required=True, help="new or empty folder"
    )
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu, mps or 0")
    parser.add_argument(
        "--overlay-conf",
        type=float,
        default=0.25,
        help="only draw boxes at or above this confidence; detections.csv keeps all",
    )
    args = parser.parse_args(argv)

    images = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No images in {args.images}")

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"{args.output} is not empty")

    model, settings = load_run(args.checkpoint)
    detections = predict_cases(model, images, settings, args.device)

    overlays = args.output / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    by_image: dict[str, list[Detection]] = {image.stem: [] for image in images}
    for detection in detections:
        by_image[detection.image_id].append(detection)

    for image_path in images:
        with Image.open(image_path) as image:
            shown = [
                d
                for d in by_image[image_path.stem]
                if d.confidence >= args.overlay_conf
            ]
            overlay = draw_detections(image, shown, settings.class_names)

        overlay.save(overlays / f"{image_path.stem}.jpg")

    with (args.output / "detections.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image_id",
                "x_center",
                "y_center",
                "width",
                "height",
                "class_id",
                "class_name",
                "confidence",
            ]
        )

        for d in detections:
            writer.writerow(
                [d.image_id, d.box.x_center, d.box.y_center, d.box.width, d.box.height,
                 d.class_id, settings.class_names[d.class_id], d.confidence]
            )  # fmt: skip

    print(f"{len(detections)} boxes on {len(images)} images, written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
