"""Predict from trained weights and export original-coordinate boxes."""

import argparse
import hashlib
from pathlib import Path

from modules import load_trained_detector
from utils import save_json


def predict_images(
    weights: Path,
    source: Path,
    output_dir: Path,
    image_size: int,
    confidence: float,
    device: str,
) -> list[dict]:
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1")
    if image_size < 32 or image_size % 32:
        raise ValueError("Image size must be a positive multiple of 32")
    if not Path(source).exists():
        raise FileNotFoundError(source)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    detector = load_trained_detector(weights)
    output_dir.mkdir(parents=True)
    predictions = []
    results = detector.predict(source=str(source), imgsz=image_size, conf=confidence,
                               iou=0.7, device=device, augment=False,
                               save=False, stream=True, verbose=False)
    for result in results:
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append({"class_id": class_id, "class_name": result.names[class_id],
                                   "confidence": float(box.conf.item()),
                                   "xyxy": box.xyxy[0].cpu().tolist()})
        source_path = str(Path(result.path).resolve())
        suffix = hashlib.sha256(source_path.encode()).hexdigest()[:10]
        result.save(filename=str(output_dir / f"{Path(result.path).stem}_{suffix}.jpg"))
        predictions.append({"image": source_path, "image_id": Path(result.path).stem,
                            "height": result.orig_shape[0], "width": result.orig_shape[1],
                            "detections": detections})
    if not predictions:
        raise ValueError("No input images were processed")
    save_json(output_dir / "predictions.json", predictions)
    save_json(output_dir / "settings.json", {"weights": str(weights), "imgsz": image_size,
                                             "confidence": confidence, "nms_iou": 0.7})
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    rows = predict_images(args.weights, args.source, args.output, args.imgsz, args.conf, args.device)
    print(f"Processed {len(rows)} images")


if __name__ == "__main__":
    main()
