"""YOLOv8 adaptation. Network and loss implementation: Ultralytics."""

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLASS_NAMES = ("melanoma", "seborrheic_keratosis", "nevus")


@dataclass(frozen=True)
class DetectorConfig:
    weights: str = "yolov8n.pt"
    class_names: tuple[str, ...] = CLASS_NAMES
    seed: int = 42

    def __post_init__(self):
        if self.class_names != CLASS_NAMES:
            raise ValueError("Class order must match the project dataset")
        if self.seed < 0:
            raise ValueError("Seed must be nonnegative")


def build_detector(config: DetectorConfig) -> Any:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds(config.seed, deterministic=True)
    detector = YOLO(config.weights, task="detect")
    architecture = detector.model.yaml
    nano = architecture.get('scale') == 'n' or (
        architecture.get('width_multiple') == 0.25 and architecture.get('depth_multiple') == 0.33)
    if detector.task != "detect" or not nano:
        raise ValueError("Expected a YOLOv8 nano detection checkpoint")
    if not any(layer.__class__.__name__ == 'C2f' for layer in detector.model.model):
        raise ValueError("Expected YOLOv8, not a newer architecture")
    cfg = deepcopy(detector.model.yaml)
    cfg["nc"] = len(config.class_names)
    adapted = DetectionModel(cfg=cfg, ch=3, nc=len(config.class_names), verbose=False)
    adapted.names = dict(enumerate(config.class_names))
    adapted.load(detector.model, verbose=False)
    adapted.args = deepcopy(detector.model.args)
    detector.model = adapted
    if detector.model.model[-1].nc != len(CLASS_NAMES):
        raise ValueError("Detection-head adaptation failed")
    return detector


def load_trained_detector(weights: Path) -> Any:
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    detector = YOLO(str(weights), task="detect")
    names = detector.names
    ordered = tuple(names[i] for i in range(len(names)))
    if detector.task != "detect" or ordered != CLASS_NAMES:
        raise ValueError("Checkpoint task/classes do not match this project")
    return detector


def model_summary(detector: Any) -> dict:
    import torch
    import ultralytics

    network = detector.model
    return {
        "classes": detector.names,
        "parameters": sum(p.numel() for p in network.parameters()),
        "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
        "head_classes": network.model[-1].nc,
        "strides": network.stride.tolist(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov8n.pt")
    args = parser.parse_args()
    print(json.dumps(model_summary(build_detector(DetectorConfig(weights=args.weights))), indent=2))
