import torch
import torch.nn as nn
from ultralytics import YOLO


class ISICDetector(nn.Module):
    """
    Modular PyTorch wrapper supporting both the standard Baseline detector (YOLOv8n)
    and the Dermoscopy-Optimized Chosen Detector (YOLOv8m / YOLOv11m).
    Zero NumPy dependency in compliance with COMP3710 rules.
    """

    def __init__(self, weights_path: str = "yolov8n.pt"):
        super().__init__()
        self.detector = YOLO(weights_path)

    def train_detector(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 32,
        device: str = "cuda",
        project: str = "runs/isic_experiments",
        name: str = "yolov8n_baseline_100ep",
        cache: str = "ram",
        workers: int = 4,
        dermoscopic_tuning: bool = False,
    ):
        # Dermoscopic-specific hyperparameter improvements for the chosen model
        extra_kwargs = {}
        if dermoscopic_tuning:
            extra_kwargs = {
                "degrees": 45.0,      # Rotational invariance for overhead skin imaging
                "flipud": 0.5,        # Vertical flip invariance
                "fliplr": 0.5,        # Horizontal flip invariance
                "box": 10.0,          # Upweight CIoU box loss to exceed IoU >= 0.80 target
                "mosaic": 0.5,        # Reduced mosaic to preserve peripheral lesion borders
                "cos_lr": True,       # Cosine annealing learning rate schedule
            }

        return self.detector.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            cache=cache,
            workers=workers,
            amp=True,
            plots=True,
            save=True,
            exist_ok=True,
            **extra_kwargs,
        )

    def evaluate_split(self, data_yaml: str, split: str = "test", imgsz: int = 640, conf: float = 0.001, iou: float = 0.5):
        return self.detector.val(
            data=data_yaml,
            split=split,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            plots=True,
        )

    def predict_images(self, source, imgsz: int = 640, conf: float = 0.01, iou: float = 0.5, device: str = "cuda"):
        return self.detector.predict(
            source=source,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )


def compute_box_iou_tensor(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch pairwise IoU calculation for [x1, y1, x2, y2] bounding boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)