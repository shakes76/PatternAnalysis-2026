import torch
import torch.nn as nn
from ultralytics import YOLO


class ISICBaselineDetector(nn.Module):
    """
    PyTorch wrapper for the Baseline YOLOv8 Detector on ISIC 2018.
    Note: No NumPy dependency is used in this module in compliance with COMP3710 guidelines.
    """

    def __init__(self, weights_path: str = "yolov8n.pt"):
        super().__init__()
        self.detector = YOLO(weights_path)

    def train_detector(self, data_yaml: str, epochs: int = 50, imgsz: int = 640, batch: int = 16,
                       device: str = "cuda", project: str = "runs/isic_baseline", name: str = "yolov8n_50ep"):
        return self.detector.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            plots=True,
            save=True,
            exist_ok=True,
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

    def predict_images(self, source, imgsz: int = 640, conf: float = 0.10, iou: float = 0.5, device: str = "cuda"):
        return self.detector.predict(
            source=source,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )


def compute_box_iou_tensor(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch calculation of pairwise Bounding Box IoU for [x1, y1, x2, y2] tensors.
    Returns an (N, M) tensor of IoU values.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)