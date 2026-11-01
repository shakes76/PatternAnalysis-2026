import os
from ultralytics import YOLO

# loading of model
def load_initial_model(weights = "yolov8n.pt"):
    """
    Loading of YOLOv8 model from ulralytics (in partciular yolov8n)
    --> weights pre-determined, pre-trained on COCO dataset
    """
    return YOLO(weights)

def load_trained(weights_path): # loading up a trained YOLO model 
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No model found at {weights_path}")
    return YOLO(str(weights_path))
