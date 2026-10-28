from ultralytics import YOLO

MODEL_CONFIGS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
}

def load_model(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    return YOLO(MODEL_CONFIGS[model_name])

def get_model_info(model):
    total_params = sum(
        parameter.numel()
        for parameter in model.model.parameters()
    )

    return {"total_parameters": total_params}

if __name__ == "__main__":
    for model_name in MODEL_CONFIGS:
        model = load_model(model_name)
        info = get_model_info(model)

        print(f"\nModel: {model_name}")
        print(f"Total parameters: {info['total_parameters']:,}")
