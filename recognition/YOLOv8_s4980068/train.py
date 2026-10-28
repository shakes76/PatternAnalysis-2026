from pathlib import Path
from modules import load_model
import argparse

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16
SEED = 42
DEVICE = 0

def train_model(model_name, data_yaml, output_dir):
    output_dir = Path(output_dir).expanduser().resolve()

    model = load_model(model_name)

    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        seed=SEED,
        device=DEVICE,
        project=str(output_dir),
        name=model_name,
        exist_ok=True,
    )

    return results

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    data_yaml = project_dir / "ISIC2018_yolo" / "data.yaml"
    output_dir = project_dir / "COMP3710_results"

    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_yaml}")

    for model_name in ["yolov8n", "yolov8s"]:
        print(f"\nStarting training: {model_name}")
        train_model("yolov8n", data_yaml, output_dir)