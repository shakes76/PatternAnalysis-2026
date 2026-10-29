import argparse
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch

from dataset import prepare_isic_yolo_dataset
from modules import ISICDetector


def plot_training_curves(run_dir: Path, output_fig: Path):
    """Reads Ultralytics results.csv and generates a clean 2-panel loss & metric figure."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found for plotting.")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Box, Cls, DFL Losses
    axes[0].plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
    axes[0].plot(df["epoch"], df["val/box_loss"], label="Val Box Loss", linestyle="--")
    axes[0].plot(df["epoch"], df["train/cls_loss"], label="Train Cls Loss")
    axes[0].set_title("YOLOv8n Baseline Training & Validation Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: mAP50, mAP50-95, Recall, Precision
    axes[1].plot(df["epoch"], df["metrics/mAP50(B)"], label="Val mAP@50", linewidth=2)
    axes[1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="Val mAP@50:95", linewidth=2)
    axes[1].plot(df["epoch"], df["metrics/recall(B)"], label="Val Recall (Sensitivity)", linestyle=":")
    axes[1].plot(df["epoch"], df["metrics/precision(B)"], label="Val Precision", linestyle=":")
    axes[1].set_title("Validation Detection Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=200)
    plt.close()
    print(f"Saved training loss & metric curves to: {output_fig}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 ISIC Training")
    parser.add_argument("--raw_dir", type=str, default="../../data/ISIC2017", help="Path to raw ISIC dataset")
    parser.add_argument("--data_dir", type=str, default="./data/isic2018_yolo",
                        help="Output directory for YOLO dataset")
    parser.add_argument("--yaml_path", type=str, default="./isic2018.yaml", help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running ISIC Training on device: {device} ===")

    # 1. Prepare Patient-Stratified Dataset
    yaml_file = prepare_isic_yolo_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.data_dir,
        yaml_out_path=args.yaml_path,
    )

    # 2. Initialize & Train Model
    model = ISICDetector(weights_path="yolov8n.pt")
    project_dir = str(Path("./runs/isic_baseline").resolve())

    start_t = time.time()
    train_results = model.train_detector(
        data_yaml=yaml_file,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=project_dir,
        name="yolov8n_100ep",
        cache="ram",
    )
    train_time = time.time() - start_t

    run_dir = Path(train_results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = run_dir / "weights" / "last.pt"

    plot_training_curves(run_dir, Path("baseline_training_curves.png"))

    print(f"\n================ TRAINING COMPLETE ================")
    print(f"Total Training Time:      {train_time / 60:.2f} minutes")
    print(f"Saved Best Weights:       {best_weights.resolve()}")
    print("===================================================")


if __name__ == "__main__":
    main()