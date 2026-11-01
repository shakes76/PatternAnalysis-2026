"""
predict.py: demo usage of the trained YOLO model on ISIC 2017 images.
Prints per-detection results and saves annotated images.
For test-set metrics (IoU, accuracy, mAP), use evaluate.py instead.

Usage: edit the SETTINGS below, then run:  python predict.py
"""
import os
from modules import load_trained
# configurations
from config import WEIGHTS, OUT_DIR, CLASS_NAMES, CONF_DEMO, IMGSZ, DATA_DIR

# -CORE SETTINGS - edit these as required
SOURCE = DATA_DIR / "test" / "images" / "ISIC_0015963.jpg" # single image or folder of images
MODEL_WEIGHTS = WEIGHTS # path to the trained best.pt
OUTPUT_DIR = OUT_DIR / "predict" # where annotated images are saved
DEVICE = "cpu" # "cpu" (for local machine), or 0 for GPU (Rangpur)


def demo_predict(model, source, device, out_dir):
    """Runs on an image or folder. Prints each detection and saves annotated images."""
    results = model.predict(source=str(source),
                            imgsz=IMGSZ,
                            conf=CONF_DEMO,
                            device=device,
                            save=True,
                            show=False,
                            project=str(out_dir),
                            name="lesion_predictions",
                            exist_ok=True) # configurations
    # NOTE: Assisted by Claude Sonnet 5.0 (Medium)
    for r in results:
        print("\nImage:", r.path)
        if r.boxes is not None and len(r.boxes) > 0: # if boxes exist
            for box in r.boxes:
                class_name = CLASS_NAMES[int(box.cls.item())]
                conf = float(box.conf.item())
                xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
                print(f"  class: {class_name}")
                print(f"  confidence: {conf:.2f}")
                print(f"  box (x1, y1, x2, y2): {xyxy}") # metrics
        else:
            print("No lesions detected")
    # NOTE: suggested by Claude Sonnet 5.0 (Medium)
    print(f"\nDone predicting. Annotated images saved under {os.path.join(out_dir, 'lesion_predictions')}")


def main():
    model = load_trained(MODEL_WEIGHTS)
    demo_predict(model, SOURCE, DEVICE, OUTPUT_DIR)


if __name__ == "__main__":
    main()
