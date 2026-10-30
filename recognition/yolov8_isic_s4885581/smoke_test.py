import os
import glob
import shutil
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

# 1. Resolve Dataset Path Dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1] if (SCRIPT_DIR.parents[1] / "recognition").exists() else SCRIPT_DIR.parent

RANGPUR_PATH = Path("/home/groups/comp3710/ISIC2018")
LOCAL_PATH_OPTIONS = [
    SCRIPT_DIR / "sample_isic_data",
    PROJECT_ROOT / "sample_isic_data",
    Path("./sample_isic_data").resolve()
]

# Pick the first available path
DATASET_ROOT = None
if RANGPUR_PATH.exists():
    DATASET_ROOT = RANGPUR_PATH
else:
    for candidate in LOCAL_PATH_OPTIONS:
        if candidate.exists() and any(candidate.iterdir()):
            DATASET_ROOT = candidate
            break

SMOKE_DIR = SCRIPT_DIR / "smoke_run_data"
NUM_SAMPLES = 10


def mask_to_yolo_bbox(mask_path):
    """Converts a binary mask to normalized YOLO bounding boxes."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    h_img, w_img = mask.shape
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 25:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        x_c = (x + w / 2.0) / w_img
        y_c = (y + h / 2.0) / h_img
        norm_w = w / w_img
        norm_h = h / h_img
        boxes.append(f"0 {x_c:.6f} {y_c:.6f} {norm_w:.6f} {norm_h:.6f}")

    return boxes


def prepare_smoke_dataset():
    if DATASET_ROOT is None:
        print("[!] ERROR: Could not locate 'sample_isic_data' or '/home/groups/comp3710/ISIC2018'.")
        print(f"    Looked in:")
        for p in LOCAL_PATH_OPTIONS:
            print(f"    - {p}")
        return None

    print(f"[*] Found dataset root at: {DATASET_ROOT}")

    # Find all images and masks recursively
    all_files = list(DATASET_ROOT.rglob("*"))
    image_paths = [p for p in all_files if p.suffix.lower() in [".jpg", ".jpeg"] and "segmentation" not in p.name.lower()]
    mask_paths = [p for p in all_files if p.suffix.lower() in [".png", ".jpg"] and "segmentation" in p.name.lower()]

    print(f"[*] Discovered {len(image_paths)} raw images and {len(mask_paths)} mask files.")

    # Match images with masks by ISIC ID
    mask_dict = {}
    for mp in mask_paths:
        base_id = mp.stem.replace("_segmentation", "")
        mask_dict[base_id] = mp

    matched_pairs = []
    for ip in image_paths:
        base_id = ip.stem
        if base_id in mask_dict:
            matched_pairs.append((ip, mask_dict[base_id]))
        if len(matched_pairs) >= NUM_SAMPLES:
            break

    if not matched_pairs:
        print("[!] Found files, but could not match any image ID to a mask ID.")
        print(f"    Sample images found: {[p.name for p in image_paths[:3]]}")
        print(f"    Sample masks found: {[p.name for p in mask_paths[:3]]}")
        return None

    print(f"[*] Successfully paired {len(matched_pairs)} image-mask samples.")

    # Set up smoke directories
    for split in ["train", "val"]:
        (SMOKE_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (SMOKE_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Use at least 1 validation image if we have 2+ samples
    split_idx = max(1, int(len(matched_pairs) * 0.7)) if len(matched_pairs) > 1 else 1

    for idx, (img_p, msk_p) in enumerate(matched_pairs):
        split = "train" if idx < split_idx else "val"
        dest_img = SMOKE_DIR / "images" / split / img_p.name
        dest_lbl = SMOKE_DIR / "labels" / split / f"{img_p.stem}.txt"

        shutil.copy(img_p, dest_img)
        bboxes = mask_to_yolo_bbox(msk_p)

        with open(dest_lbl, "w") as f:
            f.write("\n".join(bboxes) + "\n")

    # If only 1 sample exists, mirror it into val so YOLO validation doesn't crash
    val_images = list((SMOKE_DIR / "images" / "val").glob("*"))
    if not val_images:
        for f in (SMOKE_DIR / "images" / "train").glob("*"):
            shutil.copy(f, SMOKE_DIR / "images" / "val" / f.name)
        for f in (SMOKE_DIR / "labels" / "train").glob("*"):
            shutil.copy(f, SMOKE_DIR / "labels" / "val" / f.name)

    yaml_content = f"""
path: {SMOKE_DIR.resolve()}
train: images/train
val: images/val
names:
  0: lesion
"""
    yaml_path = SMOKE_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content.strip())

    return yaml_path


def run_smoke_test(data_yaml):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] Execution Device: {device.upper()}")
    if device == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    model = YOLO("yolov8n.pt")

    print("[*] Running 1-epoch verification loop...")
    model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=320,  # Smaller size for rapid local verification
        batch=2,
        device=device,
        plots=False,
        save=False,
        workers=0
    )

    print("\n--- SMOKE TEST AUDIT SUMMARY ---")
    print("✓ Model initialization: SUCCESS")
    print("✓ Dataset parsing & Bounding box extraction: SUCCESS")
    print("✓ Forward and backward passes: SUCCESS")

    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"✓ Peak Training VRAM: {peak_vram:.2f} GB")
    else:
        print("✓ Verified successfully on local CPU.")


if __name__ == "__main__":
    yaml_file = prepare_smoke_dataset()
    if yaml_file:
        try:
            run_smoke_test(yaml_file)
        finally:
            if SMOKE_DIR.exists():
                shutil.rmtree(SMOKE_DIR)
                print("[*] Temporary smoke files removed cleanly.")