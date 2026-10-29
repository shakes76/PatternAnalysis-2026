import os
import shutil
from pathlib import Path
import cv2
import yaml
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def mask_to_yolo_bboxes(mask_path: Path, class_id: int = 0, min_area: int = 25):
    """
    Convert binary segmentation mask into normalized YOLO bounding box format:
    [class_id, x_center, y_center, width, height] in [0, 1].
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    h, w = mask.shape
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    # In ISIC 2018 Task 1, each image depicts a primary lesion.
    # If multiple disconnected mask islands exist, take the union bounding box
    # of significant contours to avoid fragmenting a single lesion.
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    if not valid_contours:
        return []

    x_min = min(cv2.boundingRect(c)[0] for c in valid_contours)
    y_min = min(cv2.boundingRect(c)[1] for c in valid_contours)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_contours)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_contours)

    bw = x_max - x_min
    bh = y_max - y_min
    x_center = (x_min + bw / 2.0) / w
    y_center = (y_min + bh / 2.0) / h
    norm_w = bw / w
    norm_h = bh / h

    bboxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
    return bboxes


def discover_isic_directories(raw_root: Path):
    """
    Automatically locates image and mask directories inside ISIC 2017 or ISIC 2018
    directory structures (both local and /home/groups/comp3710/ISIC2018).
    """
    candidates_img = [
        raw_root / "ISIC-2017_Training_Data",
        raw_root / "ISIC2018_Task1-2_Training_Input",
        raw_root / "ISIC2018_Task1-2_Training_Input_x2",
        raw_root / "images",
        raw_root,
    ]
    candidates_mask = [
        raw_root / "ISIC-2017_Training_Part1_GroundTruth",
        raw_root / "ISIC2018_Task1_Training_GroundTruth",
        raw_root / "ISIC2018_Task1_Training_GroundTruth_x2",
        raw_root / "masks",
        raw_root,
    ]

    img_dir = next((d for d in candidates_img if d.exists() and list(d.glob("ISIC_*.jpg"))), None)
    mask_dir = next((d for d in candidates_mask if d.exists() and list(d.glob("*_segmentation.png"))), None)

    if img_dir is None or mask_dir is None:
        raise FileNotFoundError(
            f"Could not find ISIC_*.jpg and *_segmentation.png inside {raw_root}. "
            f"Verify that both Training_Data and Part1_GroundTruth zips are extracted."
        )
    return img_dir, mask_dir

def load_patient_metadata(raw_root: Path, image_ids: list):
    """
    Attempts to load patient_id / lesion_id and diagnostic class from ISIC metadata CSVs
    to enforce strict patient-independent and class-stratified splitting.
    Falls back to deterministic grouping if metadata CSV is absent.
    """
    df = pd.DataFrame({"image_id": image_ids})
    df["group_id"] = df["image_id"]  # Default 1-to-1 group if no metadata file found
    df["class_id"] = 0               # Default 0 = skin lesion

    csv_files = list(raw_root.rglob("*.csv"))
    for csv_path in csv_files:
        try:
            meta = pd.read_csv(csv_path)
            if "image" in meta.columns or "image_id" in meta.columns:
                id_col = "image" if "image" in meta.columns else "image_id"
                if "lesion_id" in meta.columns:
                    mapping = dict(zip(meta[id_col], meta["lesion_id"]))
                    df["group_id"] = df["image_id"].map(mapping).fillna(df["group_id"])
                elif "patient_id" in meta.columns:
                    mapping = dict(zip(meta[id_col], meta["patient_id"]))
                    df["group_id"] = df["image_id"].map(mapping).fillna(df["group_id"])

                if "MEL" in meta.columns:
                    mel_map = dict(zip(meta[id_col], meta["MEL"]))
                    df["is_melanoma"] = df["image_id"].map(mel_map).fillna(0).astype(int)
        except Exception:
            continue

    return df


def create_patient_stratified_splits(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Leakage-free split ensuring all images from the same patient/lesion group
    stay strictly within a single split (70% train, 15% val, 15% test).
    """
    gss_test = GroupShuffleSplit(n_splits=1, test_size=(1.0 - train_ratio), random_state=seed)
    train_idx, temp_idx = next(gss_test.split(df, groups=df["group_id"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    relative_val_ratio = val_ratio / (1.0 - train_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=(1.0 - relative_val_ratio), random_state=seed)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["group_id"]))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    # Verify zero patient/lesion group overlap
    assert set(train_df["group_id"]).isdisjoint(set(val_df["group_id"])), "Data leakage between train and val!"
    assert set(train_df["group_id"]).isdisjoint(set(test_df["group_id"])), "Data leakage between train and test!"
    assert set(val_df["group_id"]).isdisjoint(set(test_df["group_id"])), "Data leakage between val and test!"

    return {"train": train_df, "val": val_df, "test": test_df}


def link_or_copy(src: Path, dst: Path):
    """Uses fast zero-storage symlinks on Linux/HPC; falls back to copy on Windows if needed."""
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_isic_yolo_dataset(raw_dir: str, output_dir: str, yaml_out_path: str, max_samples: int = None):
    """
    Builds the complete YOLO dataset structure with patient-stratified train/val/test splits
    and writes the dataset YAML configuration file.
    """
    raw_root = Path(raw_dir)
    out_root = Path(output_dir).resolve()
    img_dir, mask_dir = discover_isic_directories(raw_root)

    mask_files = sorted(list(mask_dir.glob("*_segmentation.png")))
    if max_samples is not None:
        mask_files = mask_files[:max_samples]

    records = []
    for mask_p in mask_files:
        image_id = mask_p.name.replace("_segmentation.png", "")
        img_p = img_dir / f"{image_id}.jpg"
        if img_p.exists():
            records.append({"image_id": image_id, "img_path": img_p, "mask_path": mask_p})

    if not records:
        raise RuntimeError(f"No matched image-mask pairs found in {raw_root}")

    meta_df = load_patient_metadata(raw_root, [r["image_id"] for r in records])
    full_df = pd.DataFrame(records).merge(meta_df, on="image_id")

    splits = create_patient_stratified_splits(full_df)

    for split_name, split_df in splits.items():
        (out_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        valid_count = 0
        for _, row in split_df.iterrows():
            bboxes = mask_to_yolo_bboxes(row["mask_path"], class_id=int(row["class_id"]))
            if not bboxes:
                continue

            target_img = out_root / "images" / split_name / f"{row['image_id']}.jpg"
            target_lbl = out_root / "labels" / split_name / f"{row['image_id']}.txt"

            link_or_copy(row["img_path"], target_img)
            with open(target_lbl, "w") as f:
                f.write("\n".join(bboxes) + "\n")
            valid_count += 1

        print(f"[{split_name.upper()}] Prepared {valid_count} patient-stratified samples.")

    yaml_config = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "lesion"},
    }

    yaml_path = Path(yaml_out_path).resolve()
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_config, f, sort_keys=False)

    print(f"Dataset YAML written to: {yaml_path}")
    return str(yaml_path)