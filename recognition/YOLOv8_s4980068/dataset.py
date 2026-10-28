from pathlib import Path
import numpy as np
from PIL import Image
import random
import shutil
import argparse

IMAGE_FOLDER = "ISIC2018_Task1-2_Training_Input_x2"
MASK_FOLDER = "ISIC2018_Task1_Training_GroundTruth_x2"

def find_image_mask_pairs(image_dir, mask_dir):
    image_paths = sorted(image_dir.glob("*.jpg"))
    
    pairs = []
    for image_path in image_paths:
        image_id = image_path.stem
        mask_path = mask_dir / f"{image_id}_segmentation.png"

        if not mask_path.exists():
            raise FileExistsError(f"Missing mask for {image_path.name}")

        pairs.append((image_path, mask_path))

    return pairs

def mask_to_bbox(mask_path):
    with Image.open(mask_path) as image:
        mask = np.array(image.convert("L"))

    y_coords, x_coords = np.where(mask > 0)

    # Check for empty mask
    if len(x_coords) == 0:
        raise ValueError(f"No lesion found in mask: {mask_path}")

    x_min, x_max = x_coords.min(), x_coords.max() + 1
    y_min, y_max = y_coords.min(), y_coords.max() + 1

    return x_min, y_min, x_max, y_max

def bbox_to_yolo(bbox, image_width, image_height):
    x_min, y_min, x_max, y_max = bbox

    # Calculate the centre of the bounding box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Calculate its width and height
    box_width = x_max - x_min
    box_height = y_max - y_min

    # Normalize coordinates to the range [0,1]
    x_center /= image_width
    y_center /= image_height
    box_width /= image_width
    box_height /= image_height

    return x_center, y_center, box_width, box_height

def split_dataset(pairs, seed=42):
    shuffled_pairs = pairs.copy()
    random.Random(seed).shuffle(shuffled_pairs)

    total = len(shuffled_pairs)
    n_test = round(total * 0.10)
    n_val = round(total * 0.10)

    test_pairs = shuffled_pairs[:n_test]
    val_pairs = shuffled_pairs[n_test:n_test + n_val]
    train_pairs = shuffled_pairs[n_test + n_val:]

    return train_pairs, val_pairs, test_pairs

def save_split(split_name, split_pairs, output_dir):
    image_output_dir = output_dir / "images" / split_name
    label_output_dir = output_dir / "labels" / split_name

    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path in split_pairs:
        # Get bounding box from segmentation mask
        bbox = mask_to_bbox(mask_path)

        # Get original image dimensions
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        with Image.open(mask_path) as mask:
            mask_width, mask_height = mask.size

        if (image_width, image_height) != (mask_width, mask_height):
            raise ValueError(f"Image and mask dimensions do not match: {image_path.name}")

        # Convert bounding box to YOLO format
        yolo_bbox = bbox_to_yolo(bbox, image_width, image_height)

        # Create label text
        x_center, y_center, width, height = yolo_bbox

        label_text = (
            f"0 {x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}\n"
        )

        # Copy image into the appropriate split
        destination_image = image_output_dir / image_path.name
        shutil.copy2(image_path, destination_image)

        # Save the bounding-box annotation
        destination_label = label_output_dir / f"{image_path.stem}.txt"
        destination_label.write_text(label_text, encoding="utf-8")

    print(f"Saved {len(split_pairs)} images to {split_name}")

def create_data_yaml(output_dir):
    dataset_path = output_dir.resolve()

    yaml_content = (
        f"path: '{dataset_path}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: lesion\n"
    )

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"Created configuration file: {yaml_path}")

def prepare_dataset(dataset_dir, output_dir, seed=42):
    image_dir = dataset_dir / IMAGE_FOLDER
    mask_dir = dataset_dir / MASK_FOLDER

    pairs = find_image_mask_pairs(image_dir, mask_dir)
    train_pairs, val_pairs, test_pairs = split_dataset(pairs, seed)

    save_split("train", train_pairs, output_dir)
    save_split("val", val_pairs, output_dir)
    save_split("test", test_pairs, output_dir)

    create_data_yaml(output_dir)

    print("Dataset preparation completed!")


def main():
    parser = argparse.ArgumentParser(description="Prepare ISIC2018 dataset for YOLOv8 detection")
    parser.add_argument("--dataset-dir", type=Path, required=True, 
                        help="Path to the original ISIC2018 dataset")
    parser.add_argument("--output-dir",type=Path,required=True,
        help="Path to save the prepared YOLO dataset")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for reproducible dataset splitting")

    args = parser.parse_args()
    prepare_dataset(args.dataset_dir, args.output_dir, args.seed)

if __name__ == "__main__":
    main()