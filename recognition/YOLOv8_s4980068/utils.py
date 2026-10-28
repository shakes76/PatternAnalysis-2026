from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle

def read_yolo_label(label_path):
    with open(label_path, "r") as file:
        line = file.readline().strip()

    class_id, x_center, y_center, width, height = map(float, line.split())

    return int(class_id), x_center, y_center, width, height

def yolo_to_bbox(x_center, y_center, width, height, image_width, image_height):
    # Convert normalized values into pixels
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    x_min = x_center - width / 2
    y_min = y_center - height / 2

    return x_min, y_min, width, height

def visualize_bbox(image_path, label_path):
    image_path = Path(image_path)
    label_path = Path(label_path)

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    image_width, image_height = image.size

    # Read YOLO annotations
    class_id, x_center, y_center, width, height = read_yolo_label(label_path)
    x_min, y_min, box_width, box_height = yolo_to_bbox(x_center, y_center,
                                                       width, height,
                                                       image_width, image_height)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image)

    # Draw bounding box
    bbox = Rectangle((x_min, y_min), box_width, box_height,
                     linewidth=2, edgecolor="lime", facecolor="none")
    ax.add_patch(bbox)
    ax.set_title(f"YOLO Annotation - Class {class_id}")
    ax.axis("off")

    plt.show()

def validate_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)

    split_filenames = {}

    for split in ["train", "val", "test"]:
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split

        image_paths = sorted(image_dir.glob("*.jpg"))
        image_names = set()

        for image_path in image_paths:
            image_names.add(image_path.stem)

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                raise FileExistsError(f"Missing label: {label_path}")

            class_id, x, y, w, h = read_yolo_label(label_path)

            # Allow a small tolerance for rounding errors
            tolerance = 1e-6

            x_min = x - w / 2
            x_max = x + w / 2
            y_min = y - h / 2
            y_max = y + h / 2

            if class_id != 0:
                raise ValueError(f"Invalid class ID in {label_path}")

            if not (0 < x < 1 and 0 < y < 1):
                raise ValueError(f"Invalid bounding-box center: {label_path}")

            if not (0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f"Invalid bounding-box dimensions: {label_path}")

            if not (-tolerance <= x_min and x_max <= 1 + tolerance):
                raise ValueError(f"Bounding box exceeds image width: {label_path}")

            if not (-tolerance <= y_min and y_max <= 1 + tolerance):
                raise ValueError(f"Bounding box exceeds image height: {label_path}")

        # Check for unexpected label files
        label_names = {path.stem for path in label_dir.glob("*.txt")}
        if image_names != label_names:
            raise ValueError(f"Image/label mismatch in {split}")

        split_filenames[split] = image_names
        print(f"{split}: {len(image_names)} valid annotations")

    train = split_filenames["train"]
    val = split_filenames["val"]
    test = split_filenames["test"]

    if train & val or train & test or val & test:
        raise ValueError("Duplicate images found across dataset splits")

    print("\nDataset validation passed!")

if __name__ == "__main__":
    dataset_dir = Path("ISIC2018_yolo")
    validate_dataset(dataset_dir)