from pathlib import Path
from ultralytics import YOLO
import csv
import cv2
import time

IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = "cpu"
CONF_THRESHOLD = 0.25

def evaluate_model(model_name, data_yaml, weights_path, output_dir):
    model = YOLO(str(weights_path))

    metrics = model.val(data=str(data_yaml), split="test", imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
                        device=DEVICE, project=str(output_dir), name=f"{model_name}_test",
                        exist_ok=True, plots=True)

    print(f"\n--- {model_name} Test Results ---")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")
    print(f"mAP@50:        {metrics.box.map50:.4f}")
    print(f"mAP@50:95:     {metrics.box.map:.4f}")

    return metrics

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def evaluate_localization(model_name, model, dataset_dir, output_dir):
    image_dir = dataset_dir / "images" / "test"
    label_dir = dataset_dir / "labels" / "test"

    results = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        # Read the ground-truth bounding box
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        with open(label_path, "r") as f:
            values = list(map(float, f.readline().split()))

        _, x, y, w, h = values

        # Predict bounding boxes
        prediction = model.predict(source=str(image_path), imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD,
                                   device=DEVICE, verbose=False)[0]
        image_height, image_width = prediction.orig_shape

        # Convert normalized ground truth to pixel coordinates
        gt_box = [
            (x - w / 2) * image_width,
            (y - h / 2) * image_height,
            (x + w / 2) * image_width,
            (y + h / 2) * image_height,
        ]

        # Select the prediction with the highest IoU
        best_iou = 0.0
        best_confidence = 0.0

        for box in prediction.boxes:
            pred_box = box.xyxy[0].tolist()
            iou = calculate_iou(gt_box, pred_box)

            if iou > best_iou:
                best_iou = iou
                best_confidence = float(box.conf[0])

        results.append({
            "image": image_path.name,
            "iou": round(best_iou, 4),
            "confidence": round(best_confidence, 4),
        })

    # Save per-image localisation results
    save_path = output_dir / f"{model_name}_iou.csv"

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "iou", "confidence"])
        writer.writeheader()
        writer.writerows(results)

    if not results:
        raise ValueError(f"No test images evaluated in {image_dir}")

    mean_iou = sum(row["iou"] for row in results) / len(results)

    print(f"\n--- {model_name} Localization Results ---")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Images evaluated: {len(results)}")
    print(f"Results saved to: {save_path}") 

    return results

def find_failure_cases(output_dir, num_cases=5):
    for model_name in ["yolov8n", "yolov8s"]:
        csv_path = output_dir / f"{model_name}_iou.csv"

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            results = list(reader)

        # Sort images from lowest to highest IoU
        results.sort(key=lambda row: float(row["iou"]))

        print(f"\n--- {model_name} Worst {num_cases} Cases ---")

        for row in results[:num_cases]:
            print(
                f"{row['image']} | "
                f"IoU: {float(row['iou']):.4f} | "
                f"Confidence: {float(row['confidence']):.4f}"
            )

def visualize_failure_cases(models, dataset_dir, output_dir):
    image_dir = dataset_dir / "images" / "test"
    label_dir = dataset_dir / "labels" / "test"

    save_dir = output_dir / "failure_cases"
    save_dir.mkdir(parents=True, exist_ok=True)

    failure_images = [
        "ISIC_0012207.jpg",
        "ISIC_0015614.jpg",
        "ISIC_0012159.jpg",
        "ISIC_0013684.jpg",
        "ISIC_0013177.jpg",
    ]

    for filename in failure_images:
        image_path = image_dir / filename
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {filename}")
            continue

        height, width = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"

        ground_truth = image.copy()

        for line in label_path.read_text().splitlines():
            _, x, y, w, h = map(float, line.split())

            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)

            cv2.rectangle(ground_truth, (x1, y1), (x2, y2), (0, 255, 0), 3)

        panels = [ground_truth]

        # Draw predictions from both models in red
        for model_name, model in models.items():
            result = model.predict(source=str(image_path), imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD,
                                   device=DEVICE, verbose=False)[0]

            prediction = ground_truth.copy()

            if len(result.boxes) == 0:
                cv2.putText(prediction, "No detection", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])

                cv2.rectangle(prediction, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(prediction, f"Conf: {confidence:.2f}", (x1, max(y1 - 10, 50)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            (0, 0, 255), 2)
            panels.append(prediction)

        titles = ["Ground Truth"] + list(models.keys())
        titled_panels = []

        for panel, title in zip(panels, titles):
            header = cv2.copyMakeBorder(panel, 50, 0, 0, 0,
                                        cv2.BORDER_CONSTANT, value=(40, 40, 40))
            cv2.putText(header, title, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            titled_panels.append(header)

        comparison = cv2.hconcat(titled_panels)
        save_path = save_dir / f"{image_path.stem}_failure.jpg"

        cv2.imwrite(str(save_path), comparison)
        print(f"Saved: {save_path}")

def benchmark_inference(model_name, model, dataset_dir):
    image_dir = dataset_dir / "images" / "test"
    image_paths = sorted(image_dir.glob("*.jpg"))

    if not image_paths:
        raise ValueError(f"No test images found in {image_dir}")

    # Warm up the model before measuring speed
    for _ in range(5):
        model.predict(source=str(image_paths[0]), imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD,
                      device=DEVICE, verbose=False)

    # Measure the time taken to process all test images
    start_time = time.perf_counter()

    for image_path in image_paths:
        model.predict(source=str(image_path), imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD,
                      device=DEVICE, verbose=False)

    total_time = time.perf_counter() - start_time

    num_images = len(image_paths)
    fps = num_images / total_time
    avg_latency = total_time / num_images * 1000

    print(f"\n--- {model_name} Inference Benchmark ---")
    print(f"Device:         {DEVICE}")
    print(f"Images:         {num_images}")
    print(f"Total time:     {total_time:.2f} seconds")
    print(f"Average latency: {avg_latency:.2f} ms/image")
    print(f"Throughput:     {fps:.2f} FPS")

    return fps


def evaluate_thresholds(models, data_yaml, output_dir):
    thresholds = [0.10, 0.25, 0.50, 0.75]
    results = []

    for model_name, model in models.items():
        for threshold in thresholds:
            metrics = model.val(data=str(data_yaml), split="val", imgsz=IMAGE_SIZE,
                                batch=BATCH_SIZE, conf=threshold, device=DEVICE,
                                plots=False, verbose=False, project=str(output_dir),
                                name=f"{model_name}_threshold_{threshold}", exist_ok=True)

            results.append({
                "model": model_name,
                "threshold": threshold,
                "precision": round(metrics.box.mp, 4),
                "recall": round(metrics.box.mr, 4),
            })

            print(
                f"{model_name} | threshold={threshold:.2f} | "
                f"Precision={metrics.box.mp:.4f} | "
                f"Recall={metrics.box.mr:.4f}"
            )

    save_path = output_dir / "threshold_analysis.csv"

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "threshold", "precision", "recall"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nThreshold analysis saved to: {save_path}")


def count_detections(models, dataset_dir, output_dir):
    image_dir = dataset_dir / "images" / "val"
    label_dir = dataset_dir / "labels" / "val"

    thresholds = [0.10, 0.25, 0.50, 0.75]
    results = []

    for model_name, model in models.items():
        for threshold in thresholds:
            tp, fp, fn = 0, 0, 0

            for image_path in sorted(image_dir.glob("*.jpg")):
                label_path = label_dir / f"{image_path.stem}.txt"

                if not label_path.is_file():
                    raise FileNotFoundError(label_path)

                values = list(map(float, label_path.read_text().split()))
                _, x, y, w, h = values

                prediction = model.predict(source=str(image_path), imgsz=IMAGE_SIZE, conf=threshold,
                                           device=DEVICE, verbose=False)[0]

                image_height, image_width = prediction.orig_shape

                gt_box = [
                    (x - w / 2) * image_width,
                    (y - h / 2) * image_height,
                    (x + w / 2) * image_width,
                    (y + h / 2) * image_height,
                ]

                # One ground-truth lesion per image
                matched = False

                for box in prediction.boxes:
                    pred_box = box.xyxy[0].tolist()
                    iou = calculate_iou(gt_box, pred_box)

                    if iou >= 0.50 and not matched:
                        tp += 1
                        matched = True
                    else:
                        fp += 1

                if not matched:
                    fn += 1

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0

            results.append({
                "model": model_name,
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            })

            print(
                f"{model_name} | conf={threshold:.2f} | "
                f"TP={tp} FP={fp} FN={fn} | "
                f"P={precision:.4f} R={recall:.4f}"
            )

    save_path = output_dir / "threshold_counts.csv"

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "threshold", "tp", "fp", "fn", "precision", "recall"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {save_path}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parents[1]

    dataset_dir = project_dir / "ISIC2018_yolo"
    data_yaml = dataset_dir / "data.yaml"
    output_dir = project_dir / "COMP3710_results"

    # Select which tasks to run
    RUN_EVALUATION = False
    RUN_LOCALIZATION = False
    RUN_FAILURE_ANALYSIS = False
    RUN_VISUALIZATION = False
    RUN_BENCHMARK = False
    RUN_THRESHOLD_ANALYSIS = False

    models = {}

    for model_name in ["yolov8n", "yolov8s"]:
        weights_path = output_dir / model_name / "weights" / "best.pt"

        if not weights_path.is_file():
            raise FileNotFoundError(f"Model not found: {weights_path}")

        models[model_name] = YOLO(str(weights_path))

        if RUN_EVALUATION:
            evaluate_model(model_name, data_yaml, weights_path, output_dir)

        if RUN_LOCALIZATION:
            evaluate_localization(model_name, models[model_name], dataset_dir, output_dir)

        if RUN_BENCHMARK:
            benchmark_inference(model_name, models[model_name], dataset_dir)

    if RUN_FAILURE_ANALYSIS:
        find_failure_cases(output_dir, num_cases=5)

    if RUN_VISUALIZATION:
        visualize_failure_cases(models, dataset_dir, output_dir)

    if RUN_THRESHOLD_ANALYSIS:
        evaluate_thresholds(models, data_yaml, output_dir)

    count_detections(models, dataset_dir, output_dir)