"""Detection evaluation, class-aware matching and validation threshold sweep."""

import argparse
import time
from pathlib import Path
from typing import Any

from modules import CLASS_NAMES, load_trained_detector
from utils import read_config, save_json, sha256


def box_iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def operating_metrics(predictions, ground_truth, threshold, iou_threshold=0.5):
    truth = {row['image_id']: row['detections'] for row in ground_truth}
    predicted = {row['image_id']: row['detections'] for row in predictions}
    if len(truth) != len(ground_truth) or len(predicted) != len(predictions) or truth.keys() != predicted.keys():
        raise ValueError('Prediction/ground-truth image sets must agree without duplicates')
    counts = [{'tp': 0, 'fp': 0, 'fn': 0} for _ in CLASS_NAMES]
    for image_id, targets in truth.items():
        used = set()
        boxes = sorted(predicted[image_id], key=lambda box: box['confidence'], reverse=True)
        for box in boxes:
            if box['confidence'] < threshold:
                continue
            cls = box['class_id']
            matches = [(box_iou(box['xyxy'], gt['xyxy']), i) for i, gt in enumerate(targets)
                       if i not in used and gt['class_id'] == cls]
            overlap, match = max(matches, default=(0.0, -1))
            if match >= 0 and overlap >= iou_threshold:
                used.add(match)
                counts[cls]['tp'] += 1
            else:
                counts[cls]['fp'] += 1
        for i, target in enumerate(targets):
            if i not in used:
                counts[target['class_id']]['fn'] += 1
    def rates(row):
        tp, fp, fn = row['tp'], row['fp'], row['fn']
        return {**row, 'precision': tp/(tp+fp) if tp+fp else 0.0,
                'recall': tp/(tp+fn) if tp+fn else None}
    per_class = {name: rates(row) for name, row in zip(CLASS_NAMES, counts)}
    total = rates({key: sum(row[key] for row in counts) for key in ('tp', 'fp', 'fn')})
    return {'threshold': threshold, 'iou_threshold': iou_threshold, 'per_class': per_class,
            'overall': total, 'false_positives_per_image': total['fp']/len(truth) if truth else 0.0}


def evaluate_detector(
    weights: Path,
    data_yaml: Path,
    split: str,
    image_size: int,
    output_dir: Path,
    device: str = "0",
    threshold_sweep: bool = False,
    operating_conf: float | None = None,
) -> dict:
    if split not in ('val', 'test'):
        raise ValueError('Evaluation split must be val or test')
    if threshold_sweep and split != 'val':
        raise ValueError('Threshold selection is restricted to validation')
    if operating_conf is not None and not 0 <= operating_conf <= 1:
        raise ValueError('Operating confidence must be between 0 and 1')
    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    detector = load_trained_detector(weights)
    metrics = detector.val(data=str(Path(data_yaml).resolve()), split=split, imgsz=image_size,
                           conf=0.001, iou=0.7, device=device, augment=False,
                           project=str(Path(output_dir).resolve().parent),
                           name=Path(output_dir).name, exist_ok=False, plots=True)
    # Ultralytics can increment directories; use its actual save directory.
    output = Path(detector.validator.save_dir) if getattr(detector, 'validator', None) else Path(metrics.save_dir)
    report = {'split': split, 'weights_sha256': sha256(weights), 'imgsz': image_size,
              'map50': float(metrics.box.map50), 'map50_95': float(metrics.box.map),
              'per_class': {}, 'speed_ms': metrics.speed}
    for index, cls in enumerate(metrics.box.ap_class_index.tolist()):
        report['per_class'][CLASS_NAMES[int(cls)]] = {
            'precision': float(metrics.box.p[index]), 'recall': float(metrics.box.r[index]),
            'ap50': float(metrics.box.ap50[index]), 'ap50_95': float(metrics.box.ap[index]),
        }
    if threshold_sweep or operating_conf is not None:
        predictions, truth = collect_split(detector, data_yaml, split, image_size, device)
        save_json(output / f'{split}_predictions.json', predictions)
    if operating_conf is not None:
        report['operating_iou05'] = operating_metrics(predictions, truth, operating_conf, 0.5)
        report['operating_iou08'] = operating_metrics(predictions, truth, operating_conf, 0.8)
    if threshold_sweep:
        selected = choose_threshold(predictions, truth)
        save_json(output / 'threshold_selection.json', selected)
    save_json(output / 'evaluation.json', report)
    return report


def collect_split(detector, data_yaml, split, image_size, device):
    data = read_config(data_yaml)
    root = Path(data.get('path', Path(data_yaml).resolve().parent))
    if not root.is_absolute():
        root = Path(data_yaml).resolve().parent / root
    folder = root / data[split]
    predictions, truth = [], []
    for result in detector.predict(source=str(folder), imgsz=image_size, conf=0.001, iou=0.7,
                                   device=device, stream=True, verbose=False, augment=False):
        image_id = Path(result.path).stem
        height, width = result.orig_shape
        boxes = []
        if result.boxes is not None:
            boxes = [{'class_id': int(box.cls.item()), 'confidence': float(box.conf.item()),
                      'xyxy': box.xyxy[0].cpu().tolist()} for box in result.boxes]
        label = root / 'labels' / split / f'{image_id}.txt'
        if not label.exists():
            raise FileNotFoundError(label)
        targets = []
        for line in label.read_text().splitlines():
            cls, x, y, w, h = map(float, line.split())
            targets.append({'class_id': int(cls), 'xyxy': [(x-w/2)*width, (y-h/2)*height,
                                                          (x+w/2)*width, (y+h/2)*height]})
        predictions.append({'image_id': image_id, 'detections': boxes})
        truth.append({'image_id': image_id, 'detections': targets})
    return predictions, truth


def choose_threshold(
    validation_predictions: list[dict],
    validation_ground_truth: list[dict],
    policy: dict | None = None,
) -> dict:
    if not validation_ground_truth:
        raise ValueError('No validation cases')
    policy = dict(policy or {})
    name = policy.get('target_class', 'melanoma')
    if name not in CLASS_NAMES:
        raise ValueError('Unknown target class')
    grid = policy.get('thresholds', [i/20 for i in range(1, 20)])
    minimum = policy.get('min_recall')
    if (minimum is not None and not 0 <= minimum <= 1) or not grid or any(not 0 <= t <= 1 for t in grid):
        raise ValueError('Invalid threshold policy')
    rows = [operating_metrics(validation_predictions, validation_ground_truth, t) for t in grid]
    if minimum is None:
        return {'policy': policy, 'status': 'descriptive', 'selected': None,
                'curve': rows, 'iou08_at_selected': None}
    eligible = [row for row in rows if row['per_class'][name]['recall'] is not None
                and row['per_class'][name]['recall'] >= minimum]
    best = max(eligible, key=lambda row: (row['overall']['precision'], -row['false_positives_per_image'],
                                         row['threshold']), default=None)
    return {'policy': policy, 'status': 'met' if best else 'unmet', 'selected': best, 'curve': rows,
            'iou08_at_selected': operating_metrics(validation_predictions, validation_ground_truth,
                                                   best['threshold'], 0.8) if best else None}


def profile_inference(
    detector: Any,
    example_tensor: Any,
    warmup: int = 10,
    repeats: int = 100,
) -> dict:
    import torch

    if warmup < 1 or repeats < 1 or example_tensor.ndim != 4:
        raise ValueError('Invalid profiling arguments')
    network = detector.model
    was_training = network.training
    network.eval()
    cuda = example_tensor.is_cuda
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                network(example_tensor)
            if cuda:
                torch.cuda.synchronize(example_tensor.device)
                torch.cuda.reset_peak_memory_stats(example_tensor.device)
            start = time.perf_counter()
            for _ in range(repeats):
                network(example_tensor)
            if cuda:
                torch.cuda.synchronize(example_tensor.device)
            seconds = time.perf_counter() - start
    finally:
        network.train(was_training)
    return {'milliseconds_per_batch': seconds*1000/repeats,
            'images_per_second': repeats*example_tensor.shape[0]/seconds,
            'scope': 'network forward only; excludes preprocessing, NMS, IO and plotting',
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(example_tensor.device) if cuda else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--threshold-sweep', action='store_true')
    parser.add_argument('--operating-conf', type=float, help='Fixed threshold selected on validation')
    args = parser.parse_args()
    print(evaluate_detector(args.weights, args.data, args.split, args.imgsz,
                            args.output, args.device, args.threshold_sweep, args.operating_conf))
