# NOTE: Extensively assisted by Claude Sonnet 5.0 (Medium). General structure (initial pseudocode) provided

"""
evaluate.py: evaluates the trained YOLO model on the ISIC 2017 test split.
Reports localisation IoU against the task target (>= 0.8), YOLO detection
metrics (mAP), per-class classification metrics, and melanoma-vs-rest
sensitivity/specificity/AUC.

Usage:
    python evaluate.py --weights path/to/best.pt --data path/to/yolo --out eval_out
"""
import os # standard Python library
import argparse # standard Python library
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score  # for reporting of values NOTE: Claude Sonnet 5.0 (Medium)
from pathlib import Path

from modules import load_trained

# configurations
from config import (WEIGHTS, DATA_DIR, EVAL_SPLIT, EVAL_DEVICE, RUN_MAP, EVAL_OUT,
                    CLASS_NAMES, CONF_EVAL, IOU_TARGET, IMG_EXTS, NO_DET, IMGSZ, write_data_yaml)


def get_image_paths(split_dir):
    """Sorted list of image paths in a given split directory/folder."""
    return sorted(os.path.join(split_dir, i)
                  for i in os.listdir(split_dir)
                  if i.lower().endswith(IMG_EXTS))

def read_true_class(label_path):
    """Returns the class ID from a given YOLO annotation.txt file."""
    with open(label_path) as f:
        return int(f.readline().split()[0]) # returns the ACTUAL true class of a given image based on ISIC2017

def read_true_box(label_path):
    """Returns bounding box coordinates from a given YOLO annotation.txt file - for IoU.""" # NOTE: Claude Sonnet 5.0 (Medium)
    with open(label_path) as f:
        parts = f.readline().split()
    return tuple(float(x) for x in parts[1:5])

# NOTE: Claude Sonnet 5.0 (Medium)
def predict_image(model_current, img_path, device):
    """Returns all detections for one image as (class_id, conf, (cx, cy, w, h))."""
    results = model_current.predict(img_path, imgsz=IMGSZ, conf=CONF_EVAL,
                                    device=device, verbose=False)[0] # using ultralytics .predict function
    detections = []  # class_id, conf score, box coordinates
    for b in results.boxes:  # = holds all detections for one image
        class_id = int(b.cls.item())
        conf = float(b.conf.item())
        box = tuple(b.xywhn[0].tolist()) # normalised (cx, cy, w, h)
        detections.append((class_id, conf, box))
    return detections

def select_prediction(detections):
    """Returns the highest-confidence detection."""
    if not detections:
        return None  # ie. no lesion detected
    return max(detections, key=lambda d: d[1]) # ie. returns the best prediction the model can offer


def class_scores(detections):
    """Returns max confidence score per class."""
    scores = [0.0] * len(CLASS_NAMES)  # initialisation
    for class_id, conf, _ in detections:
        scores[class_id] = max(scores[class_id], conf) # maximum returned
    return scores

# NOTE: Claude Sonnet 5.0 (Medium)
def box_iou(box_a, box_b):
    """IoU of two normalised (cx, cy, w, h) boxes. Will be utilised to determine how well the models bounding
    boxes actually compare to the true bounding boxes provided via/in ISIC2017"""
    def corners(b):
        cx, cy, w, h = b
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2 

    ax1, ay1, ax2, ay2 = corners(box_a)
    bx1, by1, bx2, by2 = corners(box_b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return inter / union if union > 0 else 0.0

# NOTE: Claude Sonnet 5.0 (Medium) assisted throughout
def run_predictions(model_current, img_dir, label_dir, device, select=select_prediction):
    """Predicts every image; stores true class, predicted class, scores and IoU.""" # ie. fusing all the previous functions
    image_paths = get_image_paths(img_dir)
    results = []
    no_detections = 0  # tracking files with no lesions detected within them
    for img_path in image_paths: # for every image
        stem = os.path.splitext(os.path.basename(img_path))[0] # getting associated stem
        label_path = os.path.join(label_dir, stem + ".txt") # getting associated label (path)
        true_class = read_true_class(label_path) # getting the actual class of the given image
        true_box = read_true_box(label_path)  # read once, reused below
        detections = predict_image(model_current, img_path, device)  # actual detected lesion

        prediction = select(detections) # most likely prediction of what the image actually is according to the model
        if prediction is None: # ie. no lesion detected thus no prediction
            pred_class, pred_box = NO_DET, None 
            iou, best_iou = 0.0, 0.0
            no_detections += 1
        else:
            pred_class, pred_box = prediction[0], prediction[2]
            iou = box_iou(pred_box, true_box) # how well the models bounding box overlaps with the actual bounding box
            best_iou = max(box_iou(d[2], true_box) for d in detections) # best IoU for all the detections made

        scores = class_scores(detections)
        results.append((true_class, pred_class, scores, iou, best_iou, pred_box, true_box))
    return results, no_detections


# NOTE: General Structure provided by Claude Sonnet 5.0 (Medium), helped write/debug/format also
def compute_iou_metrics(ious, pred):
    """Summarises how well predicted boxex ovrlap with the true boxes, and 
    compares them against provided IoU target"""
    detected = pred != NO_DET # images with at least 1 detection
    mean_detected = ious[detected].mean() if detected.any() else 0.0 # average over detected images only
    return {
        "mean_all": ious.mean(), # over ALL images, misses = 0
        "mean_detected": mean_detected, # detected images only
        "share_gt_0.5": np.mean(ious > 0.5), # fraction of images with IoU >50%
        "share_at_target": np.mean(ious >= IOU_TARGET), # fraction of images that hit IoU target
        "n_detected": int(detected.sum()), # number of images with at least 1 detection
    }

def compute_per_class_iou(true, ious):
    """Mean IoU and share of images reaching the IoU target, for each true class.
    Misses count as IoU 0, so they are included in the mean."""
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = true == c  # images whose true class is c
        if mask.any():
            per_class[name] = {
                "mean_iou": float(ious[mask].mean()),
                "share_at_target": float(np.mean(ious[mask] >= IOU_TARGET)),
                "share_gt_0.5": float(np.mean(ious[mask] > 0.5)),
                "n": int(mask.sum()),
            }
    return per_class

def compute_map(model, data_yaml, device, out_dir, split):
    """YOLO built-in validation on given split - gives standard detection metrics -->
        overall and per-class mAP50, mAP50-95, P (Precision), R (Recall), plus AP at 
        IoU 0.8."""
    m = model.val(data=data_yaml, split=split, imgsz=IMGSZ, device=device,
                  workers=0, project=out_dir, name="yolo_val", exist_ok=True,
                  verbose=False) # configuration/s
    box = m.box # box
    per_class = {}
    for i, c in enumerate(box.ap_class_index):
        per_class[CLASS_NAMES[c]] = {"P": box.p[i], "R": box.r[i],
                                     "mAP50": box.ap50[i], "mAP50-95": box.ap[i]} # storing of values
    return {
        "map50": box.map50, "map": box.map, "p": box.mp, "r": box.mr,
        "ap_at_0.8": box.all_ap[:, 6].mean(),  # column 6 = IoU threshold 0.80
        "per_class": per_class,
    }

def compute_confusion_matrix(true, pred):
    """Counts how often each true classes was predicted as each class
        Rows = true class, columns = predicted class plus a final 'no detection' column."""
    n = len(CLASS_NAMES)  # number of columns
    matrix = np.zeros((n, n + 1), dtype=int)
    for t, p in zip(true, pred):
        matrix[t, n if p == NO_DET else p] += 1 # NOTE: Claude Sonnet 5.0 (Medium)
    return matrix

def compute_class_metrics(matrix):
    """Per-class precision, recall and F1 from the confusion matrix."""
    metrics = {}
    # NOTE: Claude Sonnet 5.0 (Medium)
    for c in range(len(CLASS_NAMES)):
        tp = matrix[c, c]
        fp = matrix[:, c].sum() - tp
        fn = matrix[c, :].sum() - tp # includes no-detection cases
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[CLASS_NAMES[c]] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics

def compute_melanoma_metrics(true, pred, scores):
    """Melanoma vs rest: sensitivity, specificity and AUC. Important for malignancy 
        identification"""
    # NOTE: Claude Sonnet 5.0 (Medium)
    y_true = (true == 0) # images that are truly melanoma
    y_pred = (pred == 0) # images that are predicted melanoma
    tp = np.sum(y_true & y_pred) # melanomas correctly flagged
    fn = np.sum(y_true & ~y_pred) # melanoms missed (BAD BAD BAD)
    tn = np.sum(~y_true & ~y_pred) # non-melanomsd correctly left unflagged
    fp = np.sum(~y_true & y_pred) # non-melanomas flagged as melanomas
    sensitivity = tp / (tp + fn) if tp + fn else 0.0 # RECALL - melanomas caught
    specificity = tn / (tn + fp) if tn + fp else 0.0 # non-melanomas cleared correctly
    auc = roc_auc_score(y_true, scores[:, 0])
    return sensitivity, specificity, auc

# NOTE: Claude Sonnet 5.0 (Medium) - updated on 29/10/2026 to include true class vs model bounding boxes
def save_visualisations(image_paths, indices, pred, scores, ious, pred_boxes, true, true_boxes, out_dir):
    """Saves example images showing exactly what was scored: the top-confidence predicted box (red),
    the mask-derived true box (green), and a label with true class, predicted class, confidence and IoU.
    Unlike model.predict(conf=0.25), low-confidence boxes are still drawn."""
    os.makedirs(out_dir, exist_ok=True)
    for rank, i in enumerate(indices):
        img = cv2.imread(image_paths[i])
        s = 900 / max(img.shape[:2])                     # shrink very large images
        img = cv2.resize(img, None, fx=s, fy=s)
        h, w = img.shape[:2]

        def draw(box, colour):
            cx, cy, bw, bh = box
            cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                          (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), colour, 3)

        draw(true_boxes[i], (0, 255, 0))                 # true: green
        if pred_boxes[i] is not None:
            draw(pred_boxes[i], (0, 0, 255))             # predicted: red
        pred_name = CLASS_NAMES[pred[i]] if pred[i] != NO_DET else "none"
        lines = [f"true: {CLASS_NAMES[true[i]]}",
                 f"pred: {pred_name}  conf {scores[i].max():.2f}  IoU {ious[i]:.2f}"]
        for k, text in enumerate(lines):
            cv2.putText(img, text, (10, 35 + 35 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        name = os.path.basename(image_paths[i])
        cv2.imwrite(os.path.join(out_dir, f"{rank:02d}_{name}"), img)

# NOTE: Formatting of below for data collection was written by Claude Sonnet 5.0 (Medium)

def main():
    # directories and the generated data yaml .
    img_dir = os.path.join(DATA_DIR, EVAL_SPLIT, "images")
    label_dir = os.path.join(DATA_DIR, EVAL_SPLIT, "labels")
    data_yaml = write_data_yaml(DATA_DIR, Path(EVAL_OUT) / "data_eval.yaml")

    model = load_trained(WEIGHTS)  # run at IMGSZ from config.py
    print(f"Weights: {WEIGHTS}\nSplit: {EVAL_SPLIT}, imgsz: {IMGSZ}, device: {EVAL_DEVICE}")
    results, no_det = run_predictions(model, img_dir, label_dir, EVAL_DEVICE)
    print(f"No detection: {no_det}/{len(results)}")  # images with 0 detections

    true, pred, scores, ious, best_ious, pred_boxes, true_boxes = zip(*results)  # unbundle results
    true, pred = np.array(true), np.array(pred)
    scores, ious, best_ious = np.array(scores), np.array(ious), np.array(best_ious)

    # --- localisation: task target ---
    iou = compute_iou_metrics(ious, pred)
    print("\n=== Localisation (top-confidence box vs mask-derived box) ===")
    print(f"Mean IoU, all images (misses = 0): {iou['mean_all']:.3f}")
    print(f"Mean IoU, detected only ({iou['n_detected']} images): {iou['mean_detected']:.3f}")
    print(f"Images with IoU > 0.5: {iou['share_gt_0.5']:.1%}")
    print(f"Images with IoU >= {IOU_TARGET}: {iou['share_at_target']:.1%}")
    print(f"Target (mean IoU >= {IOU_TARGET}) met: {iou['mean_all'] >= IOU_TARGET}")
    print(f"Mean IoU, best detection per image: {best_ious.mean():.3f}")
    print(f"Split evaluated: {EVAL_SPLIT}")
    print("\n=== Per-class localisation (grouped by TRUE class) ===")
    for name, v in compute_per_class_iou(true, ious).items():
        print(f"{name}: mean IoU {v['mean_iou']:.3f}, IoU > 0.5: {v['share_gt_0.5']:.1%}, "
              f"IoU >= {IOU_TARGET}: {v['share_at_target']:.1%}, n={v['n']}")

    # --- detection metrics (Ultralytics) ---
    if RUN_MAP:
        det = compute_map(model, data_yaml, EVAL_DEVICE, str(EVAL_OUT), EVAL_SPLIT)
        print(f"\n=== Detection metrics (Ultralytics val, {EVAL_SPLIT} split) ===")
        print(f"Overall: mAP50 {det['map50']:.3f}, mAP50-95 {det['map']:.3f}, "
              f"P {det['p']:.3f}, R {det['r']:.3f}")
        print(f"AP at IoU 0.8 (mean over classes): {det['ap_at_0.8']:.3f}")
        for name, m in det["per_class"].items():
            print(f"{name}: P {m['P']:.3f}, R {m['R']:.3f}, "
                  f"mAP50 {m['mAP50']:.3f}, mAP50-95 {m['mAP50-95']:.3f}")

    # --- classification ---
    matrix = compute_confusion_matrix(true, pred)
    print("\n=== Classification (top-confidence box per image) ===")
    print("Confusion matrix (rows = true, last column = no detection):\n", matrix)
    for name, m in compute_class_metrics(matrix).items():
        print(f"{name}: precision {m['precision']:.3f}, recall {m['recall']:.3f}, F1 {m['f1']:.3f}")
    print(f"Overall accuracy: {np.mean(true == pred):.3f}")
    sens, spec, auc = compute_melanoma_metrics(true, pred, scores)
    print(f"Melanoma vs rest: sensitivity {sens:.3f}, specificity {spec:.3f}, AUC {auc:.3f}")

    image_paths = get_image_paths(img_dir)
    # example images: up to 10 failures and 10 correct for failure analysis--> produces outputs/examples/...
    wrong = [i for i in range(len(true)) if true[i] != pred[i]][:10]
    right = [i for i in range(len(true)) if true[i] == pred[i]][:10]
    save_visualisations(image_paths, wrong + right, pred, scores, ious, pred_boxes,
                        true, true_boxes, os.path.join(EVAL_OUT, "examples"))
    print(f"\nExamples saved to {EVAL_OUT}/examples")


if __name__ == "__main__":
    main()
