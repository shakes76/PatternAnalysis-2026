import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from modules import ISICBaselineDetector, compute_box_iou_tensor


def load_yolo_gt_boxes(label_path: Path, img_w: int, img_h: int) -> torch.Tensor:
    """Loads ground truth YOLO normalized boxes and converts to [x1, y1, x2, y2] pixel coordinates."""
    if not label_path.exists():
        return torch.empty((0, 4), dtype=torch.float32)

    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = map(float, parts[:5])
            x1 = (xc - w / 2.0) * img_w
            y1 = (yc - h / 2.0) * img_h
            x2 = (xc + w / 2.0) * img_w
            y2 = (yc + h / 2.0) * img_h
            boxes.append([x1, y1, x2, y2])

    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def run_test_evaluation_and_autopsy(weights_path: str, test_dir: str, out_dir: str,
                                    iou_match_thresh: float = 0.5, top_k_fp: int = 10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    test_img_dir = Path(test_dir) / "images" / "test"
    test_lbl_dir = Path(test_dir) / "labels" / "test"
    img_paths = sorted(list(test_img_dir.glob("*.jpg")))

    if not img_paths:
        raise FileNotFoundError(f"No test images found in {test_img_dir}")

    model = ISICBaselineDetector(weights_path=weights_path)

    # Low confidence threshold (0.01) to sweep the full Precision-Recall curve & catch FPs
    start_t = time.time()
    preds = model.predict_images(source=[str(p) for p in img_paths], conf=0.01, iou=0.5, device=device)
    total_time = time.time() - start_t
    fps = len(img_paths) / total_time if total_time > 0 else 0.0

    all_detections = []  # (conf, is_tp, best_iou, img_path, pred_box_xyxy, gt_boxes_xyxy)
    best_ious_per_gt = []
    total_gt_boxes = 0

    for img_p, res in zip(img_paths, preds):
        img_h, img_w = res.orig_shape
        lbl_p = test_lbl_dir / f"{img_p.stem}.txt"
        gt_boxes = load_yolo_gt_boxes(lbl_p, img_w, img_h)
        total_gt_boxes += len(gt_boxes)

        pred_boxes = res.boxes.xyxy.cpu() if res.boxes is not None else torch.empty((0, 4))
        pred_confs = res.boxes.conf.cpu() if res.boxes is not None else torch.empty((0,))

        # Sort predictions by descending confidence
        if len(pred_confs) > 0:
            order = torch.argsort(pred_confs, descending=True)
            pred_boxes = pred_boxes[order]
            pred_confs = pred_confs[order]

        iou_matrix = compute_box_iou_tensor(pred_boxes, gt_boxes)

        # Track best localization IoU for each GT lesion (for standard conf >= 0.25)
        valid_mask = pred_confs >= 0.25
        if valid_mask.any() and len(gt_boxes) > 0:
            valid_ious = compute_box_iou_tensor(pred_boxes[valid_mask], gt_boxes)
            max_gt_ious, _ = valid_ious.max(dim=0)
            best_ious_per_gt.extend(max_gt_ious.tolist())
        elif len(gt_boxes) > 0:
            best_ious_per_gt.extend([0.0] * len(gt_boxes))

        matched_gt = set()
        for idx in range(len(pred_boxes)):
            conf = float(pred_confs[idx].item())
            pbox = pred_boxes[idx].tolist()
            best_iou = 0.0
            best_gt_idx = -1

            if len(gt_boxes) > 0:
                ious = iou_matrix[idx]
                best_iou_Tensor, gt_idx_Tensor = torch.max(ious, dim=0)
                best_iou = float(best_iou_Tensor.item())
                best_gt_idx = int(gt_idx_Tensor.item())

            if best_iou >= iou_match_thresh and best_gt_idx not in matched_gt:
                is_tp = 1
                matched_gt.add(best_gt_idx)
            else:
                is_tp = 0

            all_detections.append({
                "image_id": img_p.stem,
                "img_path": img_p,
                "conf": conf,
                "is_tp": is_tp,
                "iou_with_gt": best_iou,
                "pred_box": pbox,
                "gt_boxes": gt_boxes.tolist(),
            })

    # 1. Compute Precision-Recall Curve across all confidence thresholds
    det_df = pd.DataFrame(all_detections).sort_values("conf", ascending=False).reset_index(drop=True)
    det_df["cum_tp"] = det_df["is_tp"].cumsum()
    det_df["cum_fp"] = (1 - det_df["is_tp"]).cumsum()
    det_df["precision"] = det_df["cum_tp"] / (det_df["cum_tp"] + det_df["cum_fp"] + 1e-7)
    det_df["recall"] = det_df["cum_tp"] / max(total_gt_boxes, 1)

    pr_fig_path = out_path / "baseline_PR_curve.png"
    plt.figure(figsize=(7, 6))
    plt.plot(det_df["recall"], det_df["precision"], color="navy", lw=2, label="YOLOv8n Baseline P-R Frontier")
    plt.xlabel("Recall / Clinical Sensitivity (Fraction of True Lesions Detected)")
    plt.ylabel("Precision (1 - False Positive Rate)")
    plt.title("ISIC 2018 Test Set: Precision-Recall Curve (IoU >= 0.5)")
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pr_fig_path, dpi=200)
    plt.close()

    # 2. Extract Top-10 Highest-Confidence False Positives for Failure Autopsy
    fp_df = det_df[det_df["is_tp"] == 0].head(top_k_fp).copy()
    fp_csv_path = out_path / "top10_false_positives_autopsy.csv"
    fp_df[["image_id", "conf", "iou_with_gt", "pred_box"]].to_csv(fp_csv_path, index=False)

    # Render 2x5 Visual Grid of Top 10 False Positive Bounding Boxes vs Ground Truth
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(fp_df.iterrows()):
        img_bgr = cv2.imread(str(row["img_path"]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Draw Ground Truth Boxes in GREEN
        for gbox in row["gt_boxes"]:
            gx1, gy1, gx2, gy2 = map(int, gbox)
            cv2.rectangle(img_rgb, (gx1, gy1), (gx2, gy2), (0, 255, 0), 4)

        # Draw False Positive Predicted Box in RED
        px1, py1, px2, py2 = map(int, row["pred_box"])
        cv2.rectangle(img_rgb, (px1, py1), (px2, py2), (255, 0, 0), 4)

        axes[idx].imshow(img_rgb)
        axes[idx].set_title(
            f"FP #{idx+1}: {row['image_id']}\nConf: {row['conf']:.3f} | IoU: {row['iou_with_gt']:.2f}",
            fontsize=10,
        )
        axes[idx].axis("off")

    for j in range(len(fp_df), 10):
        axes[j].axis("off")

    plt.suptitle("Failure Autopsy: Top 10 High-Confidence False Positive Detections (Red = FP Pred, Green = GT)", fontsize=14)
    plt.tight_layout()
    autopsy_fig_path = out_path / "top10_fp_autopsy_grid.png"
    plt.savefig(autopsy_fig_path, dpi=200)
    plt.close()

    mean_iou = float(np.mean(best_ious_per_gt)) if best_ious_per_gt else 0.0
    prop_iou_80 = float(np.mean([iou >= 0.80 for iou in best_ious_per_gt])) if best_ious_per_gt else 0.0

    print("\n================ HELD-OUT TEST & FAILURE AUTOPSY SUMMARY ================")
    print(f"Test Set Images Evaluated:          {len(img_paths)}")
    print(f"Inference Throughput:               {fps:.2f} FPS")
    print(f"Mean Localization IoU (Conf>=0.25): {mean_iou:.4f} (Target >= 0.80)")
    print(f"Lesions Achieving IoU >= 0.80:      {prop_iou_80 * 100:.2f}%")
    print(f"Saved P-R Curve:                    {pr_fig_path.resolve()}")
    print(f"Saved Top-10 FP Autopsy Grid:       {autopsy_fig_path.resolve()}")
    print(f"Saved Top-10 FP Autopsy Table:      {fp_csv_path.resolve()}")
    print("=========================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Evaluation & Top-10 False Positive Autopsy")
    parser.add_argument("--weights", type=str, default="runs/isic_baseline/yolov8n_50ep/weights/best.pt",
                        help="Path to trained best.pt weights")
    parser.add_argument("--data_dir", type=str, default="./data/isic2018_yolo",
                        help="Path to prepared YOLO dataset root")
    parser.add_argument("--out_dir", type=str, default="./autopsy_results",
                        help="Directory to save P-R curve and Top-10 FP autopsy visualisations")
    args = parser.parse_args()

    run_test_evaluation_and_autopsy(
        weights_path=args.weights,
        test_dir=args.data_dir,
        out_dir=args.out_dir,
    )