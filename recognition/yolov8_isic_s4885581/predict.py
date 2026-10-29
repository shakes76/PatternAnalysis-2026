import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time

from modules import ISICDetector, compute_box_iou_tensor


def load_yolo_gt_boxes(label_path: Path, img_w: int, img_h: int) -> torch.Tensor:
    if not label_path.exists():
        return torch.empty((0, 4), dtype=torch.float32)
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = map(float, parts[:5])
            boxes.append([(xc - w / 2.0) * img_w, (yc - h / 2.0) * img_h,
                          (xc + w / 2.0) * img_w, (yc + h / 2.0) * img_h])
    return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)


def evaluate_model_on_test(weights_path: str, test_img_paths: list, test_lbl_dir: Path,
                           device: str, iou_match_thresh: float = 0.5):
    model = ISICDetector(weights_path=weights_path)

    start_t = time.time()
    preds = model.predict_images(source=[str(p) for p in test_img_paths], conf=0.005, iou=0.5, device=device)
    elapsed = time.time() - start_t
    fps = len(test_img_paths) / elapsed if elapsed > 0 else 0.0

    detections = []
    gt_records = []
    total_gt = 0

    for img_p, res in zip(test_img_paths, preds):
        img_h, img_w = res.orig_shape
        gt_boxes = load_yolo_gt_boxes(test_lbl_dir / f"{img_p.stem}.txt", img_w, img_h)
        total_gt += len(gt_boxes)

        pred_boxes = res.boxes.xyxy.cpu() if res.boxes is not None else torch.empty((0, 4))
        pred_confs = res.boxes.conf.cpu() if res.boxes is not None else torch.empty((0,))

        if len(pred_confs) > 0:
            order = torch.argsort(pred_confs, descending=True)
            pred_boxes = pred_boxes[order]
            pred_confs = pred_confs[order]

        iou_mat = compute_box_iou_tensor(pred_boxes, gt_boxes)

        conf_mask = pred_confs >= 0.25
        best_gt_iou = 0.0
        best_gt_conf = 0.0
        if len(gt_boxes) > 0:
            if len(pred_boxes) > 0:
                best_gt_conf = float(pred_confs.max().item())
            if conf_mask.any():
                ious_25 = compute_box_iou_tensor(pred_boxes[conf_mask], gt_boxes)
                best_gt_iou = float(ious_25.max().item())

            gt_records.append({
                "image_id": img_p.stem,
                "img_path": img_p,
                "gt_boxes": gt_boxes.tolist(),
                "pred_boxes": pred_boxes[conf_mask].tolist() if conf_mask.any() else [],
                "max_conf": best_gt_conf,
                "best_iou_at_025": best_gt_iou,
            })

        matched_gt = set()
        for idx in range(len(pred_boxes)):
            conf = float(pred_confs[idx].item())
            pbox = pred_boxes[idx].tolist()
            best_iou, best_gt_idx = 0.0, -1
            if len(gt_boxes) > 0:
                best_iou_t, gt_idx_t = torch.max(iou_mat[idx], dim=0)
                best_iou, best_gt_idx = float(best_iou_t.item()), int(gt_idx_t.item())

            is_tp = 1 if (best_iou >= iou_match_thresh and best_gt_idx not in matched_gt) else 0
            if is_tp:
                matched_gt.add(best_gt_idx)

            detections.append({
                "image_id": img_p.stem,
                "img_path": img_p,
                "conf": conf,
                "is_tp": is_tp,
                "iou_with_gt": best_iou,
                "pred_box": pbox,
                "gt_boxes": gt_boxes.tolist(),
            })

    det_df = pd.DataFrame(detections).sort_values("conf", ascending=False).reset_index(drop=True)
    det_df["cum_tp"] = det_df["is_tp"].cumsum()
    det_df["cum_fp"] = (1 - det_df["is_tp"]).cumsum()
    det_df["precision"] = det_df["cum_tp"] / (det_df["cum_tp"] + det_df["cum_fp"] + 1e-7)
    det_df["recall"] = det_df["cum_tp"] / max(total_gt, 1)

    gt_df = pd.DataFrame(gt_records)
    mean_iou = float(gt_df["best_iou_at_025"].mean()) if not gt_df.empty else 0.0
    pct_iou_80 = float((gt_df["best_iou_at_025"] >= 0.80).mean() * 100) if not gt_df.empty else 0.0

    return {
        "det_df": det_df,
        "gt_df": gt_df,
        "total_gt": total_gt,
        "fps": fps,
        "mean_iou": mean_iou,
        "pct_iou_80": pct_iou_80,
    }


def compute_operating_point_table(det_df: pd.DataFrame, total_gt: int) -> pd.DataFrame:
    thresholds = [0.10, 0.20, 0.25, 0.35, 0.50, 0.70, 0.85]
    rows = []
    for tau in thresholds:
        sub = det_df[det_df["conf"] >= tau]
        tp = int(sub["is_tp"].sum())
        fp = int((1 - sub["is_tp"]).sum())
        fn = max(total_gt - tp, 0)
        sensitivity = tp / max(total_gt, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-7)
        fp_per_100_patients = (fp / max(total_gt, 1)) * 100.0

        rows.append({
            "Threshold (tau)": tau,
            "Sensitivity / Recall (%)": round(sensitivity * 100, 2),
            "Precision (%)": round(precision * 100, 2),
            "F1-Score": round(f1, 4),
            "True Positives (TP)": tp,
            "False Alarms (FP)": fp,
            "Missed Lesions (FN)": fn,
            "Unnecessary Biopsies / 100 Cases": round(fp_per_100_patients, 1),
        })
    return pd.DataFrame(rows)


def run_test_evaluation_and_autopsy(weights_path: str, test_dir: str, out_dir: str,
                                    iou_match_thresh: float = 0.5, top_k_fp: int = 10):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    test_img_dir = Path(test_dir) / "images" / "test"
    test_lbl_dir = Path(test_dir) / "labels" / "test"
    img_paths = sorted(list(test_img_dir.glob("*.jpg")))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_res = evaluate_model_on_test(weights_path, img_paths, test_lbl_dir, device, iou_match_thresh)
    det_df = eval_res["det_df"]

    # 1. Plot P-R Curve
    pr_fig_path = out_path / "baseline_PR_curve.png"
    plt.figure(figsize=(7, 6))
    plt.plot(det_df["recall"], det_df["precision"], color="navy", lw=2, label="YOLOv8n Baseline P-R Frontier")
    plt.xlabel("Recall / Clinical Sensitivity")
    plt.ylabel("Precision (1 - False Discovery Rate)")
    plt.title("ISIC Test Set: Precision-Recall Curve (IoU >= 0.5)")
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pr_fig_path, dpi=200)
    plt.close()

    # 2. Extract Top-10 False Positives for Autopsy Grid
    fp_df = det_df[det_df["is_tp"] == 0].head(top_k_fp).copy()
    fp_csv_path = out_path / "top10_false_positives_autopsy.csv"
    fp_df[["image_id", "conf", "iou_with_gt", "pred_box"]].to_csv(fp_csv_path, index=False)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(fp_df.iterrows()):
        img_bgr = cv2.imread(str(row["img_path"]))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        for gbox in row["gt_boxes"]:
            gx1, gy1, gx2, gy2 = map(int, gbox)
            cv2.rectangle(img_rgb, (gx1, gy1), (gx2, gy2), (0, 255, 0), 4)

        px1, py1, px2, py2 = map(int, row["pred_box"])
        cv2.rectangle(img_rgb, (px1, py1), (px2, py2), (255, 0, 0), 4)

        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"FP #{idx+1}: {row['image_id']}\nConf: {row['conf']:.3f} | IoU: {row['iou_with_gt']:.2f}", fontsize=9)
        axes[idx].axis("off")

    for j in range(len(fp_df), 10):
        axes[j].axis("off")

    plt.suptitle("Failure Autopsy: Top 10 False Positives (Red = FP Pred, Green = GT)", fontsize=14)
    plt.tight_layout()
    autopsy_fig_path = out_path / "top10_fp_autopsy_grid.png"
    plt.savefig(autopsy_fig_path, dpi=200)
    plt.close()

    print(f"\n[AUTOPSY COMPLETE] Mean IoU: {eval_res['mean_iou']:.4f} | FPS: {eval_res['fps']:.2f}")
    print(f"Saved P-R Curve: {pr_fig_path.resolve()}")
    print(f"Saved Autopsy Grid: {autopsy_fig_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Test Evaluation & Autopsy")
    parser.add_argument("--weights", type=str, default="runs/isic_baseline/yolov8n_100ep/weights/best.pt")
    parser.add_argument("--data_dir", type=str, default="./data/isic2018_yolo")
    parser.add_argument("--out_dir", type=str, default="./autopsy_results")
    args = parser.parse_args()
    run_test_evaluation_and_autopsy(args.weights, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()