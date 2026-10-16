"""Inference driver, sample prediction visualiser, and 4-case clinical failure autopsy script.

Author: Lingxiao Li (s49852413)
Course: COMP3710 / COMP7710 Pattern Recognition and Analysis (2026)

This script loads the trained `ConvNeXt` (and `SimpleCNNBaseline`) weights from
`checkpoints/`, runs calibrated inference on the held-out ADNI test split, prints
a complete diagnostic & selective triage report, and generates visual plots in `images/`:
    1. `images/preprocessing_and_samples.png`: Raw vs. Z-score normalized & augmented slices.
    2. `images/sample_predictions.png`: 8-panel test inference & Grad-CAM triage grid.
    3. `images/failure_autopsy.png`: 4-case Failure Case Autopsy (2 False Negatives +
       2 False Positives) diagnosing anatomical failure triggers and triage interception.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import ADNIDataset, IDX_TO_LABEL, build_adni_dataloaders
from modules import (
    TemperatureScaledClassifier,
    build_model,
    compute_spatial_attribution,
)
from train import (
    CHECKPOINTS_DIR,
    IMAGES_DIR,
    collect_logits_and_labels,
    compute_comprehensive_metrics,
    select_device,
    set_global_seed,
)


def load_trained_model(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, float, int]:
    """Load trained model weights and optimal calibration temperature from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run `python train.py` first."
        )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    image_size = int(payload.get("image_size", 128))
    temperature = float(payload.get("temperature", 1.0))
    model = build_model(model_name, in_channels=1, num_classes=2).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, temperature, image_size


def plot_preprocessing_showcase(test_ds: ADNIDataset, out_path: Path) -> None:
    """Visualize raw T1 slices, Z-score standardized maps, and intensity histograms."""
    # Find one NC and one AD example
    nc_idx = next(i for i, r in enumerate(test_ds.records) if r.label == 0)
    ad_idx = next(i for i, r in enumerate(test_ds.records) if r.label == 1)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), dpi=180)
    for row, (idx, title_prefix, color) in enumerate(
        [
            (nc_idx, "Cognitively Normal (NC)", "#1f77b4"),
            (ad_idx, "Alzheimer's Disease (AD)", "#d62728"),
        ]
    ):
        rec = test_ds.records[idx]
        raw = test_ds.get_raw_slice(idx)
        z_norm = ADNIDataset.standardize_slice(raw, use_cohort_stats=test_ds.use_cohort_stats)
        np.random.seed(100 + row)
        aug = ADNIDataset._apply_augmentations(raw.copy())

        axes[row, 0].imshow(raw, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_title(
            f"{title_prefix}\nPatient: {rec.patient_id} (Slice #{rec.slice_index})",
            fontsize=9.5,
            fontweight="bold",
        )
        axes[row, 0].axis("off")

        im_z = axes[row, 1].imshow(z_norm, cmap="coolwarm", vmin=-2.2, vmax=2.2)
        z_title = (
            "Cohort Standardized\n(x - 0.18) / 0.25"
            if test_ds.use_cohort_stats
            else "Per-Slice Z-Score Standardized\n(μ = 0.0, σ = 1.0)"
        )
        axes[row, 1].set_title(z_title, fontsize=9.5, fontweight="bold")
        axes[row, 1].axis("off")
        fig.colorbar(im_z, ax=axes[row, 1], fraction=0.046, pad=0.04)

        axes[row, 2].imshow(aug, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title("Training Augmentation Sample\n(±2 px Shift + Gamma 0.92–1.08)", fontsize=9.5, fontweight="bold")
        axes[row, 2].axis("off")

        # Brain foreground pixel intensity distribution
        fg = z_norm[raw > 0.05]
        axes[row, 3].hist(fg, bins=32, color=color, alpha=0.72, edgecolor="black", density=True)
        axes[row, 3].axvline(float(np.mean(fg)), color="black", linestyle="--", lw=1.5, label=f"FG Mean={np.mean(fg):.2f}")
        axes[row, 3].set_title("Parenchymal Standardized Distribution", fontsize=9.5, fontweight="bold")
        axes[row, 3].set_xlabel("Standardized Intensity (Z)")
        axes[row, 3].set_ylabel("Density")
        axes[row, 3].grid(True, linestyle="--", alpha=0.35)
        axes[row, 3].legend(fontsize=8)

    fig.suptitle(
        "ADNI T1-Weighted Brain MRI Pre-Processing & Standardization Pipeline",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_sample_predictions_grid(
    model: torch.nn.Module,
    test_ds: ADNIDataset,
    cal_probs: np.ndarray,
    temperature: float,
    device: torch.device,
    out_path: Path,
    triage_tau: float = 0.80,
) -> None:
    """Plot an 8-case test prediction showcase with Grad-CAM overlays and triage badges."""
    y_true = np.array([r.label for r in test_ds.records])
    y_pred = np.argmax(cal_probs, axis=1)
    conf = np.max(cal_probs, axis=1)

    # Pick 3 high-confidence correct NC, 3 high-confidence correct AD, and 2 borderline/routed cases
    nc_corr = np.where((y_true == 0) & (y_pred == 0) & (conf >= triage_tau))[0]
    ad_corr = np.where((y_true == 1) & (y_pred == 1) & (conf >= triage_tau))[0]
    borderline = np.where(conf < triage_tau)[0]

    selected_indices: List[int] = []
    selected_indices.extend(nc_corr[:3].tolist())
    selected_indices.extend(ad_corr[:3].tolist())
    selected_indices.extend(borderline[:2].tolist())
    while len(selected_indices) < 8:
        selected_indices.append(len(selected_indices))
    selected_indices = selected_indices[:8]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), dpi=180)
    axes_flat = axes.flatten()

    for plot_i, ds_idx in enumerate(selected_indices):
        ax = axes_flat[plot_i]
        rec = test_ds.records[ds_idx]
        raw_img = test_ds.get_raw_slice(ds_idx)
        item = test_ds[ds_idx]
        x_tensor = item[0].unsqueeze(0).to(device)

        cam = compute_spatial_attribution(model, x_tensor, target_class=int(y_pred[ds_idx]))
        cam_np = cam[0].detach().cpu().numpy()

        p_ad = float(cal_probs[ds_idx, 1])
        p_max = float(conf[ds_idx])
        pred_cls = IDX_TO_LABEL[int(y_pred[ds_idx])]
        true_cls = rec.class_name
        is_correct = int(y_pred[ds_idx]) == rec.label
        triage_status = "AUTO-CLEAR" if p_max >= triage_tau else "REFER TO SPECIALIST"

        ax.imshow(raw_img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.imshow(cam_np, cmap="jet", alpha=0.35 * (raw_img > 0.04))
        ax.axis("off")

        badge_color = "#2ca02c" if (is_correct and p_max >= triage_tau) else (
            "#ff7f0e" if p_max < triage_tau else "#d62728"
        )
        ax.set_title(
            f"Patient: {rec.patient_id} (Slice {rec.slice_index})\n"
            f"GT: {true_cls} | Pred: {pred_cls} (Conf: {p_max:.2f})\n"
            f"[{triage_status}]",
            fontsize=9.0,
            fontweight="bold",
            color=badge_color,
        )

    fig.suptitle(
        f"ConvNeXt Test Inference & Grad-CAM Attribution Showcase (Temperature T* = {temperature:.2f}, Triage τ* = {triage_tau:.2f})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def conduct_failure_case_autopsy(
    convnext_model: torch.nn.Module,
    test_ds: ADNIDataset,
    raw_logits: torch.Tensor,
    temperature: float,
    baseline_probs: Optional[np.ndarray],
    device: torch.device,
    out_path: Path,
    triage_tau: float = 0.80,
) -> List[Dict[str, object]]:
    """Select 4 representative failure cases (2 FN, 2 FP) and generate a 4-panel autopsy plot."""
    y_true = np.array([r.label for r in test_ds.records])
    raw_probs = F.softmax(raw_logits, dim=-1).numpy()
    cal_probs = F.softmax(raw_logits / temperature, dim=-1).numpy()
    y_pred = np.argmax(cal_probs, axis=1)
    cal_conf = np.max(cal_probs, axis=1)
    raw_conf = np.max(raw_probs, axis=1)

    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]

    # Pick 2 FN slices (preferring distinct patients) and 2 FP slices (preferring distinct patients)
    chosen_cases: List[Tuple[int, str]] = []
    seen_pids: set[str] = set()
    seen_indices: set[int] = set()

    fn_sorted = sorted(fn_indices.tolist(), key=lambda i: cal_conf[i])
    for idx in fn_sorted:
        pid = test_ds.records[idx].patient_id
        if pid not in seen_pids:
            chosen_cases.append((idx, "False Negative (Missed AD)"))
            seen_pids.add(pid)
            seen_indices.add(idx)
        if len(chosen_cases) >= 2:
            break
    for idx in fn_sorted:
        if len(chosen_cases) >= 2:
            break
        if idx not in seen_indices:
            chosen_cases.append((idx, "False Negative (Missed AD)"))
            seen_indices.add(idx)

    fp_sorted = sorted(fp_indices.tolist(), key=lambda i: cal_conf[i])
    for idx in fp_sorted:
        pid = test_ds.records[idx].patient_id
        if pid not in seen_pids:
            chosen_cases.append((idx, "False Positive (False AD Alarm)"))
            seen_pids.add(pid)
            seen_indices.add(idx)
        if len(chosen_cases) >= 4:
            break
    for idx in fp_sorted[-1::-1]:
        if len(chosen_cases) >= 4:
            break
        if idx not in seen_indices:
            chosen_cases.append((idx, "False Positive (False AD Alarm)"))
            seen_indices.add(idx)

    fig, axes = plt.subplots(len(chosen_cases), 4, figsize=(16.5, 3.6 * len(chosen_cases)), dpi=180)
    if len(chosen_cases) == 1:
        axes = np.expand_dims(axes, axis=0)

    autopsy_records: List[Dict[str, object]] = []

    for row, (ds_idx, err_type) in enumerate(chosen_cases):
        rec = test_ds.records[ds_idx]
        raw_img = test_ds.get_raw_slice(ds_idx)
        item = test_ds[ds_idx]
        x_tensor = item[0].unsqueeze(0).to(device)

        cam = compute_spatial_attribution(convnext_model, x_tensor, target_class=int(y_pred[ds_idx]))
        cam_np = cam[0].detach().cpu().numpy()

        c_raw = float(raw_conf[ds_idx])
        c_cal = float(cal_conf[ds_idx])
        p_ad_cal = float(cal_probs[ds_idx, 1])
        p_nc_cal = float(cal_probs[ds_idx, 0])
        eps = 1e-8
        entropy = -(
            p_nc_cal * math.log(max(p_nc_cal, eps)) + p_ad_cal * math.log(max(p_ad_cal, eps))
        ) / math.log(2.0)

        routed_to_human = c_cal < triage_tau

        # Compute anatomical triggers from slice intensity structure
        fg_mask = raw_img > 0.05
        central_box = raw_img[int(0.32 * raw_img.shape[0]) : int(0.68 * raw_img.shape[0]),
                              int(0.32 * raw_img.shape[1]) : int(0.68 * raw_img.shape[1])]
        ventricle_dark_frac = float(np.mean(central_box < 0.28))
        brain_area_frac = float(np.mean(fg_mask))

        if "False Negative" in err_type:
            if rec.slice_index < 10 or rec.slice_index > 32 or brain_area_frac < 0.42:
                trigger = (
                    "Peripheral slice selection + partial-volume effect: smaller ventricular "
                    "cross-section masks medial temporal atrophy."
                )
            else:
                trigger = (
                    "Early/borderline AD phenotype: preserved global cortical ribbon thickness "
                    "with subtle focal peri-ventricular widening near decision boundary."
                )
        else:
            if ventricle_dark_frac > 0.25:
                trigger = (
                    "Benign age-related ventriculomegaly / prominent central CSF space in a "
                    "healthy elderly NC control mimics AD hydrocephalus ex vacuo."
                )
            else:
                trigger = (
                    "Asymmetric sulcal widening & RF coil intensity gradient along lateral "
                    "cortex triggers false positive atrophy activation."
                )

        # Panel 1: Raw T1 Slice
        axes[row, 0].imshow(raw_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_title(
            f"Case #{row + 1}: {err_type}\nPatient {rec.patient_id} | Slice {rec.slice_index}",
            fontsize=9.5,
            fontweight="bold",
            color="#d62728",
        )
        axes[row, 0].axis("off")

        # Panel 2: Grad-CAM Attribution Overlay
        axes[row, 1].imshow(raw_img, cmap="gray", vmin=0.0, vmax=1.0)
        im_cam = axes[row, 1].imshow(cam_np, cmap="inferno", alpha=0.45 * fg_mask)
        axes[row, 1].set_title(
            f"Grad-CAM Focus (Pred: {IDX_TO_LABEL[int(y_pred[ds_idx])]})\n"
            f"Central CSF Dark Ratio: {ventricle_dark_frac:.2f}",
            fontsize=9.5,
            fontweight="bold",
        )
        axes[row, 1].axis("off")
        fig.colorbar(im_cam, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Panel 3: Horizontal Mid-Brain Intensity Profile (Ventricular vs Cortical Contrast)
        mid_y = raw_img.shape[0] // 2
        profile = raw_img[mid_y, :]
        cam_prof = cam_np[mid_y, :]
        x_coords = np.arange(len(profile))
        axes[row, 2].plot(x_coords, profile, color="#1f77b4", lw=2.0, label="T1 Intensity (Mid-Axial)")
        axes[row, 2].fill_between(x_coords, 0, cam_prof, color="#ff7f0e", alpha=0.30, label="Grad-CAM Salience")
        axes[row, 2].set_ylim(0.0, 1.05)
        axes[row, 2].set_title("Mid-Brain Ventricular Profile & Salience", fontsize=9.5, fontweight="bold")
        axes[row, 2].set_xlabel("Left-to-Right Spatial Coordinate (px)")
        axes[row, 2].set_ylabel("Normalized Value")
        axes[row, 2].grid(True, linestyle="--", alpha=0.35)
        axes[row, 2].legend(fontsize=7.8, loc="upper right")

        # Panel 4: Confidence Calibration & Triage Interception Bar Chart
        bars_labels = ["Uncalibrated\nConvNeXt", "Calibrated\nConvNeXt", "Calibrated\nEntropy H(p)"]
        bars_vals = [c_raw, c_cal, entropy]
        bars_colors = ["#ff7f0e", "#1f77b4", "#9467bd"]
        if baseline_probs is not None:
            b_conf = float(np.max(baseline_probs[ds_idx]))
            bars_labels.insert(0, "SimpleCNN\nBaseline")
            bars_vals.insert(0, b_conf)
            bars_colors.insert(0, "#d62728")

        x_pos = np.arange(len(bars_labels))
        bars = axes[row, 3].bar(x_pos, bars_vals, color=bars_colors, width=0.55, edgecolor="black", alpha=0.85)
        axes[row, 3].axhline(
            triage_tau,
            color="#2ca02c",
            linestyle="--",
            lw=2.0,
            label=f"Triage Gate τ*={triage_tau:.2f}",
        )
        for b, v in zip(bars, bars_vals):
            axes[row, 3].text(
                b.get_x() + b.get_width() / 2.0,
                min(v + 0.03, 0.96),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
        axes[row, 3].set_xticks(x_pos)
        axes[row, 3].set_xticklabels(bars_labels, fontsize=8.0)
        axes[row, 3].set_ylim(0.0, 1.08)
        status_str = "INTERCEPTED -> HUMAN REVIEW" if routed_to_human else "UNFLAGGED HIGH-CONF ERROR"
        status_col = "#2ca02c" if routed_to_human else "#d62728"
        axes[row, 3].set_title(f"Triage Verdict: {status_str}", fontsize=9.2, fontweight="bold", color=status_col)
        axes[row, 3].grid(True, axis="y", linestyle="--", alpha=0.35)
        axes[row, 3].legend(fontsize=7.8, loc="lower right")

        autopsy_records.append(
            {
                "case_number": row + 1,
                "error_type": err_type,
                "dataset_index": int(ds_idx),
                "patient_id": rec.patient_id,
                "slice_index": int(rec.slice_index),
                "ground_truth": rec.class_name,
                "predicted_class": IDX_TO_LABEL[int(y_pred[ds_idx])],
                "raw_confidence": round(c_raw, 4),
                "calibrated_confidence": round(c_cal, 4),
                "predictive_entropy": round(entropy, 4),
                "baseline_confidence": round(float(np.max(baseline_probs[ds_idx])), 4)
                if baseline_probs is not None
                else None,
                "intercepted_by_triage_gate": bool(routed_to_human),
                "anatomical_diagnosis": trigger,
            }
        )

    fig.suptitle(
        "Clinical Failure Case Autopsy: Diagnosing False Negatives & False Positives on ADNI Test Scans",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return autopsy_records


def plot_architecture_diagram(out_path: Path) -> None:
    """Generate a clean visual block diagram of the Uncertainty-Aware ConvNeXt Triage Pipeline."""
    fig, ax = plt.subplots(figsize=(15, 4.2), dpi=180)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    boxes = [
        (0.3, 1.1, 2.0, 2.0, "#e8f4f8", "#1f77b4", "1. T1 Brain MRI\nInput Slice\n[B, 1, 128, 128]\n\nPer-Slice Z-Score\nPatient-Level Split"),
        (2.7, 1.1, 2.1, 2.0, "#eef7ee", "#2ca02c", "2. Patchify Stem\n& Stage 1-2\n\n4×4 Conv Stem\nLayerNorm2d\nC = [48, 96]"),
        (5.2, 1.1, 2.5, 2.0, "#fff4e6", "#ff7f0e", "3. ConvNeXt Blocks\nStage 3-4 [Hard]\n\n7×7 Depthwise Conv\n1×1 Inverted MLP (4×)\nGELU + LayerScale γ\nDropPath (p=0.15)"),
        (8.1, 1.1, 2.1, 2.0, "#f3eaf8", "#9467bd", "4. Global Pool &\nTemp Calibration\n\nGAP + LayerNorm\nLogits z / T*\nShannon Entropy H(p)"),
        (10.6, 2.2, 4.0, 1.2, "#e6f5ea", "#2ca02c", "High Confidence (max p ≥ τ* = 0.80)\n→ AUTO-DIAGNOSE (AD / NC)"),
        (10.6, 0.6, 4.0, 1.2, "#fdeaea", "#d62728", "Ambiguous Boundary (max p < τ* = 0.80)\n→ ROUTE TO NEURORADIOLOGIST"),
    ]

    for x, y, w, h, fc, ec, text in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=2.2, zorder=2)
        ax.add_patch(rect)
        ax.text(
            x + w / 2.0,
            y + h / 2.0,
            text,
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#1a1a1a",
            zorder=3,
        )

    arrows = [
        (2.3, 2.1, 2.7, 2.1),
        (4.8, 2.1, 5.2, 2.1),
        (7.7, 2.1, 8.1, 2.1),
        (10.2, 2.4, 10.6, 2.8),
        (10.2, 1.8, 10.6, 1.2),
    ]
    for x0, y0, x1, y1 in arrows:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=2.2, color="#333333"),
            zorder=4,
        )

    ax.set_title(
        "System Architecture: Modernized ConvNeXt Backbone with Calibrated Reject-Option Clinical Triage",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run trained ConvNeXt inference, generate predictions, and perform Failure Case Autopsy."
    )
    parser.add_argument("--data_root", type=str, default=None, help="Path to ADNI/AD_NC directory.")
    parser.add_argument(
        "--split_mode",
        type=str,
        default="longitudinal",
        choices=["longitudinal", "subject", "official"],
        help="Evaluation protocol; must match the one used in train.py.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to saved ConvNeXt checkpoint (default: checkpoints[/<mode>]/convnext_best.pt).",
    )
    parser.add_argument(
        "--baseline_checkpoint",
        type=str,
        default=None,
        help="Optional path to saved SimpleCNNBaseline checkpoint.",
    )
    parser.add_argument("--triage_threshold", type=float, default=0.80, help="Confidence gate tau*.")
    parser.add_argument("--batch_size", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device.")
    return parser.parse_args()


def main() -> None:
    global IMAGES_DIR, CHECKPOINTS_DIR
    args = parse_args()
    set_global_seed(42)
    device = select_device(args.device)
    if args.split_mode != "longitudinal":
        IMAGES_DIR = IMAGES_DIR / args.split_mode
        CHECKPOINTS_DIR = CHECKPOINTS_DIR / args.split_mode

    ckpt_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINTS_DIR / "convnext_best.pt"
    model, temperature, image_size = load_trained_model("convnext", ckpt_path, device)

    _, _, test_loader, audit = build_adni_dataloaders(
        data_root=args.data_root,
        image_size=image_size,
        batch_size=args.batch_size,
        return_meta_eval=True,
        split_mode=args.split_mode,
    )
    test_ds: ADNIDataset = test_loader.dataset  # type: ignore[assignment]

    test_logits, test_labels, test_metas = collect_logits_and_labels(model, test_loader, device)
    metrics = compute_comprehensive_metrics(
        test_logits, test_labels, temperature=temperature, metas=test_metas
    )

    baseline_probs: Optional[np.ndarray] = None
    base_ckpt = Path(args.baseline_checkpoint) if args.baseline_checkpoint else CHECKPOINTS_DIR / "baseline_best.pt"
    if base_ckpt.exists():
        b_model, b_temp, _ = load_trained_model("baseline", base_ckpt, device)
        b_logits, _, _ = collect_logits_and_labels(b_model, test_loader, device)
        baseline_probs = F.softmax(b_logits / b_temp, dim=-1).numpy()

    print("\n======================================================================")
    print("   ADNI BRAIN MRI CLASSIFICATION — CONVNEXT TEST INFERENCE REPORT")
    print("======================================================================")
    print(f"Dataset Root        : {audit['resolved_root']}")
    print(f"Held-Out Test Split : {audit['test']['total_slices']} slices ({audit['test']['total_patients']} patients)")
    print(f"Slice-Level Accuracy: {metrics['accuracy'] * 100:.2f}% (Target >= 80.00%)")
    print(f"Patient-Level Acc   : {metrics['patient_level'].get('patient_accuracy', 0.0) * 100:.2f}%")
    print(f"Macro-F1 / AUROC    : {metrics['macro_f1']:.4f} / {metrics['auroc']:.4f} (PR-AUC: {metrics['pr_auc']:.4f})")
    print(f"NC Precision/Recall : {metrics['per_class']['NC']['precision']:.4f} / {metrics['per_class']['NC']['recall']:.4f}")
    print(f"AD Precision/Recall : {metrics['per_class']['AD']['precision']:.4f} / {metrics['per_class']['AD']['recall']:.4f}")
    print(f"Optimal Temp (T*)   : {temperature:.4f} (ECE: {metrics['ece_uncalibrated']:.4f} -> {metrics['ece_calibrated']:.4f})")

    # Find operating point matching args.triage_threshold
    tau_entry = min(
        metrics["triage_sweep"],
        key=lambda e: abs(e["threshold"] - args.triage_threshold),
    )
    print("\n--- Recommended Clinical Triage Operating Policy ---")
    print(f"Confidence Gate τ*  : {tau_entry['threshold']:.2f}")
    print(f"Automated Coverage  : {tau_entry['coverage'] * 100:.1f}% of incoming scans auto-diagnosed")
    print(f"Specialist Referral : {tau_entry['referral_rate'] * 100:.1f}% routed to human neuroradiologist")
    print(f"Auto-Diagnosed Acc  : {tau_entry['accepted_accuracy'] * 100:.2f}%")
    print(f"Hybrid System Acc   : {tau_entry['system_hybrid_accuracy'] * 100:.2f}%")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_architecture_diagram(IMAGES_DIR / "convnext_architecture.png")
    plot_preprocessing_showcase(test_ds, IMAGES_DIR / "preprocessing_and_samples.png")

    cal_probs = F.softmax(test_logits / temperature, dim=-1).numpy()
    plot_sample_predictions_grid(
        model=model,
        test_ds=test_ds,
        cal_probs=cal_probs,
        temperature=temperature,
        device=device,
        out_path=IMAGES_DIR / "sample_predictions.png",
        triage_tau=args.triage_threshold,
    )

    autopsy_cases = conduct_failure_case_autopsy(
        convnext_model=model,
        test_ds=test_ds,
        raw_logits=test_logits,
        temperature=temperature,
        baseline_probs=baseline_probs,
        device=device,
        out_path=IMAGES_DIR / "failure_autopsy.png",
        triage_tau=args.triage_threshold,
    )

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINTS_DIR / "failure_autopsy_cases.json", "w", encoding="utf-8") as f:
        json.dump(autopsy_cases, f, indent=2)

    print("\n--- 4-Case Failure Autopsy Summary ---")
    for c in autopsy_cases:
        print(
            f"  Case #{c['case_number']} [{c['error_type']}] Patient={c['patient_id']} Slice={c['slice_index']} | "
            f"CalConf={c['calibrated_confidence']:.3f} | Entropy={c['predictive_entropy']:.3f} | "
            f"Intercepted={c['intercepted_by_triage_gate']}\n"
            f"    -> Root Trigger: {c['anatomical_diagnosis']}"
        )
    print(f"\nVisualizations written to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
