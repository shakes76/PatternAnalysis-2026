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

    # Pick 3 high-confidence correct NC, 3 high-confidence correct AD, and 2 borderline/routed cases from 8 distinct scans/subjects
    nc_corr = np.where((y_true == 0) & (y_pred == 0) & (conf >= triage_tau))[0]
    ad_corr = np.where((y_true == 1) & (y_pred == 1) & (conf >= triage_tau))[0]
    borderline = np.where(conf < triage_tau)[0]

    selected_indices: List[int] = []
    seen_patients: set[str] = set()

    def _pick_distinct(candidates: np.ndarray, k: int) -> None:
        for idx_val in candidates.tolist():
            rec_cand = test_ds.records[int(idx_val)]
            key = rec_cand.subject_id or rec_cand.patient_id
            if key not in seen_patients:
                seen_patients.add(key)
                selected_indices.append(int(idx_val))
                if len(selected_indices) >= k:
                    break

    _pick_distinct(nc_corr, 3)
    _pick_distinct(ad_corr, 6)
    _pick_distinct(borderline, 8)
    for fallback_idx in range(len(test_ds.records)):
        if len(selected_indices) >= 8:
            break
        rec_fb = test_ds.records[fallback_idx]
        key_fb = rec_fb.subject_id or rec_fb.patient_id
        if key_fb not in seen_patients:
            seen_patients.add(key_fb)
            selected_indices.append(fallback_idx)
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
        f"ConvNeXt Test Inference & Stage-3 (6×6) Grad-CAM Attribution (T* = {temperature:.2f}, Val-Selected τ* = {triage_tau:.2f})",
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
    triage_tau: float = 0.75,
    scan_triage_tau: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Select 4 failure cases (1 intercepted FN, 1 high-conf escaped FN, 1 intercepted FP, 1 high-conf escaped FP) and generate quantitative autopsy."""
    if scan_triage_tau is None:
        scan_triage_tau = triage_tau
    y_true = np.array([r.label for r in test_ds.records])
    raw_probs = F.softmax(raw_logits, dim=-1).numpy()
    cal_probs = F.softmax(raw_logits / temperature, dim=-1).numpy()
    y_pred = np.argmax(cal_probs, axis=1)
    cal_conf = np.max(cal_probs, axis=1)
    raw_conf = np.max(raw_probs, axis=1)

    # Pre-compute per-scan slice index ranges and 20-slice scan-level mean AD probability
    scan_slice_indices: Dict[str, List[int]] = {}
    scan_ad_probs: Dict[str, List[float]] = {}
    baseline_scan_ad_probs: Dict[str, List[float]] = {}
    for i, r in enumerate(test_ds.records):
        scan_slice_indices.setdefault(r.patient_id, []).append(int(r.slice_index))
        scan_ad_probs.setdefault(r.patient_id, []).append(float(cal_probs[i, 1]))
        if baseline_probs is not None:
            baseline_scan_ad_probs.setdefault(r.patient_id, []).append(float(baseline_probs[i, 1]))

    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]

    # Select 2 FN (1 intercepted < tau*, 1 escaped high-conf >= tau*) and 2 FP (1 intercepted < tau*, 1 escaped high-conf >= tau*)
    chosen_cases: List[Tuple[int, str]] = []
    seen_pids: set[str] = set()
    seen_indices: set[int] = set()

    def _pick_case(candidates: List[int], label_desc: str) -> None:
        for idx in candidates:
            pid = test_ds.records[idx].patient_id
            if pid not in seen_pids and idx not in seen_indices:
                chosen_cases.append((idx, label_desc))
                seen_pids.add(pid)
                seen_indices.add(idx)
                return
        for idx in candidates:
            if idx not in seen_indices:
                chosen_cases.append((idx, label_desc))
                seen_indices.add(idx)
                return

    fn_intercepted = sorted([int(i) for i in fn_indices if cal_conf[i] < triage_tau], key=lambda i: cal_conf[i])
    fn_escaped = sorted([int(i) for i in fn_indices if cal_conf[i] >= triage_tau], key=lambda i: -cal_conf[i])
    fn_all_asc = sorted([int(i) for i in fn_indices], key=lambda i: cal_conf[i])
    fn_all_desc = sorted([int(i) for i in fn_indices], key=lambda i: -cal_conf[i])

    _pick_case(fn_intercepted or fn_all_asc, "False Negative (Intercepted Borderline AD)")
    _pick_case(fn_escaped or fn_all_desc, "False Negative (High-Conf Escaped AD)")

    fp_intercepted = sorted([int(i) for i in fp_indices if cal_conf[i] < triage_tau], key=lambda i: cal_conf[i])
    fp_escaped = sorted([int(i) for i in fp_indices if cal_conf[i] >= triage_tau], key=lambda i: -cal_conf[i])
    fp_all_asc = sorted([int(i) for i in fp_indices], key=lambda i: cal_conf[i])
    fp_all_desc = sorted([int(i) for i in fp_indices], key=lambda i: -cal_conf[i])

    _pick_case(fp_intercepted or fp_all_asc, "False Positive (Intercepted Borderline NC)")
    _pick_case(fp_escaped or fp_all_desc, "False Positive (High-Conf Escaped NC)")

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

        # Also compute 6x6 coarse grid peak coordinates
        cam_6x6 = F.adaptive_avg_pool2d(cam.unsqueeze(1), (6, 6))[0, 0].detach().cpu().numpy()
        peak_r, peak_c = np.unravel_index(int(np.argmax(cam_6x6)), cam_6x6.shape)

        c_raw = float(raw_conf[ds_idx])
        c_cal = float(cal_conf[ds_idx])
        p_ad_cal = float(cal_probs[ds_idx, 1])
        p_nc_cal = float(cal_probs[ds_idx, 0])
        eps = 1e-8
        entropy = -(
            p_nc_cal * math.log(max(p_nc_cal, eps)) + p_ad_cal * math.log(max(p_ad_cal, eps))
        ) / math.log(2.0)

        routed_to_human = c_cal < triage_tau

        # Quantitative slice and scan telemetry
        fg_mask = raw_img > 0.05
        central_box = raw_img[
            int(0.32 * raw_img.shape[0]) : int(0.68 * raw_img.shape[0]),
            int(0.32 * raw_img.shape[1]) : int(0.68 * raw_img.shape[1]),
        ]
        ventricle_dark_frac = float(np.mean(central_box < 0.28))
        brain_area_frac = float(np.mean(fg_mask))

        s_indices = scan_slice_indices[rec.patient_id]
        s_min, s_max = min(s_indices), max(s_indices)
        rel_slice_pos = float((rec.slice_index - s_min) / max(1, s_max - s_min))
        quintile_idx = min(5, int(rel_slice_pos * 5.0) + 1)
        scan_mean_p_ad = float(np.mean(scan_ad_probs[rec.patient_id]))
        scan_conf = float(max(scan_mean_p_ad, 1.0 - scan_mean_p_ad))
        scan_pred_label = 1 if scan_mean_p_ad >= 0.5 else 0
        rescued_by_scan_vote = bool(scan_pred_label == rec.label)
        intercepted_by_scan_gate = bool(scan_conf < scan_triage_tau)

        rescue_note = (
            f"20-slice scan mean p(AD)={scan_mean_p_ad:.2f} (scan_conf={scan_conf:.2f}) RESCUES this slice error at scan level."
            if rescued_by_scan_vote
            else (
                f"20-slice scan mean p(AD)={scan_mean_p_ad:.2f} (scan_conf={scan_conf:.2f} < τ*_scan={scan_triage_tau:.2f}) "
                f"misclassifies the scan but is INTERCEPTED by the scan-level triage gate."
                if intercepted_by_scan_gate
                else f"20-slice scan mean p(AD)={scan_mean_p_ad:.2f} (scan_conf={scan_conf:.2f} >= τ*_scan={scan_triage_tau:.2f}) "
                f"also misclassifies the full scan and ESCAPES scan-level triage."
            )
        )
        trigger = (
            f"Slice {rec.slice_index} in [{s_min}..{s_max}] (rel_pos={rel_slice_pos:.2f}, Quintile Q{quintile_idx}); "
            f"foreground={brain_area_frac:.2f}, central dark-CSF ratio={ventricle_dark_frac:.2f}, "
            f"Stage-3 (6×6) Grad-CAM peak=({int(peak_r)},{int(peak_c)}). {rescue_note}"
        )

        # Panel 1: Raw T1 Slice
        axes[row, 0].imshow(raw_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_title(
            f"Case #{row + 1}: {err_type}\nScan {rec.patient_id} ({rec.subject_id}) | Slice {rec.slice_index}",
            fontsize=9.0,
            fontweight="bold",
            color="#d62728",
        )
        axes[row, 0].axis("off")

        # Panel 2: Grad-CAM Attribution Overlay
        axes[row, 1].imshow(raw_img, cmap="gray", vmin=0.0, vmax=1.0)
        im_cam = axes[row, 1].imshow(cam_np, cmap="inferno", alpha=0.45 * fg_mask)
        axes[row, 1].set_title(
            f"Stage-3 (6×6) Grad-CAM (Pred: {IDX_TO_LABEL[int(y_pred[ds_idx])]})\n"
            f"Central Dark CSF: {ventricle_dark_frac:.2f} | Scan p(AD): {scan_mean_p_ad:.2f}",
            fontsize=9.0,
            fontweight="bold",
        )
        axes[row, 1].axis("off")
        fig.colorbar(im_cam, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Panel 3: Horizontal Mid-Sagittal Intensity Profile (Anterior-Posterior Contrast)
        mid_y = raw_img.shape[0] // 2
        profile = raw_img[mid_y, :]
        cam_prof = cam_np[mid_y, :]
        x_coords = np.arange(len(profile))
        axes[row, 2].plot(x_coords, profile, color="#1f77b4", lw=2.0, label="T1 Intensity (Mid-Sagittal)")
        axes[row, 2].fill_between(x_coords, 0, cam_prof, color="#ff7f0e", alpha=0.30, label="Grad-CAM Salience")
        axes[row, 2].set_ylim(0.0, 1.05)
        axes[row, 2].set_title("Mid-Sagittal Intensity Profile & Salience", fontsize=9.5, fontweight="bold")
        axes[row, 2].set_xlabel("Anterior-Posterior Spatial Coordinate (px)")
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
            label=f"Val-Selected τ*={triage_tau:.2f}",
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
        status_str = "INTERCEPTED (< τ*) -> HUMAN REVIEW" if routed_to_human else "ESCAPED GATE (≥ τ*) -> HIGH-CONF ERROR"
        status_col = "#2ca02c" if routed_to_human else "#d62728"
        axes[row, 3].set_title(f"Triage Verdict: {status_str}", fontsize=8.8, fontweight="bold", color=status_col)
        axes[row, 3].grid(True, axis="y", linestyle="--", alpha=0.35)
        axes[row, 3].legend(fontsize=7.8, loc="lower right")

        autopsy_records.append(
            {
                "case_number": row + 1,
                "error_type": err_type,
                "dataset_index": int(ds_idx),
                "patient_id": rec.patient_id,
                "subject_id": rec.subject_id,
                "slice_index": int(rec.slice_index),
                "scan_slice_range": [int(s_min), int(s_max)],
                "relative_slice_position": round(rel_slice_pos, 4),
                "slice_position_quintile": f"Q{quintile_idx}",
                "ground_truth": rec.class_name,
                "predicted_class": IDX_TO_LABEL[int(y_pred[ds_idx])],
                "raw_confidence": round(c_raw, 4),
                "calibrated_confidence": round(c_cal, 4),
                "predictive_entropy": round(entropy, 4),
                "baseline_confidence": round(float(np.max(baseline_probs[ds_idx])), 4)
                if baseline_probs is not None
                else None,
                "baseline_scan_20slice_mean_prob_ad": round(float(np.mean(baseline_scan_ad_probs[rec.patient_id])), 4)
                if baseline_probs is not None and rec.patient_id in baseline_scan_ad_probs
                else None,
                "brain_foreground_fraction": round(brain_area_frac, 4),
                "central_csf_dark_fraction": round(ventricle_dark_frac, 4),
                "gradcam_stage2_peak_rc": [int(peak_r), int(peak_c)],
                "scan_20slice_mean_prob_ad": round(scan_mean_p_ad, 4),
                "scan_20slice_confidence": round(scan_conf, 4),
                "rescued_by_20slice_scan_vote": rescued_by_scan_vote,
                "intercepted_by_triage_gate": bool(routed_to_human),
                "intercepted_by_scan_triage_gate": intercepted_by_scan_gate,
                "anatomical_diagnosis": trigger,
            }
        )

    fig.suptitle(
        "Clinical Failure Case Autopsy: Intercepted Borderline Errors (< τ*) vs. High-Confidence Escaped Errors (≥ τ*)",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return autopsy_records


def plot_architecture_diagram(out_path: Path, triage_tau: float = 0.75) -> None:
    """Generate a clean visual block diagram matching the exact 96x96 ConvNeXt implementation."""
    fig, ax = plt.subplots(figsize=(15, 4.2), dpi=180)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    boxes = [
        (
            0.3,
            1.1,
            2.0,
            2.0,
            "#e8f4f8",
            "#1f77b4",
            "1. Sagittal T1 MRI\nInput Slice\n[B, 1, 96, 96]\n\nTight Brain ROI Crop\nCohort Norm (0.18, 0.25)",
        ),
        (
            2.7,
            1.1,
            2.1,
            2.0,
            "#eef7ee",
            "#2ca02c",
            "2. Patchify Stem\n& Stage 1–2\n\n4×4 Conv Stem (24×24)\nLayerNorm2d\nC=[32, 64] (24×24→12×12)",
        ),
        (
            5.2,
            1.1,
            2.5,
            2.0,
            "#fff4e6",
            "#ff7f0e",
            "3. ConvNeXt Blocks\nStage 3–4 [Hard]\n\n7×7 Depthwise Conv\n1×1 Inverted MLP (4×)\nC=[128, 256] (6×6→3×3)\nDropPath (p=0.10)",
        ),
        (
            8.1,
            1.1,
            2.1,
            2.0,
            "#f3eaf8",
            "#9467bd",
            "4. Calibration &\nStage-3 Grad-CAM\n\n6×6 Spatial Attribution\nLogits z / T*\nVal-Selected Gate τ*",
        ),
        (
            10.6,
            2.2,
            4.0,
            1.2,
            "#e6f5ea",
            "#2ca02c",
            f"High Confidence (max p ≥ τ* = {triage_tau:.2f})\n→ AUTO-CLEAR (Slice & 20-Slice Scan)",
        ),
        (
            10.6,
            0.6,
            4.0,
            1.2,
            "#fdeaea",
            "#d62728",
            f"Ambiguous Boundary (max p < τ* = {triage_tau:.2f})\n→ ROUTE TO NEURORADIOLOGIST",
        ),
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
            fontsize=8.6,
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
        "System Architecture: Modernized 96×96 ConvNeXt Backbone (2.18M Params) with Calibrated Selective Triage",
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
    parser.add_argument(
        "--triage_threshold",
        type=float,
        default=None,
        help="Confidence gate tau* (defaults to validation-selected tau* stored in checkpoint).",
    )
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
    ckpt_payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    use_cal_for_triage = bool(ckpt_payload.get("use_calibrated_for_triage", True))
    triage_tau = (
        float(args.triage_threshold)
        if args.triage_threshold is not None
        else float(ckpt_payload.get("val_selected_tau", 0.75))
    )
    scan_triage_tau = (
        float(args.triage_threshold)
        if args.triage_threshold is not None
        else float(ckpt_payload.get("val_selected_scan_tau", triage_tau))
    )

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
        test_logits,
        test_labels,
        temperature=temperature,
        metas=test_metas,
        use_calibrated_for_triage=use_cal_for_triage,
        selected_tau=triage_tau,
        selected_scan_tau=scan_triage_tau,
    )

    baseline_probs: Optional[np.ndarray] = None
    base_ckpt = Path(args.baseline_checkpoint) if args.baseline_checkpoint else CHECKPOINTS_DIR / "baseline_best.pt"
    if base_ckpt.exists():
        b_model, b_temp, _ = load_trained_model("baseline", base_ckpt, device)
        b_payload = torch.load(base_ckpt, map_location=device, weights_only=False)
        b_use_cal = bool(b_payload.get("use_calibrated_for_triage", True))
        b_logits, _, _ = collect_logits_and_labels(b_model, test_loader, device)
        baseline_probs = F.softmax(b_logits / (b_temp if b_use_cal else 1.0), dim=-1).numpy()

    print("\n======================================================================")
    print(f"   ADNI BRAIN MRI CLASSIFICATION — CONVNEXT TEST INFERENCE ({args.split_mode.upper()})")
    print("======================================================================")
    print(f"Dataset Root        : {audit['resolved_root']}")
    print(
        f"Held-Out Test Split : {audit['test']['total_slices']} slices "
        f"({audit['test']['total_patients']} scans, {audit['test']['unique_subjects']} subjects)"
    )
    print(f"Slice-Level Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Scan Mean-Prob Acc  : {metrics['patient_level'].get('scan_mean_prob_accuracy', 0.0) * 100:.2f}%")
    print(f"Scan Majority-Vote  : {metrics['patient_level'].get('scan_majority_vote_accuracy', 0.0) * 100:.2f}%")
    print(f"Subject-Level Acc   : {metrics['patient_level'].get('subject_accuracy', 0.0) * 100:.2f}%")
    print(f"Macro-F1 / AUROC    : {metrics['macro_f1']:.4f} / {metrics['auroc']:.4f} (PR-AUC: {metrics['pr_auc']:.4f})")
    print(f"NC Precision/Recall : {metrics['per_class']['NC']['precision']:.4f} / {metrics['per_class']['NC']['recall']:.4f}")
    print(f"AD Precision/Recall : {metrics['per_class']['AD']['precision']:.4f} / {metrics['per_class']['AD']['recall']:.4f}")
    print(f"Optimal Temp (T*)   : {temperature:.4f} (ECE: {metrics['ece_uncalibrated']:.4f} -> {metrics['ece_calibrated']:.4f})")

    tau_entry = min(
        metrics["triage_sweep"],
        key=lambda e: abs(e["threshold"] - triage_tau),
    )
    print("\n--- Validation-Selected Clinical Triage Operating Policy ---")
    print(f"Confidence Gate τ*  : {tau_entry['threshold']:.2f} (selected strictly on validation split; scan τ*_scan={scan_triage_tau:.2f})")
    print(f"Automated Coverage  : {tau_entry['coverage'] * 100:.1f}% of incoming slices auto-cleared")
    print(f"Specialist Referral : {tau_entry['referral_rate'] * 100:.1f}% routed to human neuroradiologist")
    print(f"Auto-Cleared Acc    : {tau_entry['accepted_accuracy'] * 100:.2f}%")
    print(f"Intercepted FN Rate : {tau_entry.get('intercepted_fn_fraction', 0.0) * 100:.1f}% of false negatives caught")
    print(f"Oracle Hybrid Acc   : {tau_entry['system_hybrid_accuracy'] * 100:.2f}% (assuming 100% specialist accuracy on referrals)")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_architecture_diagram(IMAGES_DIR / "convnext_architecture.png", triage_tau=triage_tau)
    plot_preprocessing_showcase(test_ds, IMAGES_DIR / "preprocessing_and_samples.png")

    cal_probs = F.softmax(test_logits / temperature, dim=-1).numpy()
    plot_sample_predictions_grid(
        model=model,
        test_ds=test_ds,
        cal_probs=cal_probs,
        temperature=temperature,
        device=device,
        out_path=IMAGES_DIR / "sample_predictions.png",
        triage_tau=triage_tau,
    )

    autopsy_cases = conduct_failure_case_autopsy(
        convnext_model=model,
        test_ds=test_ds,
        raw_logits=test_logits,
        temperature=temperature,
        baseline_probs=baseline_probs,
        device=device,
        out_path=IMAGES_DIR / "failure_autopsy.png",
        triage_tau=triage_tau,
        scan_triage_tau=scan_triage_tau,
    )

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINTS_DIR / "failure_autopsy_cases.json", "w", encoding="utf-8") as f:
        json.dump(autopsy_cases, f, indent=2)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"failure_autopsy_{args.split_mode}.json", "w", encoding="utf-8") as f:
        json.dump(autopsy_cases, f, indent=2)

    print("\n--- 4-Case Failure Autopsy Summary ---")
    for c in autopsy_cases:
        print(
            f"  Case #{c['case_number']} [{c['error_type']}] Scan={c['patient_id']} ({c['subject_id']}) Slice={c['slice_index']} | "
            f"CalConf={c['calibrated_confidence']:.3f} | Entropy={c['predictive_entropy']:.3f} | "
            f"Intercepted={c['intercepted_by_triage_gate']} | RescuedByScanVote={c['rescued_by_20slice_scan_vote']}\n"
            f"    -> Telemetry: {c['anatomical_diagnosis']}"
        )
    print(f"\nVisualizations written to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()

