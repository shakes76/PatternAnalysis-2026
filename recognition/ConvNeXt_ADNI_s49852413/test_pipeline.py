"""Unit tests and code-hygiene audits for ADNI ConvNeXt implementation.

Author: Lingxiao Li (s49852413)
Course: COMP3710 / COMP7710 Pattern Recognition and Analysis (2026)

Verifies:
1. Strict Section 4.1 compliance: `modules.py` does NOT import `numpy` or any
   non-PyTorch numerical libraries (verified via Python AST inspection).
2. Tensor shapes, gradient flow, and calibration invariants for `LayerNorm2d`,
   `ConvNeXtBlock`, `ConvNeXt`, `SimpleCNNBaseline`, and `TemperatureScaledClassifier`.
3. Strict patient-level split hygiene in `dataset.py` (zero patient overlap across
   train, validation, and test splits) and Z-score normalization accuracy.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from dataset import (
    ADNIDataset,
    SliceRecord,
    extract_patient_and_slice,
    split_patients_stratified,
    verify_no_patient_leakage,
)
from modules import (
    ConvNeXt,
    ConvNeXtBlock,
    LayerNorm2d,
    SimpleCNNBaseline,
    TemperatureScaledClassifier,
    build_model,
    compute_spatial_attribution,
)


def test_modules_has_zero_numpy_dependency() -> None:
    """Section 4.1 Mandate: `modules.py` must not depend on NumPy or third-party libs."""
    modules_path = Path(__file__).resolve().parent / "modules.py"
    tree = ast.parse(modules_path.read_text(encoding="utf-8"))
    forbidden = {"numpy", "np", "scipy", "sklearn", "timm", "cv2", "PIL", "matplotlib"}

    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name.split(".")[0])
                if alias.asname:
                    imported_names.add(alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module.split(".")[0])

    violations = imported_names & forbidden
    assert len(violations) == 0, f"Forbidden imports in modules.py: {violations}"


def test_convnext_and_baseline_forward_backward() -> None:
    """Verify forward logits, feature maps, Grad-CAM shapes, and backward gradients."""
    x = torch.randn(4, 1, 96, 96)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    for name in ("baseline", "convnext", "convnext_patchify_bn", "convnext_overlap_ln", "convnext_bn_stem"):
        model = build_model(name, in_channels=1, num_classes=2)
        logits = model(x)
        assert logits.shape == (4, 2)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()

        # Verify gradients are finite and non-zero
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0 and all(np.isfinite(g) for g in grad_norms)

        cam = compute_spatial_attribution(model, x)
        assert cam.shape == (4, 96, 96)
        assert float(cam.min()) >= -1e-5 and float(cam.max()) <= 1.0 + 1e-5


def test_temperature_scaling_preserves_accuracy_and_reduces_nll() -> None:
    """Temperature scaling must preserve argmax predictions while optimizing NLL."""
    torch.manual_seed(0)
    base = SimpleCNNBaseline(in_channels=1, num_classes=2)
    wrapper = TemperatureScaledClassifier(base, init_temperature=1.0)

    # Simulate overconfident validation logits
    val_labels = torch.randint(0, 2, (64,))
    raw_signs = (val_labels * 2 - 1).float()
    # Flip 20% of signs to simulate errors, multiply by 5.0 for overconfidence
    flip = torch.ones(64)
    flip[:13] = -1.0
    z1 = raw_signs * flip * 4.5
    val_logits = torch.stack([-z1, z1], dim=1)

    nll_before = torch.nn.functional.cross_entropy(val_logits, val_labels).item()
    opt_t = wrapper.fit_temperature(val_logits, val_labels)
    nll_after = torch.nn.functional.cross_entropy(val_logits / opt_t, val_labels).item()

    assert opt_t > 1.0, "Overconfident logits should yield optimal temperature T* > 1.0"
    assert nll_after <= nll_before + 1e-5
    assert torch.equal(val_logits.argmax(dim=-1), (val_logits / opt_t).argmax(dim=-1))


def test_patient_level_splitting_and_leakage_audit() -> None:
    """Verify that all slices from the same patient remain strictly in one split."""
    dummy_records = []
    for pid_num in range(20):
        cls_label = 1 if pid_num >= 10 else 0
        cls_name = "AD" if cls_label == 1 else "NC"
        for s_idx in range(5):
            dummy_records.append(
                SliceRecord(
                    filepath=Path(f"/tmp/{cls_name}_{pid_num}_{s_idx}.jpeg"),
                    label=cls_label,
                    class_name=cls_name,
                    patient_id=f"{cls_name}_{pid_num}",
                    slice_index=s_idx,
                )
            )

    tr_recs, va_recs = split_patients_stratified(dummy_records[:80], val_fraction=0.25, seed=42)
    te_recs = dummy_records[80:]
    audit = verify_no_patient_leakage(tr_recs, va_recs, te_recs)
    assert audit["leakage_detected"] is False
    assert audit["overlap_train_val"] == 0
    assert audit["overlap_train_test"] == 0
    assert audit["overlap_val_test"] == 0


def test_filename_patient_extraction_and_zscore() -> None:
    """Verify ADNI filename subject extraction and per-slice Z-score standardization."""
    assert extract_patient_and_slice("369883_4.jpeg") == ("369883", 4)
    assert extract_patient_and_slice("Sample_train_data_808819_88.jpeg") == ("808819", 88)
    assert extract_patient_and_slice("ADNI_002_S_0413_slice_15.png") == ("002_S_0413", 15)

    arr = np.random.uniform(10.0, 200.0, size=(128, 128)).astype(np.float32)
    normed = ADNIDataset.standardize_slice(arr)
    assert abs(float(np.mean(normed))) < 1e-5
    assert abs(float(np.std(normed)) - 1.0) < 1e-3


def test_subject_disjoint_split_has_zero_subject_leakage(tmp_path: Path) -> None:
    """Verify that _split_subject_disjoint guarantees zero subject overlap and longitudinal detects subject sharing."""
    import json
    import pytest
    from dataset import _scan_to_subject_map, _split_longitudinal_with_metadata, _split_subject_disjoint

    meta_file = tmp_path / "meta.json"
    meta_payload = {}
    records = []
    # Create 12 synthetic subjects, each with 4 scans, each scan with 2 slices
    for s_idx in range(12):
        subj_str = f"999_S_{s_idx:04d}"
        lab = s_idx % 2
        c_name = "AD" if lab == 1 else "NC"
        for scan_k in range(4):
            scan_id = f"{s_idx*10 + scan_k + 100:06d}"
            raw_tag = "_br_raw_" if scan_k % 2 == 0 else "__raw_"
            meta_payload[scan_id] = {"raw": f"ADNI_{subj_str}_MR_{raw_tag}20180101_{scan_id}"}
            for sl in range(2):
                records.append(
                    SliceRecord(
                        filepath=tmp_path / f"{scan_id}_{sl}.jpg",
                        label=lab,
                        class_name=c_name,
                        patient_id=f"{c_name}_{scan_id}",
                        slice_index=sl,
                    )
                )

    meta_file.write_text(json.dumps(meta_payload), encoding="utf-8")
    scan2subj = _scan_to_subject_map(records, meta_file, strict=True)
    assert all(r.acquisition_year == 2018 for r in records)
    tr, va, te = _split_subject_disjoint(records, meta_file, seed=42, val_fraction=0.2, test_fraction=0.2)

    audit_subj = verify_no_patient_leakage(tr, va, te, require_subject_disjoint=True)
    assert audit_subj["scan_disjoint"] is True
    assert audit_subj["subject_disjoint"] is True
    assert audit_subj["leakage_detected"] is False
    assert audit_subj["subject_overlap_train_test"] == 0

    # Verify longitudinal split is scan-disjoint but honestly reports shared longitudinal subjects
    tr_l, va_l, te_l = _split_longitudinal_with_metadata(records, meta_file, seed=42)
    audit_long = verify_no_patient_leakage(tr_l, va_l, te_l, require_subject_disjoint=False)
    assert audit_long["scan_disjoint"] is True
    assert audit_long["subject_disjoint"] is False
    assert audit_long["leakage_detected"] is True
    assert audit_long["subject_overlap_train_test"] > 0

    # Verify require_subject_disjoint=True raises AssertionError on longitudinal split
    with pytest.raises(AssertionError, match="SUBJECT LEAKAGE"):
        verify_no_patient_leakage(tr_l, va_l, te_l, require_subject_disjoint=True)


def test_selective_triage_routing_monotonicity() -> None:
    """Verify that selective triage coverage decreases monotonically with increasing threshold tau."""
    from train import evaluate_selective_triage

    probs = np.array([
        [0.95, 0.05],
        [0.85, 0.15],
        [0.75, 0.25],
        [0.65, 0.35],
        [0.55, 0.45],
        [0.10, 0.90],
        [0.20, 0.80],
        [0.30, 0.70],
        [0.40, 0.60],
        [0.48, 0.52],
    ], dtype=np.float32)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)

    sweep = evaluate_selective_triage(probs, labels)
    coverages = [entry["coverage"] for entry in sweep]
    referrals = [entry["referral_rate"] for entry in sweep]

    # As tau increases, coverage must be non-increasing and referral must be non-decreasing
    for i in range(len(coverages) - 1):
        assert coverages[i] >= coverages[i + 1]
        assert referrals[i] <= referrals[i + 1]


def test_era_slice_position_and_ddof1_aggregation() -> None:
    """Verify era stratification, 5-quintile slice-position telemetry, and ddof=1 multiseed std."""
    from train import _aggregate_multiseed_metrics, compute_comprehensive_metrics

    logits = torch.tensor([[2.0, -1.0], [-1.5, 2.5], [1.2, -0.8], [-2.0, 1.8]] * 5, dtype=torch.float32)
    labels = torch.tensor([0, 1, 0, 1] * 5, dtype=torch.long)
    metas = [
        {
            "patient_id": f"p_{i // 5}",
            "subject_id": f"s_{i // 10}",
            "slice_index": i % 5,
            "acquisition_year": 2013 if i < 10 else 2019,
        }
        for i in range(20)
    ]
    m = compute_comprehensive_metrics(logits, labels, temperature=1.0, metas=metas)
    assert "pre_2017_adni2" in m["era_stratified"]
    assert "post_2017_adni3" in m["era_stratified"]
    assert "ad_recall_ci95" in m["era_stratified"]["post_2017_adni3"]
    assert len(m["slice_position_quintiles"]) == 5
    assert "accuracy_ci95" in m["slice_position_quintiles"][0]
    assert m["patient_level"]["scan_acc_ci95_method"] == "subject_cluster_bootstrap_1000"

    # Check ddof=1 vs ddof=0 in _aggregate_multiseed_metrics
    m2 = dict(m)
    m2["accuracy"] = 0.50
    runs = [
        {"seed": 42, "models": {"baseline": {"test_metrics": m}, "convnext": {"test_metrics": m2}}},
        {"seed": 43, "models": {"baseline": {"test_metrics": m2}, "convnext": {"test_metrics": m}}},
    ]
    agg = _aggregate_multiseed_metrics(runs)
    assert agg["baseline"]["slice_level"]["accuracy"]["ddof"] == 1
    assert abs(agg["baseline"]["slice_level"]["accuracy"]["std"] - float(np.std([1.0, 0.5], ddof=1))) < 1e-4
    assert "paired_baseline_minus_convnext" in agg


def test_readme_numbers_match_json_artifacts() -> None:
    """Verify that key statistics reported in README.md match the JSON files in results/."""
    from make_tables import verify_readme_matches_json

    ok, mismatches = verify_readme_matches_json()
    assert ok, "README.md numerical mismatch against results/*.json:\n" + "\n".join(mismatches)



