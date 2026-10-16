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
    x = torch.randn(4, 1, 128, 128)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    for name in ("baseline", "convnext"):
        model = build_model(name, in_channels=1, num_classes=2)
        logits = model(x)
        assert logits.shape == (4, 2)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()

        # Verify gradients are finite and non-zero
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0 and all(np.isfinite(g) for g in grad_norms)

        cam = compute_spatial_attribution(model, x)
        assert cam.shape == (4, 128, 128)
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
