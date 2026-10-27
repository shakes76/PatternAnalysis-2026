"""Inspect the fixed training sample and existing smoke checkpoints; no training."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (HipMRIPairDataset, collate_padded, crop_to_original,
                     validate_pair_geometry)
from losses import masked_one_hot, segmentation_cross_entropy
from metrics import SegmentationMetrics
from modules import build_model


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parent
    repo, workspace = project.parents[1], project.parents[3]
    parser.add_argument("--data-root", type=Path, default=workspace / "Final Report/HipMRI_2D_Improved_UNet_学习/data/keras_slices_data")
    parser.add_argument("--output", type=Path, default=Path("outputs/pipeline_review_20261025.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new review output path to preserve evidence.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    image_path = args.data_root / "keras_slices_train/case_004_week_0_slice_12.nii.gz"
    mask_path = args.data_root / "keras_slices_seg_train/seg_004_week_0_slice_12.nii.gz"
    image_nifti, mask_nifti = nib.load(image_path), nib.load(mask_path)
    difference = validate_pair_geometry(image_nifti, mask_nifti)
    raw_image, raw_mask = image_nifti.get_fdata(), mask_nifti.get_fdata()
    dataset = HipMRIPairDataset([(image_path, mask_path)])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_padded)
    batch = next(iter(loader))
    schema = json.loads((project / "label_schema.json").read_text(encoding="utf-8"))
    assert schema["num_classes"] == 6 and [item["id"] for item in schema["classes"]] == list(range(6))
    target, valid = batch["target"], batch["valid_mask"]
    one_hot = masked_one_hot(target, valid, 6)
    original_size = batch["original_sizes"][0]
    assert np.array_equal(crop_to_original(target[0], original_size).numpy(), raw_mask)
    example_row, example_col = np.argwhere(raw_mask == 3)[0].tolist()
    report = {
        "review_kind": "fixed_training_reference_pipeline", "status": "passed",
        "created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(),
        "performed_by": "Codex automation; user understanding remains unconfirmed",
        "scope": ["one fixed training sample", "existing independent-rebuild smoke checkpoints"],
        "test_split_read": False, "new_training_performed": False,
        "pair": {"image": image_path.name, "mask": mask_path.name,
                 "raw_shape": list(raw_image.shape), "affine_max_abs_difference": difference,
                 "selected_affine": image_nifti.affine.tolist(),
                 "declared_spatial_units": image_nifti.header.get_xyzt_units()[0]},
        "preprocessing": {"raw_image_range": [float(raw_image.min()), float(raw_image.max())],
                          "normalized_range": [float(batch["image"].min()), float(batch["image"].max())],
                          "batch_image_shape": list(batch["image"].shape), "target_shape": list(target.shape),
                          "one_hot_shape": list(one_hot.shape), "valid_pixels": int(valid.sum()),
                          "padding_pixels": int((~valid).sum()), "original_size": list(original_size),
                          "original_mask_ids_preserved_exactly": True,
                          "target_counts_by_id": torch.bincount(target[valid], minlength=6).tolist(),
                          "working_target_id": 5, "working_target_present_in_reference": bool((target[valid] == 5).any())},
        "one_hot_example": {"row": example_row, "column": example_col, "target_id": 3,
                            "vector": one_hot[0, :, example_row, example_col].tolist()},
        "label_mapping_boundary": schema["mapping_boundary"],
        "models": {},
        "claim_boundary": "This is a training-reference consistency review, not formal training, independent model evaluation, medical alignment proof, or user verification.",
    }
    image_device, target_device, valid_device = [batch[key].to(device) for key in ("image", "target", "valid_mask")]
    for name, folder, evidence_name in (
        ("baseline_unet", "baseline_smoke_20261025", "baseline_smoke.json"),
        ("improved_unet", "improved_smoke_20261025", "improved_smoke.json"),
    ):
        checkpoint_path = repo / "outputs" / folder / "best.pt"
        evidence = json.loads((project / "evidence" / evidence_name).read_text(encoding="utf-8"))
        assert sha256(checkpoint_path) == evidence["checkpoint_sha256"], "Checkpoint changed since smoke evidence."
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert checkpoint["model"] == name
        model = build_model(name, **checkpoint["model_config"]).to(device).eval()
        model.load_state_dict(checkpoint["model_state"])
        before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        with torch.inference_mode():
            logits = model(image_device)
            loss = segmentation_cross_entropy(logits, target_device, valid_device)
            prediction = logits.argmax(dim=1)
        assert all(torch.equal(before[key], value.detach().cpu()) for key, value in model.state_dict().items())
        assert abs(loss.item() - evidence["checkpoint_reload"]["reference_ce"]) < 1e-5
        metrics = SegmentationMetrics(6)
        metrics.update(prediction, target_device, valid_device)
        report["models"][name] = {
            "checkpoint_epoch": checkpoint["epoch"], "checkpoint_sha256": sha256(checkpoint_path),
            "eval_mode": not model.training, "logits_require_grad": logits.requires_grad,
            "logit_shape": list(logits.shape), "cross_entropy": float(loss.item()),
            "reference_ce_matches_previous_smoke": True, "model_state_unchanged": True,
            "restored_prediction_shape": list(crop_to_original(prediction[0], original_size).shape),
            "metrics": metrics.compute(),
        }
        del model, before, logits, loss, prediction
    report["source_sha256"] = {
        name: sha256(project / name)
        for name in ("dataset.py", "modules.py", "losses.py", "metrics.py", "review_pipeline.py", "label_schema.json")
    }
    report["source_hash_basis"] = "Verified working-tree file bytes before Git line-ending normalization."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
