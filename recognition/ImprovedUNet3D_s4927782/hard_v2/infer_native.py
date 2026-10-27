"""Image-only adapter for the two frozen v1 seed42 HipMRI checkpoints.

No mask, cache manifest, patient selection, cropping, postprocessing or model
selection is used. Native-grid restoration cannot recover detail lost while
resizing to the 64 x 128 x 128 training grid. The original models are reused
verbatim, checked by digest, and are not new independent model implementations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np

DEFAULT_PROJECT = (Path(__file__).resolve().parents[1] / "PatternAnalysis-2026"
                   / "recognition" / "ImprovedUNet3D_s4927782")
MODULES_SHA256 = "e752b242154ba01165e34b2323aab499aeb3dbd5346f62c8e1f301840f010fc9"
FROZEN_CHECKPOINTS = {
    "cfdcfa08147c295962e7d19f2ad16eb20a08590f9a28e00e95a281417aec3c10": "unet2d",
    "9ea46e13a95617ec4ad80c21ba459087fb62941fb04406c7ffeade22ad014b62": "improved3d",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_affine(affine):
    """Accept finite nonsingular orthogonal axes, including oblique rotations.

    Sheared grids are rejected: v1's spacing-only representation cannot faithfully
    describe them. Oblique orthogonal grids retain the full chosen NIfTI affine.
    """
    affine = np.asarray(affine, dtype=np.float64)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-8)):
        raise ValueError("Expected a finite homogeneous 4x4 image affine.")
    linear = affine[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    if np.any(spacing <= 0) or abs(np.linalg.det(linear)) < 1e-10:
        raise ValueError("Image affine has singular or undefined spatial axes.")
    directions = linear / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), rtol=1e-5, atol=1e-5):
        raise ValueError("Sheared/nonorthogonal image grids are unsupported by the frozen v1 adapter.")
    return spacing


def grid_transform(canonical_affine, canonical_shape_xyz, target_shape_dhw):
    """Map cache XYZ voxel centres into the canonical source XYZ grid.

    SciPy zoom(grid_mode=True) uses (new_index + .5) * old/new - .5.
    """
    old = np.asarray(canonical_shape_xyz, dtype=np.float64)
    new = np.asarray(tuple(reversed(target_shape_dhw)), dtype=np.float64)
    if (old.shape != (3,) or new.shape != (3,) or not np.isfinite(old).all()
            or not np.isfinite(new).all() or np.any(old <= 0) or np.any(new <= 0)):
        raise ValueError("Source and target must have three positive dimensions.")
    transform = np.eye(4)
    scale = old / new
    transform[:3, :3] = np.diag(scale)
    transform[:3, 3] = .5 * (scale - 1)
    return np.asarray(canonical_affine, dtype=np.float64) @ transform


def preprocess_image(image_path, target_shape_dhw=(64, 128, 128)):
    """Independently reproduce v1 image operations; never accepts a label path."""
    import nibabel as nib
    from scipy.ndimage import zoom

    if (len(target_shape_dhw) != 3
            or any(type(size) is not int or size <= 0 for size in target_shape_dhw)):
        raise ValueError("Target dimensions must be three positive integers.")
    original = nib.load(str(image_path))
    if len(original.shape) != 3 or min(original.shape) < 1:
        raise ValueError("Expected one nonempty 3D image NIfTI.")
    spacing_from_affine = validate_affine(original.affine)
    orientation = nib.aff2axcodes(original.affine)
    if any(axis is None for axis in orientation):
        raise ValueError("Image affine has unknown spatial orientation.")
    original_spacing = np.asarray(original.header.get_zooms()[:3], dtype=np.float64)
    if (not np.isfinite(original_spacing).all() or np.any(original_spacing <= 0)
            or not np.allclose(original_spacing, spacing_from_affine, rtol=1e-5, atol=1e-5)):
        raise ValueError("Image header spacing disagrees with its chosen affine.")
    canonical = nib.as_closest_canonical(original)
    validate_affine(canonical.affine)
    if nib.aff2axcodes(canonical.affine) != ("R", "A", "S"):
        raise ValueError("Cannot establish a finite RAS canonical image grid.")
    image = canonical.get_fdata(dtype=np.float32, caching="unchanged")
    if not np.isfinite(image).all():
        raise ValueError("Image contains nonfinite intensities.")
    support = image != 0
    count = int(np.count_nonzero(support))
    mean = float(np.mean(image[support], dtype=np.float64)) if count else 0.0
    std = float(np.std(image[support], dtype=np.float64)) if count else 0.0
    normalized = np.zeros(image.shape, dtype=np.float32)
    if count:
        normalized[support] = np.clip(
            (image[support].astype(np.float64) - mean) / max(std, 1e-8), -5., 5.
        ).astype(np.float32)
    target_xyz = tuple(reversed(target_shape_dhw))
    ratio = np.asarray(target_xyz, dtype=np.float64) / np.asarray(image.shape, dtype=np.float64)
    options = dict(grid_mode=True, mode="nearest", prefilter=False)
    resized = zoom(normalized, ratio, order=1, **options)
    resized_support = zoom(support.astype(np.uint8), ratio, order=0, **options).astype(bool)
    resized[~resized_support] = 0.
    if resized.shape != target_xyz:
        raise ValueError("Image resampling did not produce the checkpoint grid.")
    cache_affine = grid_transform(canonical.affine, image.shape, target_shape_dhw)
    metadata = {
        "original_shape_xyz": list(original.shape),
        "original_affine_xyz": np.asarray(original.affine).tolist(),
        "original_orientation": list(orientation),
        "original_header_zooms_xyz": original_spacing.tolist(),
        "original_spatial_unit": original.header.get_xyzt_units()[0],
        "canonical_shape_xyz": list(canonical.shape),
        "canonical_affine_xyz": np.asarray(canonical.affine).tolist(),
        "cache_shape_dhw": list(target_shape_dhw),
        "cache_affine_xyz": cache_affine.tolist(),
        "cache_spacing_dhw": (np.asarray(canonical.header.get_zooms()[:3]) / ratio)[::-1].tolist(),
        "normalization": {"method": "per_case_nonzero_zscore_then_clip", "nonzero_count": count,
                          "mean": mean, "standard_deviation": std, "minimum_scale": 1e-8,
                          "clip": [-5., 5.]},
        "resampling": {"image_order": 1, "image_support_order": 0, **options},
        "affine_policy": "Chosen NIfTI affine; finite orthogonal/oblique axes accepted; shear and undefined axes rejected.",
    }
    return np.ascontiguousarray(resized.transpose(2, 1, 0), dtype=np.float32), original, metadata


def restore_prediction(prediction_dhw, original, cache_affine_xyz, label_ids):
    """Nearest inverse-affine sampling into native geometry; output raw labels."""
    from scipy.ndimage import affine_transform

    prediction = np.asarray(prediction_dhw)
    if (prediction.ndim != 3 or not np.issubdtype(prediction.dtype, np.integer)
            or prediction.size == 0 or prediction.min() < 0 or prediction.max() >= len(label_ids)):
        raise ValueError("Expected a nonempty cache-grid integer class-index prediction.")
    cache_affine = np.asarray(cache_affine_xyz, dtype=np.float64)
    validate_affine(cache_affine)
    validate_affine(original.affine)
    native_to_cache = np.linalg.solve(cache_affine, np.asarray(original.affine, dtype=np.float64))
    native_indices = affine_transform(
        np.ascontiguousarray(prediction.transpose(2, 1, 0)),
        matrix=native_to_cache[:3, :3], offset=native_to_cache[:3, 3],
        output_shape=original.shape, order=0, mode="nearest", prefilter=False,
    )
    raw_labels = np.asarray(label_ids, dtype=np.int64)[native_indices]
    if raw_labels.min() >= 0 and raw_labels.max() <= 255:
        raw_labels = raw_labels.astype(np.uint8)
    else:
        raw_labels = raw_labels.astype(np.int32)
    return raw_labels, native_to_cache


def load_model(project_root, checkpoint_path, device):
    """Reuse only known original models/checkpoints; do not import data helpers."""
    import torch

    checkpoint_sha = sha256(checkpoint_path)
    if checkpoint_sha not in FROZEN_CHECKPOINTS:
        raise ValueError("Checkpoint is not one of the two frozen v1 seed42 best checkpoints.")
    module_path = Path(project_root).resolve() / "modules.py"
    if sha256(module_path) != MODULES_SHA256:
        raise ValueError("Original modules.py digest differs from frozen v1.")
    specification = importlib.util.spec_from_file_location("frozen_hipmri_v1_modules", module_path)
    modules = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(modules)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if (config["model"] != FROZEN_CHECKPOINTS[checkpoint_sha] or config["seed"] != 42
            or config["num_classes"] != 6 or config["label_ids"] != list(range(6))
            or config["target_shape_dhw"] != [64, 128, 128] or config["base_channels"] != 16
            or config.get("development_only") or config.get("max_batches") is not None):
        raise ValueError("Frozen checkpoint config is inconsistent with v1.")
    factory = modules.UNet2D if config["model"] == "unet2d" else modules.ImprovedUNet3D
    model = factory(config["num_classes"], config["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, config, checkpoint["epoch"], checkpoint_sha, module_path


def infer_array(model, image_dhw, model_name, device, slice_batch_size=8, amp=False):
    import torch

    image = torch.from_numpy(np.ascontiguousarray(image_dhw)).unsqueeze(0)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        if model_name == "improved3d":
            logits = model(image.unsqueeze(0).to(device))["logits"]
            if not bool(torch.isfinite(logits).all().item()):
                raise ValueError("Nonfinite primary model logits.")
            prediction = logits.argmax(dim=1)[0].cpu()
        else:
            parts = []
            for chunk in image[0].unsqueeze(1).split(slice_batch_size):
                logits = model(chunk.to(device))["logits"]
                if not bool(torch.isfinite(logits).all().item()):
                    raise ValueError("Nonfinite primary model logits.")
                parts.append(logits.argmax(dim=1).cpu())
            prediction = torch.cat(parts, dim=0)
    return prediction.numpy().astype(np.uint8)


def save_outputs(output_dir, prediction, original, metadata, label_ids, receipt):
    """Write only to a new directory; never replace earlier evidence."""
    import nibabel as nib

    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output directory already exists; use a fresh directory.")
    native, inverse = restore_prediction(prediction, original, metadata["cache_affine_xyz"], label_ids)
    output.mkdir(parents=True, exist_ok=False)
    cache_path = output / "prediction_cache.npz"
    np.savez_compressed(cache_path, prediction=prediction,
                        affine_xyz=np.asarray(metadata["cache_affine_xyz"]),
                        spacing_dhw=np.asarray(metadata["cache_spacing_dhw"]),
                        label_ids=np.asarray(label_ids, dtype=np.int64))
    native_path = output / "prediction_native.nii.gz"
    header = original.header.copy()
    header.set_data_dtype(native.dtype)
    header.set_slope_inter(1., 0.)
    restored = nib.Nifti1Image(native, original.affine, header=header)
    # Preserve the source's chosen affine exactly; qform may be approximate for
    # a float-encoded header, while the sform defines the chosen output geometry.
    restored.set_sform(original.affine, code=int(original.header["sform_code"]) or 1)
    qform, qcode = original.get_qform(coded=True)
    if qform is not None:
        restored.set_qform(qform, int(qcode))
    nib.save(restored, native_path)
    # Check the serialized result, including scaling and geometry.
    reloaded = nib.load(native_path)
    if (reloaded.shape != original.shape or not np.allclose(reloaded.affine, original.affine, rtol=0, atol=1e-5)
            or not np.array_equal(np.asanyarray(reloaded.dataobj), native)):
        raise RuntimeError("Serialized native prediction differs from intended labels or geometry.")
    receipt.update({
        "schema_version": 1, "status": "completed", "image_only_inference": True,
        "mask_or_manifest_read": False, "model_changed": False,
        "geometry": metadata, "native_to_cache_voxel_transform_xyz": inverse.tolist(),
        "outputs": {"cache": {"path": cache_path.name, "sha256": sha256(cache_path), "values": "class indices"},
                    "native": {"path": native_path.name, "sha256": sha256(native_path), "values": "raw label IDs"}},
        "restoration": "Nearest inverse-affine sampling of cache-grid hard labels; mode=nearest at image edges.",
        "limitation": "Grid restoration does not recover information lost by downsampling and does not establish native-resolution accuracy.",
        "assistance_record": "AI-assisted independent image-only deployment adapter; original trained models reused by SHA-256; no new training or selection.",
    })
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch-size", type=int, default=8, help="2D slices per inference batch")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("Batch size must be positive.")
    output = args.output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error("Use a fresh output directory; existing evidence cannot be overwritten.")
    import torch
    import nibabel
    import scipy

    image_path = args.image.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable.")
    start = time.perf_counter()
    model, config, epoch, checkpoint_sha, module_path = load_model(args.project_root, checkpoint_path, device)
    amp = config["amp"] and device.type == "cuda" if args.amp is None else args.amp
    if amp and device.type != "cuda":
        parser.error("--amp requires CUDA; use --no-amp on CPU.")
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    image_sha = sha256(image_path)
    image, original, metadata = preprocess_image(image_path, tuple(config["target_shape_dhw"]))
    prediction = infer_array(model, image, config["model"], device, args.batch_size, amp)
    if sha256(image_path) != image_sha or sha256(checkpoint_path) != checkpoint_sha or sha256(module_path) != MODULES_SHA256:
        raise RuntimeError("A guarded image/model input changed during inference.")
    receipt = {
        "image": str(image_path), "image_sha256": image_sha,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": epoch, "model_config": config,
        "model_source": str(module_path), "model_source_sha256": MODULES_SHA256,
        "model_attribution": "Original ImprovedUNet3D_s4927782 modules.py; training commit 69400a70b9b7bbed1971408e74785e926236410a.",
        "adapter_source_sha256": sha256(__file__),
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "numpy_version": np.__version__, "nibabel_version": nibabel.__version__, "scipy_version": scipy.__version__,
        "device": str(device), "amp": bool(amp), "slice_batch_size": args.batch_size,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "elapsed_seconds_before_serialization": time.perf_counter() - start,
    }
    save_outputs(output, prediction, original, metadata, config["label_ids"], receipt)
    print(f"Image-only {config['model']} inference saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
