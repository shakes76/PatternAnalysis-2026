"""HipMRI audits, patient splits and aligned 3D training data.

Run this saved file on a Rangpur compute node, for example:
    python dataset.py --case B006_Week0

Patient IDs below are inferred from filenames and still require confirmation.
Preprocessing follows the task sheet Appendix B: per-volume z-score and
categorical masks. Interpolation is implemented with PyTorch, using the
same voxel-centre grid for images and labels; see the README for trade-offs.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DEFAULT_DATA_ROOT = Path("/home/groups/comp3710/HipMRI_Study_open")
# CSIRO collection v2, BriefDataDescription.txt (retrieved 15 October 2026).
LABEL_SOURCE = "https://data.csiro.au/dap/ws/v2/collections/52549/support/5034"
LABEL_NAMES = {
    0: "Background", 1: "Body", 2: "Bones", 3: "Bladder",
    4: "Rectum", 5: "Prostate",
}
PILOT_SHAPE_DHW = (64, 128, 128)


def read_patient_manifest(manifest_path):
    """Validate the saved split, including all groups, before selecting one.

    This reads filenames/identities only. It never opens held-out images to
    compute training statistics, and never generates a replacement split.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    groups = manifest["splits"]
    if set(groups) != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain train, validation and test.")
    patients_seen, cases_seen, identities = set(), set(), []
    for name, group in groups.items():
        patients, scans = group["patient_ids"], group["scans"]
        if not patients or not scans or len(patients) != len(set(patients)):
            raise ValueError(f"Empty or duplicate patient group in {name}.")
        if patients_seen.intersection(patients):
            raise ValueError("Patient leakage between manifest splits.")
        patients_seen.update(patients)
        if {scan["patient_id"] for scan in scans} != set(patients):
            raise ValueError(f"Patient list and scan membership differ in {name}.")
        for scan in scans:
            case, patient = scan["case_id"], scan["patient_id"]
            match = re.fullmatch(r"([A-Za-z]+\d+)_Week(\d+)", case)
            if (match is None or match.group(1) != patient
                    or int(match.group(2)) != scan["week"]):
                raise ValueError(f"Inconsistent case/patient/week: {case}.")
            if case in cases_seen:
                raise ValueError(f"Duplicate case in manifest: {case}.")
            cases_seen.add(case)
            for role, suffix in (("image", "LFOV"), ("mask", "SEMANTIC")):
                if Path(scan[f"{role}_path"]).name not in (
                    f"{case}_{suffix}.nii", f"{case}_{suffix}.nii.gz"
                ):
                    raise ValueError(f"Incorrect {role} filename for {case}.")
            identities.append((case, patient))
    digest = hashlib.sha256(json.dumps(sorted(identities)).encode("utf-8")).hexdigest()
    if digest != manifest["inventory_identity_sha256"]:
        raise ValueError("Manifest identity hash does not match its contents.")
    if manifest["patient_overlap"] is not False:
        raise ValueError("Manifest records patient overlap.")
    return manifest


def load_preprocessed_pair(pair, target_shape=PILOT_SHAPE_DHW):
    """Load one complete case, reorient to RAS and resize without cropping.

    NIfTI arrays are X,Y,Z; PyTorch spatial tensors are Z,Y,X (D,H,W).
    Each MRI uses its own mean/std, as in Appendix B, without using a mask
    or pooled validation/test statistics. Constant images normalize to zero.
    Trilinear images (align_corners=False) and nearest-exact masks share
    the same voxel-centre sampling grid. Fixed shape is not fixed mm spacing.
    """
    target_shape = tuple(target_shape)
    if (len(target_shape) != 3
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in target_shape)):
        raise ValueError("target_shape must contain three positive integers (D,H,W).")
    image_original = nib.load(pair["image_path"])
    mask_original = nib.load(pair["mask_path"])
    if (len(image_original.shape) != 3 or image_original.shape != mask_original.shape
            or not np.allclose(image_original.affine, mask_original.affine, rtol=0, atol=1e-4)):
        raise ValueError(f"Image/mask grids differ: {pair['case_id']}.")
    for volume in (image_original, mask_original):
        spacing = np.asarray(volume.header.get_zooms())
        if not np.all(np.isfinite(volume.affine)) or not np.all(np.isfinite(spacing) & (spacing > 0)):
            raise ValueError("Invalid spatial metadata.")
    image_nii = nib.as_closest_canonical(image_original)
    mask_nii = nib.as_closest_canonical(mask_original)
    if not np.allclose(image_nii.affine, mask_nii.affine, rtol=0, atol=1e-4):
        raise ValueError("Canonical image/mask grids differ.")
    pixels = image_nii.get_fdata(dtype=np.float32)
    labels = np.asanyarray(mask_nii.dataobj)
    if not np.all(np.isfinite(pixels)):
        raise ValueError("MRI contains nonfinite intensities.")
    if not np.all(np.isin(labels, list(LABEL_NAMES))):
        raise ValueError("Mask must contain integer codes 0 through 5.")
    image = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 1, 0))).float()
    mask = torch.from_numpy(np.ascontiguousarray(labels.transpose(2, 1, 0))).long()
    mean, std = image.mean(), image.std(correction=0)
    image = (image - mean) / std.clamp_min(1e-8)
    image = F.interpolate(image[None, None], size=target_shape,
                          mode="trilinear", align_corners=False)[0]
    mask = F.interpolate(mask[None, None].float(), size=target_shape,
                         mode="nearest-exact")[0, 0].long()
    target = F.one_hot(mask, num_classes=len(LABEL_NAMES)).permute(3, 0, 1, 2).float()

    # New XYZ voxel centre j maps to old (j + .5) * scale - .5.
    scale_xyz = np.asarray(image_nii.shape) / np.asarray(target_shape[::-1])
    mapping = np.eye(4)
    mapping[:3, :3] = np.diag(scale_xyz)
    mapping[:3, 3] = (scale_xyz - 1) / 2
    output_affine = image_nii.affine @ mapping
    return {
        "case_id": pair["case_id"], "patient_id": pair["patient_id"],
        "image": image.contiguous(), "mask": mask.contiguous(),
        "target": target.contiguous(),
        "affine_xyz": torch.tensor(output_affine, dtype=torch.float64),
        "original_affine_xyz": torch.tensor(image_original.affine, dtype=torch.float64),
        "original_shape_xyz": torch.tensor(image_original.shape),
        "normalization_mean": mean, "normalization_std": std,
    }


class HipMRIVolumeDataset(Dataset):
    """One complete 3D image/target per item, shared by both model routes.

    data_root relocates the manifest's exact basenames, without changing its
    patient assignments. Only the requested split's image files are opened.
    A training-only paired left/right flip is optional and off for the pilot.
    No slice from a held-out patient is used for fitting or augmentation.
    """

    def __init__(self, manifest_path, split="train", data_root=DEFAULT_DATA_ROOT,
                 target_shape=PILOT_SHAPE_DHW, flip_probability=0.0):
        manifest = read_patient_manifest(manifest_path)
        if split not in manifest["splits"]:
            raise ValueError(f"Unknown split: {split}.")
        if not 0 <= flip_probability <= 1 or (split != "train" and flip_probability != 0):
            raise ValueError("Paired augmentation is allowed only in training (probability 0..1).")
        self.pairs = []
        for scan in manifest["splits"][split]["scans"]:
            pair = dict(scan)
            for role, folder in (("image", "semantic_MRs"), ("mask", "semantic_labels_only")):
                pair[f"{role}_path"] = str(Path(data_root) / folder / Path(scan[f"{role}_path"]).name)
            self.pairs.append(pair)
        self.target_shape = tuple(target_shape)
        self.flip_probability = flip_probability

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        sample = load_preprocessed_pair(self.pairs[index], self.target_shape)
        if self.flip_probability and torch.rand(()).item() < self.flip_probability:
            for key in ("image", "mask", "target"):
                sample[key] = sample[key].flip(-1)
            # W is the X axis of the canonical RAS volume.
            reflection = torch.eye(4, dtype=torch.float64)
            reflection[0, 0] = -1
            reflection[0, 3] = self.target_shape[2] - 1
            sample["affine_xyz"] = sample["affine_xyz"] @ reflection
        return sample


def volume_to_slices(image, target):
    """Unbatched 1xDxHxW / KxDxHxW -> Dx1xHxW / DxKxHxW.

    This is a view of the same preprocessed case, not a new random slice split.
    Reassembly is the inverse permutation; keep slice order during evaluation.
    """
    if (image.ndim != 4 or target.ndim != 4 or image.shape[0] != 1
            or image.shape[1:] != target.shape[1:]):
        raise ValueError("Expected aligned unbatched 3D image and target tensors.")
    return image.permute(1, 0, 2, 3).contiguous(), target.permute(1, 0, 2, 3).contiguous()


def build_inventory(data_root):
    """Match image/mask filenames by case, without loading scan contents.

    B006_Week0 and B006_Week1 are grouped under candidate patient ID B006.
    Duplicate or unexpected names are reported rather than silently selected.
    This checks file pairing, not spatial metadata or labels of every volume.
    """
    root = Path(data_root)
    indexed = {}
    issues = []
    counts = {}
    for role, folder, suffix in (
        ("image", "semantic_MRs", "_LFOV"),
        ("mask", "semantic_labels_only", "_SEMANTIC"),
    ):
        directory = root / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"Required data directory missing: {directory}")
        indexed[role] = {}
        counts[role] = 0
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.name.endswith(".nii.gz"):
                stem = path.name[:-7]
            elif path.name.endswith(".nii"):
                stem = path.name[:-4]
            else:
                continue
            counts[role] += 1
            if not stem.endswith(suffix):
                issues.append(f"Unexpected {role} filename: {path.name}")
                continue
            case_id = stem[:-len(suffix)]
            indexed[role].setdefault(case_id, []).append(path)

    pairs = []
    cases = sorted(set(indexed["image"]) | set(indexed["mask"]))
    for case_id in cases:
        images = indexed["image"].get(case_id, [])
        masks = indexed["mask"].get(case_id, [])
        if len(images) != 1 or len(masks) != 1:
            issues.append(f"{case_id}: {len(images)} images, {len(masks)} masks; expected one each")
            continue
        match = re.fullmatch(r"([A-Za-z]+\d+)_Week(\d+)", case_id)
        if match is None:
            issues.append(f"Unrecognised patient/week naming: {case_id}")
            continue
        pairs.append({
            "case_id": case_id,
            "patient_id": match.group(1),
            "week": int(match.group(2)),
            "image_path": str(images[0].resolve()),
            "mask_path": str(masks[0].resolve()),
        })
    patient_counts = {}
    for pair in pairs:
        patient_id = pair["patient_id"]
        patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1
    if not pairs:
        issues.append("No valid image/mask pairs found")
    return {
        "data_root": str(root.resolve()),
        "image_files": counts["image"],
        "mask_files": counts["mask"],
        "paired_scans": len(pairs),
        "candidate_patients": len(patient_counts),
        "scans_per_candidate_patient": dict(sorted(patient_counts.items())),
        "patient_id_rule": "Filename prefix before _Week; identity mapping requires confirmation",
        "pairs": pairs,
        "issues": issues,
    }


def volume_metadata(volume):
    """Read shape and spatial metadata without loading image intensities."""
    return {
        "shape": tuple(volume.shape),
        "voxel_spacing": tuple(float(v) if np.isfinite(v) else None for v in volume.header.get_zooms()),
        "orientation": nib.aff2axcodes(volume.affine),
        "stored_dtype": str(volume.get_data_dtype()),
    }


def make_patient_split(inventory, seed=42, val_patients=6, test_patients=6):
    """Make one reproducible split using filename-derived patient groups.

    Counts and seed are project choices, not prescribed course settings.
    Test data must not be used to choose a more favourable seed or split.
    """
    if inventory["issues"]:
        raise ValueError("Resolve inventory issues before splitting.")
    patients = sorted({pair["patient_id"] for pair in inventory["pairs"]})
    if val_patients < 1 or test_patients < 1:
        raise ValueError("Validation and test each require at least one patient.")
    if val_patients + test_patients >= len(patients):
        raise ValueError("Patient counts must leave a nonempty training set.")
    random.Random(seed).shuffle(patients)
    groups = {
        "test": sorted(patients[:test_patients]),
        "validation": sorted(patients[test_patients:test_patients + val_patients]),
        "train": sorted(patients[test_patients + val_patients:]),
    }
    splits = {}
    for name, patient_ids in groups.items():
        selected = set(patient_ids)
        splits[name] = {
            "patient_ids": patient_ids,
            "scans": sorted(
                (pair for pair in inventory["pairs"] if pair["patient_id"] in selected),
                key=lambda pair: pair["case_id"],
            ),
        }
    overlap = any(
        set(groups[a]) & set(groups[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    if overlap:
        raise ValueError("Patient overlap detected.")
    # Bind this split to the case/group list, independent of absolute paths.
    identities = sorted((p["case_id"], p["patient_id"]) for p in inventory["pairs"])
    digest = hashlib.sha256(json.dumps(identities).encode("utf-8")).hexdigest()
    return {
        "seed": seed,
        "patient_id_rule": inventory["patient_id_rule"],
        "inventory_identity_sha256": digest,
        "patient_overlap": overlap,
        "label_source": LABEL_SOURCE,
        "label_names": LABEL_NAMES,
        "splits": splits,
    }


def audit_pair(data_root, case_id):
    """Inspect one image/mask pair without changing either source file.

    Matching shape/affine is a preliminary grid check. An overlay and the
    publisher label description establish the remaining interpretation context.
    """
    data_root = Path(data_root)
    image_path = data_root / "semantic_MRs" / f"{case_id}_LFOV.nii.gz"
    mask_path = data_root / "semantic_labels_only" / f"{case_id}_SEMANTIC.nii.gz"
    return inspect_pair_files(image_path, mask_path, case_id)


def inspect_pair_files(image_path, mask_path, case_id, check_intensities=False):
    """Inspect exact inventory paths; optionally read all MRI intensities.

    This is a data-integrity check, not fitting a normalization or selecting
    model settings. Original arrays, class codes and source files are unchanged.
    """
    image = nib.load(image_path)
    mask = nib.load(mask_path)

    label_values, label_counts = np.unique(np.asanyarray(mask.dataobj), return_counts=True)
    valid_labels = bool(
        label_values.size > 0
        and np.all(np.isfinite(label_values))
        and np.all(label_values == np.round(label_values))
        and np.all(label_values >= 0)
    )
    checks = {
        "both_3d": len(image.shape) == 3 and len(mask.shape) == 3,
        "same_shape": image.shape == mask.shape,
        "same_affine": bool(
            np.allclose(image.affine, mask.affine, rtol=0, atol=1e-4)
        ),
        "finite_nonnegative_integer_labels": valid_labels,
        "known_label_codes": bool(np.all(np.isin(label_values, list(LABEL_NAMES)))),
    }
    both_3d = checks["both_3d"]
    image_spacing = np.asarray(image.header.get_zooms(), dtype=float)
    mask_spacing = np.asarray(mask.header.get_zooms(), dtype=float)
    checks["valid_voxel_spacing"] = bool(
        both_3d and np.all(np.isfinite(image_spacing))
        and np.all(np.isfinite(mask_spacing))
        and np.all(image_spacing > 0) and np.all(mask_spacing > 0)
    )
    checks["same_voxel_spacing"] = bool(
        both_3d and np.allclose(image_spacing, mask_spacing, rtol=0, atol=1e-4)
    )
    if check_intensities:
        checks["finite_image_intensities"] = bool(
            np.all(np.isfinite(np.asanyarray(image.dataobj)))
        )
    return {
        "case_id": case_id,
        "image": volume_metadata(image),
        "mask": volume_metadata(mask),
        "number_of_mask_values": int(label_values.size),
        "mask_values_preview": [v.item() if np.isfinite(v) else str(v) for v in label_values[:20]],
        "mask_value_counts": {str(v): int(n) for v, n in zip(label_values, label_counts)},
        "checks": checks,
    }


def audit_all_pairs(inventory, output_path):
    """Read each listed pair and checkpoint a report every ten cases.

    No scans are dropped, transformed or used to tune a model. A complete
    report with failures is retained so that problems can be investigated.
    """
    if inventory["issues"]:
        raise ValueError("Resolve inventory issues before the full audit.")
    pairs = inventory["pairs"]
    identities = sorted((p["case_id"], p["patient_id"]) for p in pairs)
    report = {
        "expected_pairs": len(pairs), "checked_pairs": 0,
        "passed_pairs": 0, "failed_pairs": 0, "complete": False,
        "inventory_identity_sha256": hashlib.sha256(
            json.dumps(identities).encode("utf-8")
        ).hexdigest(),
        "label_source": LABEL_SOURCE,
        "label_names": LABEL_NAMES,
        "cases": [],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save_report():
        # Replace only after a complete write, preserving the last checkpoint.
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output_path)

    save_report()
    print(f"Reading {len(pairs)} MRI/mask pairs; source files stay unchanged.", flush=True)
    for number, pair in enumerate(pairs, start=1):
        try:
            observation = inspect_pair_files(
                pair["image_path"], pair["mask_path"], pair["case_id"],
                check_intensities=True,
            )
            passed = all(observation["checks"].values())
            observation["passed"] = passed
        except (OSError, ValueError, EOFError, np.linalg.LinAlgError,
                nib.filebasedimages.ImageFileError) as error:
            passed = False
            observation = {"case_id": pair["case_id"], "passed": False,
                           "error": f"{type(error).__name__}: {error}"}
        report["cases"].append(observation)
        report["checked_pairs"] = number
        report["passed_pairs" if passed else "failed_pairs"] += 1
        if not passed:
            print(f"FAIL: {pair['case_id']}", flush=True)
        if number % 10 == 0 or not passed or number == len(pairs):
            save_report()
            print(f"Checked {number}/{len(pairs)}; failures={report['failed_pairs']}", flush=True)
    report["complete"] = True
    save_report()
    return report


def check_training_loader(manifest_path, data_root, target_shape, samples, output_path):
    """Exercise training cases only; save checks, not training results.

    The first case plus the longest training case are included when possible,
    so the depth-144 scan is not silently missed. Only header shapes are read
    for choosing the longest case; no validation/test scan is loaded.
    """
    if samples < 1:
        raise ValueError("samples must be positive.")
    dataset = HipMRIVolumeDataset(manifest_path, "train", data_root, target_shape)
    longest = max(range(len(dataset)), key=lambda i: nib.load(dataset.pairs[i]["image_path"]).shape[2])
    selected = list(dict.fromkeys([0, longest] + list(range(len(dataset)))))[:samples]
    report = {"split": "train", "shape_dhw": list(target_shape), "cases": [],
              "complete": False, "is_training_result": False}
    for index in selected:
        sample = dataset[index]
        image, mask, target = sample["image"], sample["mask"], sample["target"]
        slices, slice_targets = volume_to_slices(image, target)
        checks = {
            "expected_image_shape": tuple(image.shape) == (1, *target_shape),
            "expected_target_shape": tuple(target.shape) == (6, *target_shape),
            "finite_image": bool(torch.isfinite(image).all()),
            "one_hot": bool(torch.all(target.sum(dim=0) == 1)),
            "encoding_matches_mask": bool(torch.equal(target.argmax(dim=0), mask)),
            "slice_image_roundtrip": bool(torch.equal(slices.permute(1, 0, 2, 3), image)),
            "slice_target_roundtrip": bool(torch.equal(slice_targets.permute(1, 0, 2, 3), target)),
            "all_six_labels_retained": set(mask.unique().tolist()) == set(LABEL_NAMES),
        }
        counts = torch.bincount(mask.flatten(), minlength=6).tolist()
        observation = {
            "case_id": sample["case_id"],
            "original_shape_xyz": sample["original_shape_xyz"].tolist(),
            "image_shape": list(image.shape), "target_shape": list(target.shape),
            "affine_xyz": sample["affine_xyz"].tolist(),
            "label_voxel_counts": counts,
            "normalization_mean": sample["normalization_mean"].item(),
            "normalization_std": sample["normalization_std"].item(),
            "checks": checks, "passed": all(checks.values()),
        }
        report["cases"].append(observation)
        print(f"{sample['case_id']}: image={tuple(image.shape)}, target={tuple(target.shape)}, "
              f"labels={mask.unique().tolist()}, passed={observation['passed']}", flush=True)
    report["complete"] = True
    report["passed"] = all(case["passed"] for case in report["cases"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Report saved:", output_path)
    print("LOADER_CHECK_PASSED" if report["passed"] else "LOADER_CHECK_FAILED")
    return report


def main():
    parser = argparse.ArgumentParser(description="Inventory or audit HipMRI data.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--case", default="B006_Week0")
    parser.add_argument("--overlay", type=Path, help="Save a numeric-label preview PNG.")
    parser.add_argument("--inventory", type=Path, help="Save a full filename-pair inventory JSON.")
    parser.add_argument("--split", type=Path, help="Create a new patient-split JSON; refuses overwrite.")
    parser.add_argument("--audit-all", type=Path, help="Read all pairs and save an integrity report JSON.")
    parser.add_argument("--loader-check", type=Path, help="Check prepared training tensors; save JSON.")
    parser.add_argument("--manifest", type=Path, default=Path("runs/split_seed42.json"))
    parser.add_argument("--shape", type=int, nargs=3, default=PILOT_SHAPE_DHW, metavar=("D", "H", "W"))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-patients", type=int, default=6)
    parser.add_argument("--test-patients", type=int, default=6)
    args = parser.parse_args()

    if args.loader_check:
        if args.split or args.inventory or args.overlay or args.audit_all:
            parser.error("Use --loader-check separately from other output modes.")
        torch.set_num_threads(1)
        report = check_training_loader(args.manifest, args.data_root, tuple(args.shape),
                                       args.samples, args.loader_check)
        return 0 if report["passed"] else 1

    if args.audit_all:
        if args.split or args.inventory or args.overlay:
            parser.error("Use --audit-all separately from other output modes.")
        report = audit_all_pairs(build_inventory(args.data_root), args.audit_all)
        print("Audit complete:", report["complete"])
        print("Checked pairs:", report["checked_pairs"])
        print("Passed pairs:", report["passed_pairs"])
        print("Failed pairs:", report["failed_pairs"])
        print("Report saved:", args.audit_all)
        return 1 if report["failed_pairs"] else 0

    if args.split:
        if args.inventory or args.overlay:
            parser.error("Use --split separately from --inventory and --overlay.")
        if args.split.exists():
            parser.error("Split file already exists; reuse it rather than overwriting.")
        inventory = build_inventory(args.data_root)
        split = make_patient_split(inventory, args.seed, args.val_patients, args.test_patients)
        args.split.parent.mkdir(parents=True, exist_ok=True)
        with args.split.open("x", encoding="utf-8") as stream:
            json.dump(split, stream, indent=2)
            stream.write("\n")
        for name, group in split["splits"].items():
            print(f"{name}: {len(group['patient_ids'])} patients, {len(group['scans'])} scans")
        print("patient_overlap:", split["patient_overlap"])
        print("Seed:", split["seed"])
        print("Split saved:", args.split)
        print("Both models must use this same split. Filename identity rule still needs confirmation.")
        return 0

    if args.inventory:
        if args.overlay:
            parser.error("Use --inventory separately from --overlay.")
        inventory = build_inventory(args.data_root)
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        for key in ("image_files", "mask_files", "paired_scans", "candidate_patients"):
            print(f"{key}: {inventory[key]}")
        print("issues:", len(inventory["issues"]))
        for issue in inventory["issues"][:10]:
            print(" -", issue)
        print("Inventory saved:", args.inventory)
        print("Patient IDs are inferred from filenames; no train/test split created.")
        return 1 if inventory["issues"] else 0

    report = audit_pair(args.data_root, args.case)
    print("Case:", report["case_id"])
    for role in ("image", "mask"):
        for field, value in report[role].items():
            print(f"{role} {field}: {value}")
    print("Number of mask values:", report["number_of_mask_values"])
    print("Mask values (first 20):", report["mask_values_preview"])
    print("Label names:", LABEL_NAMES)
    for name, passed in report["checks"].items():
        print(f"{name}: {passed}")
    if args.overlay and all(report["checks"].values()):
        save_overlay(args.data_root, args.case, args.overlay)
        print("Overlay saved:", args.overlay)
    print("Next: validate the remaining volumes and patient grouping before training.")
    # A failed check stops a caller from treating this pair as ready for use.
    return 0 if all(report["checks"].values()) else 1


def save_overlay(data_root, case_id, output_path):
    """Preview each nonzero code with the publisher's anatomical label name.

    Use the axial slice with the largest area for each code. Reorient both
    volumes to RAS together; display scaling never changes the source data.
    Background code 0 is omitted from the preview.
    """
    import matplotlib

    matplotlib.use("Agg")  # Save images without a desktop on Rangpur.
    import matplotlib.pyplot as plt

    root = Path(data_root)
    image = nib.as_closest_canonical(nib.load(root / "semantic_MRs" / f"{case_id}_LFOV.nii.gz"))
    mask = nib.as_closest_canonical(nib.load(root / "semantic_labels_only" / f"{case_id}_SEMANTIC.nii.gz"))
    pixels = image.get_fdata(dtype=np.float32)
    labels = np.asanyarray(mask.dataobj)
    codes = [int(code) for code in np.unique(labels) if code != 0]
    if not codes:
        raise ValueError("No nonzero mask codes to preview.")
    finite = pixels[np.isfinite(pixels)]
    if finite.size == 0:
        raise ValueError("MRI contains no finite intensities.")
    low, high = np.percentile(finite, [1, 99])
    fig, axes = plt.subplots(len(codes), 3, figsize=(10, 3 * len(codes)), squeeze=False)
    for row, code in enumerate(codes):
        region = labels == code
        z = int(np.argmax(region.sum(axis=(0, 1))))
        section = pixels[:, :, z].T
        binary = region[:, :, z].T
        for ax in axes[row]:
            ax.set_axis_off()
        axes[row, 0].imshow(section, cmap="gray", origin="lower", vmin=low, vmax=high)
        axes[row, 0].set_title(f"{case_id}: axial RAS z={z}")
        axes[row, 1].imshow(binary, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Label {code}: {LABEL_NAMES.get(code, 'Unknown')}")
        axes[row, 2].imshow(section, cmap="gray", origin="lower", vmin=low, vmax=high)
        axes[row, 2].imshow(np.ma.masked_where(~binary, binary), cmap="autumn", origin="lower", vmin=0, vmax=1, alpha=0.45, interpolation="nearest")
        axes[row, 2].set_title(f"MRI + {LABEL_NAMES.get(code, 'Unknown')}")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
