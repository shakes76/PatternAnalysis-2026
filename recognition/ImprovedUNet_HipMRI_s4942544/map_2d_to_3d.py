"""
map_2d_to_3d.py -- Link every 2D HipMRI slice to its source 3D volume.

Why this exists: the 2D slices (case_004_week_0_slice_12) and the 3D volumes
(B006_Week0_LFOV) use DIFFERENT patient identifiers. To compare 2D and 3D models
fairly, the 3D model must be trained, validated and tested on exactly the same
patients as the 2D models -- so we need to know which 3D volume each 2D scan
came from.

Method: every 2D ground-truth mask slice is compared, by exact content hash,
against every slice of every 3D label volume along all three axes and under all
eight flips/rotations. Label masks contain only integers, so an exact match is
unambiguous. Each 2D scan is assigned to the volume that most of its slices match.

Outputs (in --out):
    volumes_3d.json   shape / voxel spacing / orientation / labels of every 3D volume
    mapping_2d_3d.json  per-scan mapping: 3D volume, axis, transform, slice indices
    splits_3d.json    3D volume names for train / validate / test, inherited
                      from the patient-level 2D split

Usage (Rangpur CPU node, ~5-10 min):
    python map_2d_to_3d.py --out mapping_out
"""
import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np

ROOT = "/home/groups/comp3710/HipMRI_Study_open"
SPLITS_2D = {"train": "keras_slices_seg_train", "validate": "keras_slices_seg_validate",
             "test": "keras_slices_seg_test"}
NAME_2D = re.compile(r"seg_(\d+)_week_(\d+)_slice_(\d+)")
# Most volumes are named B006_Week0_SEMANTIC.nii.gz, but the prefix letter is not
# guaranteed, so any letters + digits are accepted; anything unparsed is reported.
NAME_3D = re.compile(r"([A-Za-z]+\d+)_Week(\d+)", re.IGNORECASE)

# The eight ways a 2D array can be flipped/rotated (the dihedral group).
TRANSFORMS = {
    "identity": lambda a: a, "rot90": lambda a: np.rot90(a, 1), "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3), "flipud": np.flipud, "fliplr": np.fliplr,
    "transpose": lambda a: a.T, "antitranspose": lambda a: np.rot90(a, 2).T,
}


def digest(arr):
    a = np.ascontiguousarray(arr.astype(np.uint8))
    return hashlib.md5(a.tobytes() + str(a.shape).encode()).hexdigest()


def load(path):
    a = nib.load(path).get_fdata(caching="unchanged")
    while a.ndim > 3:
        a = a[..., 0]
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT)
    p.add_argument("--out", default="mapping_out")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------ 2D side
    t0 = time.time()
    jobs = []
    for split, d in SPLITS_2D.items():
        folder = os.path.join(args.root, "keras_slices_data", d)
        for f in os.listdir(folder):
            m = NAME_2D.search(f)
            if m:
                jobs.append((split, (m.group(1), m.group(2), int(m.group(3))), os.path.join(folder, f)))

    def read_2d(job):
        split, key, path = job
        a = load(path)
        if a.ndim == 3:
            a = a[:, :, 0]
        return key, (split, digest(a), bool((a == 5).any()), a.shape)

    with ThreadPoolExecutor(args.threads) as pool:  # file reading dominates; threads overlap the I/O
        slices_2d = dict(pool.map(read_2d, jobs))   # (patient, week, slice) -> (split, hash, has_prostate, shape)
    shapes_2d = Counter(v[3] for v in slices_2d.values())
    print(f"2D: {len(slices_2d)} mask slices, shapes {dict(shapes_2d)} ({time.time() - t0:.0f}s)", flush=True)
    wanted_shapes = set(shapes_2d)

    # ------------------------------------------------------------------ 3D side
    lab_dir = os.path.join(args.root, "semantic_labels_only")
    mr_dir = os.path.join(args.root, "semantic_MRs")
    mr_files = set(os.listdir(mr_dir))
    volumes = {}
    index = defaultdict(list)  # hash -> [(volume, axis, transform, slice index)]
    lab_files = sorted(f for f in os.listdir(lab_dir) if f.endswith((".nii", ".nii.gz")))
    unparsed_3d = [f for f in lab_files if not NAME_3D.search(f)]
    print(f"3D: {len(lab_files)} label files, {len(unparsed_3d)} with unparsed names {unparsed_3d[:5]}", flush=True)

    def read_3d(f):
        img = nib.load(os.path.join(lab_dir, f))
        # Nifti data arrive in Fortran order; slicing that layout is ~1000x slower, so make it C-ordered.
        return f, img, np.ascontiguousarray(np.asarray(img.dataobj), dtype=np.uint8)

    t0 = time.time()
    pool = ThreadPoolExecutor(args.threads)
    for n_done, (f, img, lab) in enumerate(pool.map(read_3d, [f for f in lab_files if NAME_3D.search(f)]), 1):
        while lab.ndim > 3:
            lab = lab[..., 0]
        m = NAME_3D.search(f)
        vol = f"{m.group(1)}_Week{m.group(2)}"
        # The matching MR volume: same "<id>_Week<n>_" stem, preferring the LFOV file.
        mr_candidates = sorted(g for g in mr_files if g.lower().startswith(vol.lower() + "_"))
        mr_name = next((g for g in mr_candidates if "LFOV" in g), mr_candidates[0] if mr_candidates else None)
        mr_shape = list(nib.load(os.path.join(mr_dir, mr_name)).shape) if mr_name else None
        counts = np.bincount(lab.ravel(), minlength=6)
        volumes[vol] = {
            "patient": m.group(1), "week": m.group(2), "label_file": f,
            "mr_file": mr_name, "mr_shape": mr_shape,
            "shape": list(lab.shape), "spacing_mm": [round(float(z), 4) for z in img.header.get_zooms()[:3]],
            "orientation": "".join(nib.aff2axcodes(img.affine)),
            "labels": {str(i): int(c) for i, c in enumerate(counts) if c},
        }
        for axis in range(3):
            for k in range(lab.shape[axis]):
                s = np.take(lab, k, axis=axis)
                for tname, t in TRANSFORMS.items():
                    ts = t(s)
                    if ts.shape in wanted_shapes:
                        index[digest(ts)].append((vol, axis, tname, k))
        if n_done % 20 == 0 or n_done == len(lab_files) - len(unparsed_3d):
            print(f"   indexed {n_done} volumes ({time.time() - t0:.0f}s)", flush=True)
    pool.shutdown()
    print(f"3D: {len(volumes)} label volumes indexed", flush=True)

    # ------------------------------------------------------------------ match
    by_scan = defaultdict(list)
    for key, (split, h, has_p, _) in slices_2d.items():
        by_scan[(key[0], key[1])].append((key[2], split, h, has_p))

    mapping, unmatched = {}, []
    for (pat, wk), items in sorted(by_scan.items()):
        votes = Counter()
        for s, split, h, has_p in items:
            hits = index.get(h, [])
            # Only distinctive (prostate-containing) slices, matching few places, get a vote.
            if has_p and 0 < len(hits) <= 8:
                for vol, axis, tname, _ in hits:
                    votes[(vol, axis, tname)] += 1
        name = f"case_{pat}_week_{wk}"
        if not votes:
            unmatched.append(name)
            continue
        (vol, axis, tname), n = votes.most_common(1)[0]
        # Slices are stacked in order, so 3D index = sign * 2D index + offset. Every
        # candidate (2D slice, 3D slice) pair votes for its (sign, offset); the winner
        # then places every slice, including ones whose content repeats in the volume.
        candidates = {}
        line_votes = Counter()
        for s, split, h, has_p in items:
            ks = [k for v, a, t, k in index.get(h, []) if (v, a, t) == (vol, axis, tname)]
            candidates[s] = set(ks)
            for k in ks:
                line_votes[(1, k - s)] += 1
                line_votes[(-1, k + s)] += 1
        (sign, offset), _ = line_votes.most_common(1)[0]
        slice_map = {s: sign * s + offset for s, ks in candidates.items() if sign * s + offset in ks}
        mapping[name] = {
            "split": items[0][1], "patient_2d": pat, "week_2d": wk, "volume": vol,
            "patient_3d": volumes[vol]["patient"], "axis": axis, "transform": tname,
            "slice_direction": sign, "slice_offset": offset,
            "votes": n, "prostate_slices": sum(1 for it in items if it[3]),
            "slices_matched": len(slice_map), "slices_total": len(items),
            "slice_to_3d_index": {str(k): v for k, v in sorted(slice_map.items())},
        }

    # ------------------------------------------------------------------ checks
    patient_map = defaultdict(set)
    for m in mapping.values():
        patient_map[m["patient_2d"]].add(m["patient_3d"])
    one_to_one = all(len(v) == 1 for v in patient_map.values())
    used = Counter(m["volume"] for m in mapping.values())
    duplicates = [v for v, c in used.items() if c > 1]

    splits_3d = defaultdict(list)
    for m in mapping.values():
        splits_3d[m["split"]].append(m["volume"])
    patients_by_split = {s: sorted({volumes[v]["patient"] for v in vols}) for s, vols in splits_3d.items()}
    overlap = {f"{a}&{b}": sorted(set(patients_by_split.get(a, [])) & set(patients_by_split.get(b, [])))
               for a, b in [("train", "validate"), ("train", "test"), ("validate", "test")]}
    unassigned = sorted(set(volumes) - set(used))

    with open(os.path.join(args.out, "volumes_3d.json"), "w") as f:
        json.dump(volumes, f, indent=1)
    with open(os.path.join(args.out, "mapping_2d_3d.json"), "w") as f:
        json.dump(mapping, f, indent=1)
    with open(os.path.join(args.out, "splits_3d.json"), "w") as f:
        json.dump({"splits": {s: sorted(v) for s, v in splits_3d.items()},
                   "patients": patients_by_split,
                   "patient_map_2d_to_3d": {k: sorted(v) for k, v in sorted(patient_map.items())},
                   "unassigned_volumes": unassigned}, f, indent=1)

    # ------------------------------------------------------------------ report
    shapes = Counter(tuple(v["shape"]) for v in volumes.values())
    spacings = Counter(tuple(v["spacing_mm"]) for v in volumes.values())
    orients = Counter(v["orientation"] for v in volumes.values())
    print("\n" + "=" * 66)
    print("3D VOLUMES")
    print(f"   label shapes:  {dict(shapes.most_common(6))}")
    print(f"   MR shapes:     {dict(Counter(tuple(v['mr_shape'] or []) for v in volumes.values()).most_common(6))}")
    print(f"   spacing (mm):  {dict(spacings.most_common(6))}")
    print(f"   orientation:   {dict(orients)}")
    print(f"   label values:  {sorted({int(k) for v in volumes.values() for k in v['labels']})}")
    print(f"   volumes missing an MR file: {[k for k, v in volumes.items() if v['mr_file'] is None]}")
    print("\n2D -> 3D MAPPING")
    print(f"   scans mapped: {len(mapping)} / {len(by_scan)}   unmatched: {unmatched[:10]}")
    print(f"   axis/transform used: {dict(Counter((m['axis'], m['transform']) for m in mapping.values()))}")
    frac = [m["slices_matched"] / m["slices_total"] for m in mapping.values()]
    print(f"   slices matched per scan: min {min(frac):.0%}, mean {np.mean(frac):.0%}" if frac else "")
    print(f"   each 2D patient -> exactly one 3D patient: {one_to_one}")
    print(f"   3D volumes claimed by more than one 2D scan: {duplicates[:10]}")
    print(f"   3D volumes not linked to any 2D scan: {len(unassigned)} {unassigned[:10]}")
    print("\n3D SPLIT (inherited from the 2D patient-level split)")
    for s in ["train", "validate", "test"]:
        print(f"   {s:8s}: {len(splits_3d.get(s, []))} volumes, {len(patients_by_split.get(s, []))} patients")
    print(f"   patient overlap between splits: {overlap}")
    print(f"   test patients (3D ids): {patients_by_split.get('test')}")
    example = next(iter(mapping.values()), None)
    if example:
        ks = list(example["slice_to_3d_index"].items())
        print(f"\n   example: case_{example['patient_2d']}_week_{example['week_2d']} -> {example['volume']}, "
              f"axis {example['axis']}, {example['transform']}, 2D->3D slices {ks[:3]} ... {ks[-2:]}")
    print(f"\nSaved to {args.out}/: volumes_3d.json, mapping_2d_3d.json, splits_3d.json")


if __name__ == "__main__":
    main()
