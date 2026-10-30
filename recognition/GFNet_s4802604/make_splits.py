"""Build a subject-level train/val/test split for the ADNI AD vs NC data.

The provided train/test folders are split by scan, not by subject, so the same
person can appear in both. This script pools everything, groups scans by ADNI
subject ID, and splits by subject (stratified by label, seed 42) so no subject
appears in more than one split. It also writes a small report on how badly the
provided split leaks.

Output: ~/adni_work/splits.json (not committed) and leak_report.md (committed).
"""
import json
import os
import random
import re
from collections import Counter, defaultdict

ROOT = "/home/groups/comp3710/ADNI"
OUT_DIR = os.path.expanduser("~/adni_work")
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # test gets the remaining 15%
SLICES_PER_SCAN = 20

meta = json.load(open(f"{ROOT}/meta_data_with_label.json"))


def subject_of(scan_id):
    """ADNI subject ID (e.g. 068_S_0473) from the raw filename in the metadata."""
    return re.search(r"ADNI_(\d+_S_\d+)", meta[scan_id]["raw"]).group(1)


# scan id -> info, and which scans are in the provided train/test folders
scans = {}
provided = {"train": set(), "test": set()}
for split in ("train", "test"):
    for cls in ("AD", "NC"):
        for fname in os.listdir(f"{ROOT}/AD_NC/{split}/{cls}"):
            scan_id = fname.rsplit("_", 1)[0]
            scans[scan_id] = {"subject": subject_of(scan_id), "label": meta[scan_id]["label"]}
            provided[split].add(scan_id)

# 1. How much does the provided split leak?
prov_train_subj = {scans[s]["subject"] for s in provided["train"]}
prov_test_subj = {scans[s]["subject"] for s in provided["test"]}
overlap = prov_train_subj & prov_test_subj

# 2. Group scans by subject and split by subject
by_subject = defaultdict(list)
for scan_id, info in scans.items():
    by_subject[info["subject"]].append(scan_id)


def subject_label(subj):
    return Counter(scans[s]["label"] for s in by_subject[subj]).most_common(1)[0][0]


random.seed(SEED)
split_subjects = {"train": [], "val": [], "test": []}
for label in sorted({subject_label(s) for s in by_subject}):
    subs = sorted(s for s in by_subject if subject_label(s) == label)
    random.shuffle(subs)
    n = len(subs)
    a, b = int(TRAIN_FRAC * n), int((TRAIN_FRAC + VAL_FRAC) * n)
    split_subjects["train"] += subs[:a]
    split_subjects["val"] += subs[a:b]
    split_subjects["test"] += subs[b:]

# Safety check: no subject in more than one split
names = list(split_subjects)
for i in range(3):
    for j in range(i + 1, 3):
        assert not set(split_subjects[names[i]]) & set(split_subjects[names[j]])

split_scans = {n: [s for sub in subs for s in by_subject[sub]] for n, subs in split_subjects.items()}

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(split_scans, open(f"{OUT_DIR}/splits.json", "w"))

# 3. Report
lines = ["# ADNI split audit", ""]
lines.append(f"Total: {len(by_subject)} subjects, {len(scans)} scans, "
             f"{len(scans) * SLICES_PER_SCAN} slices ({SLICES_PER_SCAN} slices per scan).")
lines.append("Label coding in metadata: 0 = NC, 2 = AD.")
lines.append("")
lines.append("## Provided split leaks")
lines.append(f"- Provided train subjects: {len(prov_train_subj)}")
lines.append(f"- Provided test subjects: {len(prov_test_subj)}")
lines.append(f"- Subjects in BOTH: {len(overlap)} "
             f"({100 * len(overlap) / len(prov_test_subj):.0f}% of test subjects)")
lines.append("")
lines.append("## Subject-level split used (seed 42, stratified by label)")
lines.append("| Split | Subjects | Scans | Slices | NC scans | AD scans |")
lines.append("|---|---|---|---|---|---|")
for n in names:
    c = Counter(scans[s]["label"] for s in split_scans[n])
    lines.append(f"| {n} | {len(split_subjects[n])} | {len(split_scans[n])} | "
                 f"{len(split_scans[n]) * SLICES_PER_SCAN} | {c[0]} | {c[2]} |")
report = "\n".join(lines)
open("leak_report.md", "w").write(report + "\n")
print(report)
