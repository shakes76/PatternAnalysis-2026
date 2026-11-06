"""    
 Checks:
  1. Images, scans and patients per class, each counted directly.
  2. Metadata label agrees with the folder (AD=2, NC=0 in the metadata).
  3. Shows example metadata 'raw' paths so you can see the patient ID by eye.
  4. Every scan's 20 slices are consecutive.
  5. Exact duplicate images (same pixels in two files).
  
 python data-analysis.py <ADNI_Unsplit folder> <meta_data_with_label.json>
"""

import collections
import hashlib
import json
import os
import re
import sys

from PIL import Image

root, meta_path = sys.argv[1], sys.argv[2]
with open(meta_path) as fh:
    meta = json.load(fh)
pid_re = re.compile(r"(\d{3}_S_\d+)")

scan_slices = collections.defaultdict(list)  # (cls, scan) -> slice indices
files = []
for cls in sorted(os.listdir(root)):
    d = os.path.join(root, cls)
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        m = re.match(r"^(\d+)_(\d+)\.jpe?g$", f, re.I)
        if not m:
            print("SKIPPED (name):", cls, f)
            continue
        scan_slices[(cls, m.group(1))].append(int(m.group(2)))
        files.append((cls, os.path.join(d, f)))

print("== 1. Direct counts per class ==")
for cls in sorted({c for c, _ in scan_slices}):
    scans = [s for c, s in scan_slices if c == cls]
    n_img = sum(len(scan_slices[(cls, s)]) for s in scans)
    pids = {pid_re.search(meta[s]["raw"]).group(1) for s in scans}
    print(f"{cls}: images={n_img} scans={len(scans)} patients={len(pids)}")
all_pids = {pid_re.search(meta[s]["raw"]).group(1) for _, s in scan_slices}
print("total patients:", len(all_pids), "total scans:", len(scan_slices),
      "total images:", len(files))

print("\n== 2. Metadata label vs folder ==")
agree = collections.Counter()
for cls, s in scan_slices:
    agree[(cls, meta[s].get("label"))] += 1
for k, v in sorted(agree.items(), key=str):
    print(f"folder={k[0]} metadata_label={k[1]}: {v} scans")

print("\n== 3. Example raw paths (patient ID should be the NNN_S_NNNN part) ==")
for cls in sorted({c for c, _ in scan_slices}):
    for s in [s for c, s in scan_slices if c == cls][:2]:
        print(cls, s, "->", meta[s]["raw"])
multi = collections.Counter(pid_re.findall(meta[s]["raw"]).__len__()
                            for _, s in scan_slices)
print("number of NNN_S_NNNN matches per raw path:", dict(multi))

print("\n== 4. Consecutive slices ==")
bad = [k for k, v in scan_slices.items()
       if sorted(v) != list(range(min(v), min(v) + len(v)))]
print("scans with non-consecutive slices:", len(bad), bad[:5])

print("\n== 5. Exact duplicate images (hashing all files, takes a minute) ==")
seen = {}
dups = []
for cls, p in files:
    with Image.open(p) as im:
        h = hashlib.sha256(im.tobytes()).hexdigest()
    if h in seen:
        dups.append((seen[h], (cls, os.path.basename(p))))
    else:
        seen[h] = (cls, os.path.basename(p))
print("duplicate pairs:", len(dups))
cross = [d for d in dups if d[0][0] != d[1][0]]
print("duplicates across AD/NC:", len(cross))
for d in dups[:5]:
    print("  ", d)
