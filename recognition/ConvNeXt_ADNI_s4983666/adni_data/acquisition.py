"""Acquisition year and sequence of each scan, parsed from the metadata ``raw`` path.

Only used for analysis (docs/ACQUISITION_CONFOUND.md): the models never see
these values. Example path:
``ADNI_<NNN_S_NNNN>_MR_Accelerated_Sagittal_MPRAGE__br_raw_<YYYYMMDDhhmmss...>_<series>_I<scan ID>.nii``
"""

import json
from pathlib import Path
import re

RAW_PATTERN = re.compile(r"_MR_(.*?)_(?:br_)?raw_(\d{4})\d{4}")
SHARED_PERIOD_START = 2017  # From 2017 on, both AD and NC scans were acquired.


def acquisition_info(data_root) -> dict[str, dict]:
    """Map image ID to {"sequence", "year"} for every metadata record with a parsable path."""
    metadata = json.loads((Path(data_root) / "meta_data_with_label.json").read_text(encoding="utf-8-sig"))
    info = {}
    for image_id, record in metadata.items():
        match = RAW_PATTERN.search(record["raw"])
        if match:
            info[image_id] = {"sequence": match.group(1).strip("_"), "year": int(match.group(2))}
    return info
