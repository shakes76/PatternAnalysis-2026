"""
Split ADNI_Unsplit into test train val by patient. 
Before using the script, need to have an empty directory in form:

ADNI_Processed_2/test/0_NC
ADNI_Processed_2/test/1_AD
ADNI_Processed_2/train/0_NC
ADNI_Processed_2/train/1_AD
ADNI_Processed_2/val/0_NC
ADNI_Processed_2/val/1_AD

Once that is made, simply run the script, ensuring the file paths below are corrrect
"""
import os
import json
import random
import shutil
from sklearn.model_selection import train_test_split

# Classes
CLASSES = ["0_NC", "1_AD"]

# File paths 
BASE = r"C:\Users\elvin\OneDrive\Desktop\UQ\Year 3, Semester 2\COMP3710\Final Project"
SOURCE_DIR = os.path.join(BASE, "ADNI_Unsplit")
OUTPUT_DIR = os.path.join(BASE, "ADNI_Processed_2")
METADATA_FILE = os.path.join(BASE, "meta_data_with_label.json")

# Small helper functions
def get_scan_and_slice(filename):
    """'1003730_100.jpeg' -> (1003730, 100)"""
    name, _ = os.path.splitext(filename)
    parts = name.split("_")
    scan_id = parts[0]
    slice_number = int(parts[1])
    return scan_id, slice_number


def get_patient_id(raw_path):
    pieces = raw_path.replace("\\", "/").replace("/", "_").split("_")
    for i in range(len(pieces) - 2):
        a, b, c = pieces[i], pieces[i + 1], pieces[i + 2]
        if len(a) == 3 and a.isdigit() and b == "S" and len(c) == 4 and c.isdigit():
            return a + "_S_" + c

# Read metadata
with open(METADATA_FILE) as f:
    metadata = json.load(f)


# Compile the slices into scans
scans = {}

for class_name in CLASSES:
    class_folder = os.path.join(SOURCE_DIR, class_name)
    for filename in os.listdir(class_folder):
        scan_id, slice_number = get_scan_and_slice(filename)

        if scan_id not in scans:
            scans[scan_id] = {"class": class_name, "files": [], "first_slice": slice_number}

        scans[scan_id]["files"].append(os.path.join(class_folder, filename))
        if slice_number < scans[scan_id]["first_slice"]:
            scans[scan_id]["first_slice"] = slice_number

print("Found", len(scans), "scans")

# Group the scans by patients
patients = {}

for scan_id in scans:
    patient_id = get_patient_id(metadata[scan_id]["raw"])
    scans[scan_id]["patient"] = patient_id

    if patient_id not in patients:
        patients[patient_id] = []
    patients[patient_id].append(scan_id)

print("Found", len(patients), "patients")

# Split patients into test train val
patient_ids = sorted(patients)
labels = [scans[patients[p][0]]["class"] for p in patient_ids]

train_ids, temp_ids, _, temp_labels = train_test_split(patient_ids, labels, test_size=0.30, stratify=labels, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, stratify=temp_labels, random_state=42)

patient_split = {}
for p in train_ids: patient_split[p] = "train"
for p in val_ids:   patient_split[p] = "val"
for p in test_ids:  patient_split[p] = "test"

# Copy the files
copied = 0
for scan_id in scans:
    scan = scans[scan_id]
    split = patient_split[scan["patient"]]
    if scan["class"] == "0_NC":
        destination = os.path.join(OUTPUT_DIR, split, "0_NC")
    else:
        destination = os.path.join(OUTPUT_DIR, split, "1_AD")
    for file_path in scan["files"]:
        shutil.copy2(file_path, destination)
        copied += 1
        if copied % 2000 == 0:
            print("Copied", copied, "files")
print("Copied", copied, "files in total")