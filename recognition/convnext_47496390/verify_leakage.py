#https://share.gemini.google/IBsqVznhVwgE
import os
from pathlib import Path
import re

def extract_patient_id(filename):
    """Extracts patient ID from filename format like '392586_92.jpeg'."""
    match = re.match(r"^(\d+)_", filename)
    return match.group(1) if match else None

def get_patient_ids_from_dir(directory_path):
    """Recursively collects all unique patient IDs from a directory."""
    patient_ids = set()
    path = Path(directory_path)
    
    if not path.exists():
        print(f"Warning: Directory {directory_path} does not exist.")
        return patient_ids

    for file_path in path.rglob("*.*"):
        if file_path.suffix.lower() in [".jpeg", ".jpg", ".png"]:
            pid = extract_patient_id(file_path.name)
            if pid:
                patient_ids.add(pid)
            else:
                print(f"Could not parse patient ID from filename: {file_path.name}")
                
    return patient_ids

def verify_data_leakage(base_dir="ADNI/AD_NC"):
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    print(f"Scanning training set in: {train_dir}")
    train_patients = get_patient_ids_from_dir(train_dir)
    
    print(f"Scanning testing set in: {test_dir}")
    test_patients = get_patient_ids_from_dir(test_dir)

    print(f"\nTotal unique training patients: {len(train_patients)}")
    print(f"Total unique testing patients: {len(test_patients)}")

    # Check for leakage between train and test
    leakage = train_patients.intersection(test_patients)

    if leakage:
        print(f"\n[ERROR] Data Leakage Detected! Found {len(leakage)} overlapping patients between train and test sets:")
        print(sorted(list(leakage)))
    else:
        print("\n[SUCCESS] No data leakage detected! Train and test patient sets are completely disjoint.")

if __name__ == "__main__":
    # Adjust base_dir if your script is run from a different location
    verify_data_leakage("/home/groups/comp3710/ADNI/AD_NC")
