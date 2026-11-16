"""
audit_adni.py

A small script to look at the ADNI dataset on Rangpur and report what is
actually in there. I am writing this before the data loader because the loader
has to split the data by patient, and I cannot do that until I know how patient
IDs appear in the file names.

What this prints:
  1. The folder tree, so I can see how classes and splits are organised.
  2. How many images sit in each folder.
  3. A handful of example file names from each folder.
  4. The size, mode and pixel range of one sample image.
  5. A guess at the patient ID pattern, and how many unique patients there are.

This is light enough to run on the login node - it only walks the filesystem
and opens a couple of images.

Run it with:
    python audit_adni.py
or point it somewhere else with:
    python audit_adni.py /some/other/path
"""

import os
import re
import sys
from collections import Counter, defaultdict

# Default location of the dataset on Rangpur, from the assignment sheet.
DEFAULT_ADNI_PATH = "/home/groups/comp3710/ADNI"

# File extensions we treat as images.
IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".nii", ".nii.gz", ".bmp", ".tif", ".tiff")


def is_image_file(file_name):
    """Return True if the file name ends with one of the image extensions."""
    return file_name.lower().endswith(IMAGE_EXTENSIONS)


def show_folder_tree(root_path, max_depth=3):
    """
    Print the folder structure under root_path, stopping at max_depth levels.

    I limit the depth because the leaf folders can hold thousands of images and
    printing all of them is useless. I only want the shape of the tree here.
    """
    print("=" * 70)
    print("1. FOLDER STRUCTURE (depth limited to %d)" % max_depth)
    print("=" * 70)

    for current_folder, subfolders, files in os.walk(root_path):
        # Work out how deep we are by counting path separators below the root.
        relative_path = os.path.relpath(current_folder, root_path)
        depth = 0 if relative_path == "." else relative_path.count(os.sep) + 1

        if depth > max_depth:
            # Stop descending any further down this branch.
            subfolders[:] = []
            continue

        indent = "    " * depth
        folder_name = os.path.basename(current_folder) or root_path
        image_count = sum(1 for f in files if is_image_file(f))

        if image_count > 0:
            print("%s%s/   <- %d image files" % (indent, folder_name, image_count))
        else:
            print("%s%s/" % (indent, folder_name))


def find_leaf_image_folders(root_path):
    """
    Walk the tree and return a dict of {folder_path: [image file names]}
    for every folder that directly contains image files.
    """
    folders_with_images = {}

    for current_folder, subfolders, files in os.walk(root_path):
        image_files = [f for f in files if is_image_file(f)]
        if image_files:
            folders_with_images[current_folder] = image_files

    return folders_with_images


def show_counts_and_samples(root_path, folders_with_images, samples_to_show=5):
    """Print how many images are in each folder plus a few example file names."""
    print()
    print("=" * 70)
    print("2. IMAGE COUNTS AND EXAMPLE FILE NAMES")
    print("=" * 70)

    total_images = 0

    for folder_path in sorted(folders_with_images.keys()):
        image_files = folders_with_images[folder_path]
        total_images += len(image_files)

        relative_path = os.path.relpath(folder_path, root_path)
        print()
        print("%s  ->  %d images" % (relative_path, len(image_files)))

        # Sort first so the examples are stable between runs.
        for file_name in sorted(image_files)[:samples_to_show]:
            print("      %s" % file_name)

    print()
    print("TOTAL IMAGES FOUND: %d" % total_images)


def inspect_one_image(folders_with_images):
    """
    Open a single image and report its size, colour mode and pixel range.

    I need the image size to set up resizing in the data loader, and the mode
    tells me whether these are greyscale or RGB, which decides the number of
    input channels for the model.
    """
    print()
    print("=" * 70)
    print("3. SAMPLE IMAGE PROPERTIES")
    print("=" * 70)

    if not folders_with_images:
        print("No image folders found, skipping.")
        return

    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed here, so I cannot open an image.")
        print("Try: pip install --user pillow")
        return

    # Pick the first folder and the first file in it.
    first_folder = sorted(folders_with_images.keys())[0]
    first_file = sorted(folders_with_images[first_folder])[0]
    full_path = os.path.join(first_folder, first_file)

    print("Opening: %s" % full_path)

    try:
        with Image.open(full_path) as image:
            print("  size (width, height): %s" % (image.size,))
            print("  mode: %s   (L means greyscale, RGB means colour)" % image.mode)

            # Look at the actual pixel values so I know the range before normalising.
            pixel_values = list(image.getdata())
            if isinstance(pixel_values[0], tuple):
                flat_values = [channel for pixel in pixel_values for channel in pixel]
            else:
                flat_values = pixel_values

            print("  pixel min: %s" % min(flat_values))
            print("  pixel max: %s" % max(flat_values))
    except Exception as error:
        print("  Could not open the image: %s" % error)


def guess_patient_id_pattern(folders_with_images):
    """
    Try to work out how patient IDs are written into the file names.

    This matters more than anything else in this script. ADNI images are 2D
    slices taken from 3D scans, so the same patient appears in many files. If
    slices from one patient end up in both the training and the test set, the
    model can memorise that patient and the test score becomes meaningless.
    That is the data leakage the assignment sheet specifically warns about.

    A common naming style for this dataset is something like:
        1003730_100.jpeg
    where the part before the underscore is the patient and the part after is
    the slice number. I check for that pattern but also print the raw leading
    digits so I can confirm it by eye rather than trusting the guess.
    """
    print()
    print("=" * 70)
    print("4. PATIENT ID PATTERN (the important one)")
    print("=" * 70)

    # Pattern: digits, underscore, digits, then the extension.
    underscore_pattern = re.compile(r"^(\d+)_(\d+)\.")

    for folder_path in sorted(folders_with_images.keys()):
        image_files = folders_with_images[folder_path]

        matched_ids = []
        unmatched_examples = []

        for file_name in image_files:
            match = underscore_pattern.match(file_name)
            if match:
                matched_ids.append(match.group(1))
            elif len(unmatched_examples) < 3:
                unmatched_examples.append(file_name)

        print()
        print("Folder: %s" % folder_path)
        print("  files matching the <patient>_<slice> pattern: %d of %d"
              % (len(matched_ids), len(image_files)))

        if matched_ids:
            unique_patients = sorted(set(matched_ids))
            print("  unique patient IDs: %d" % len(unique_patients))
            print("  first few patient IDs: %s" % unique_patients[:5])

            # How many slices does each patient contribute? If this number is
            # large, leakage would be severe, so it is worth seeing.
            slices_per_patient = Counter(matched_ids)
            counts = list(slices_per_patient.values())
            print("  slices per patient: min %d, max %d, average %.1f"
                  % (min(counts), max(counts), sum(counts) / len(counts)))

        if unmatched_examples:
            print("  files that did NOT match the pattern: %s" % unmatched_examples)


def check_patient_overlap_between_folders(folders_with_images):
    """
    If the dataset already ships with train and test folders, check whether any
    patient appears in both. If the provided split already leaks, I have to
    rebuild the splits myself from the combined pool.
    """
    print()
    print("=" * 70)
    print("5. DO ANY PATIENTS APPEAR IN MORE THAN ONE FOLDER?")
    print("=" * 70)

    underscore_pattern = re.compile(r"^(\d+)_(\d+)\.")
    patients_by_folder = defaultdict(set)

    for folder_path, image_files in folders_with_images.items():
        for file_name in image_files:
            match = underscore_pattern.match(file_name)
            if match:
                patients_by_folder[folder_path].add(match.group(1))

    folder_list = sorted(patients_by_folder.keys())

    if len(folder_list) < 2:
        print("Fewer than two folders with identifiable patient IDs, nothing to compare.")
        return

    found_any_overlap = False

    for i in range(len(folder_list)):
        for j in range(i + 1, len(folder_list)):
            folder_a = folder_list[i]
            folder_b = folder_list[j]
            shared = patients_by_folder[folder_a] & patients_by_folder[folder_b]

            if shared:
                found_any_overlap = True
                print()
                print("OVERLAP between:")
                print("  %s" % folder_a)
                print("  %s" % folder_b)
                print("  shared patients: %d, e.g. %s"
                      % (len(shared), sorted(shared)[:5]))

    if not found_any_overlap:
        print("No patient appears in two different folders. The shipped split is clean.")


def main():
    # Allow a different path to be passed in, otherwise use the Rangpur default.
    root_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ADNI_PATH

    print("Auditing ADNI dataset at: %s" % root_path)
    print()

    if not os.path.isdir(root_path):
        print("That path does not exist or is not a folder.")
        print("Check the path and try again.")
        return

    show_folder_tree(root_path)

    folders_with_images = find_leaf_image_folders(root_path)

    show_counts_and_samples(root_path, folders_with_images)
    inspect_one_image(folders_with_images)
    guess_patient_id_pattern(folders_with_images)
    check_patient_overlap_between_folders(folders_with_images)

    print()
    print("=" * 70)
    print("Audit finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()
