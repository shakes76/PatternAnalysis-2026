"""
visualise_data.py

Looks at the actual ADNI images before trusting any model trained on them.

I wrote this for three reasons:

  1. I should see the data I am working with. Numbers in an audit table are not
     the same as looking at the pictures.
  2. The README needs a figure of the data, and the assignment asks for one.
  3. Figure 2 is the visual proof of the leakage argument. My tutor warned me
     about data leakage at the check-off, and showing 20 slices from one patient
     side by side makes the point better than a paragraph does: they are nearly
     the same image, so splitting by image instead of by patient would put
     near-duplicates on both sides of the split.

Run it with:
    python visualise_data.py --data-root /home/groups/comp3710/ADNI/AD_NC
    python visualise_data.py --data-root adni_subset/AD_NC --output-dir figures

It only reads images and writes PNGs, so it is light enough for a login node.
"""

import argparse
import os
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from dataset import scan_split_folder, LABEL_TO_CLASS, CLASS_TO_LABEL


def load_image_as_array(image_path):
    """Open one image and return it as a 2D numpy array of raw 0-255 values."""
    with Image.open(image_path) as raw_image:
        return np.array(raw_image.convert("L"))


def group_samples_by_patient(samples):
    """
    Turn the flat sample list into {patient_id: {"label": int, "paths": [...]}}.

    Sorting the paths matters for figure 2 - I want the slices in their natural
    order so the progression through the brain reads correctly.
    """
    by_patient = {}

    for image_path, label, patient_id in samples:
        if patient_id not in by_patient:
            by_patient[patient_id] = {"label": label, "paths": []}
        by_patient[patient_id]["paths"].append(image_path)

    for patient_id in by_patient:
        by_patient[patient_id]["paths"].sort()

    return by_patient


def figure_class_comparison(by_patient, output_dir, patients_per_class=6, seed=0):
    """
    Figure 1: a middle slice from several AD patients next to several NC patients.

    One slice per patient, and a different patient in every panel, so I am
    looking at between-patient variation rather than the same brain twice.

    I take the middle slice of each patient's stack because the first and last
    slices are near the edge of the brain and show very little tissue.
    """
    chooser = random.Random(seed)

    patients_by_label = {0: [], 1: []}
    for patient_id, info in by_patient.items():
        patients_by_label[info["label"]].append(patient_id)

    for label in patients_by_label:
        patients_by_label[label].sort()
        chooser.shuffle(patients_by_label[label])

    figure, axes = plt.subplots(2, patients_per_class, figsize=(2.0 * patients_per_class, 4.6))

    for row, label in enumerate((1, 0)):  # AD on the top row, NC underneath
        class_name = LABEL_TO_CLASS[label]
        chosen_patients = patients_by_label[label][:patients_per_class]

        for column, patient_id in enumerate(chosen_patients):
            paths = by_patient[patient_id]["paths"]
            middle_slice_path = paths[len(paths) // 2]

            image = load_image_as_array(middle_slice_path)

            axis = axes[row, column]
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
            axis.set_title("%s\npatient %s" % (class_name, patient_id), fontsize=8)
            axis.axis("off")

    figure.suptitle("ADNI: middle slice from different patients (top AD, bottom NC)",
                    fontsize=11)
    figure.tight_layout()

    output_path = os.path.join(output_dir, "fig1_class_comparison.png")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("wrote %s" % output_path)


def figure_one_patient_all_slices(by_patient, output_dir, seed=0):
    """
    Figure 2: every slice belonging to a single patient.

    This is the leakage figure. All 20 images come from one 3D scan of one
    person, so neighbouring slices are almost identical. If I split the dataset
    by image, roughly three of these would land in validation while the other
    seventeen stayed in training - and the model could score well on validation
    by recognising this particular brain, not by recognising the disease.

    Splitting by patient keeps all twenty on the same side of the split.
    """
    chooser = random.Random(seed)

    patient_ids = sorted(by_patient.keys())
    chosen_patient = chooser.choice(patient_ids)

    info = by_patient[chosen_patient]
    paths = info["paths"]
    class_name = LABEL_TO_CLASS[info["label"]]

    columns = 10
    rows = (len(paths) + columns - 1) // columns

    figure, axes = plt.subplots(rows, columns, figsize=(1.5 * columns, 1.7 * rows))
    axes = np.atleast_2d(axes)

    for index in range(rows * columns):
        row, column = divmod(index, columns)
        axis = axes[row, column]

        if index < len(paths):
            image = load_image_as_array(paths[index])
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
            # Show the slice number from the file name, not the loop index.
            slice_number = os.path.basename(paths[index]).split("_")[1].split(".")[0]
            axis.set_title("slice %s" % slice_number, fontsize=7)

        axis.axis("off")

    figure.suptitle(
        "All %d slices from one patient (%s, class %s)\n"
        "Neighbouring slices are near duplicates - this is why the split must be "
        "by patient, not by image"
        % (len(paths), chosen_patient, class_name),
        fontsize=11,
    )
    figure.tight_layout()

    output_path = os.path.join(output_dir, "fig2_one_patient_all_slices.png")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("wrote %s" % output_path)


def figure_class_averages(by_patient, output_dir, max_patients_per_class=60):
    """
    Figure 3: the average AD brain, the average NC brain, and the difference.

    Averaging many patients blurs away individual anatomy and leaves whatever is
    common to the class. The difference map then shows where the two classes
    disagree on average.

    What I expect from the medical literature: Alzheimer's involves enlarged
    ventricles (the fluid spaces in the middle of the brain) and thinner cortex
    at the outer edge. So the difference map should light up centrally and around
    the rim. If it instead lights up in one corner or along one edge, that would
    suggest the classes differ by some scanning artifact rather than by anatomy,
    which would be a serious finding worth chasing before trusting any model.

    I average the middle slice only. Averaging across different slice positions
    would mix anatomy from different depths and smear the result into mush.
    """
    sums_by_label = {0: None, 1: None}
    counts_by_label = {0: 0, 1: 0}

    for patient_id in sorted(by_patient.keys()):
        info = by_patient[patient_id]
        label = info["label"]

        if counts_by_label[label] >= max_patients_per_class:
            continue

        paths = info["paths"]
        image = load_image_as_array(paths[len(paths) // 2]).astype(np.float64)

        if sums_by_label[label] is None:
            sums_by_label[label] = image
        else:
            sums_by_label[label] += image

        counts_by_label[label] += 1

    if counts_by_label[0] == 0 or counts_by_label[1] == 0:
        print("skipping figure 3, not enough patients in one class")
        return

    mean_nc = sums_by_label[0] / counts_by_label[0]
    mean_ad = sums_by_label[1] / counts_by_label[1]
    difference = mean_ad - mean_nc

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(mean_ad, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Mean AD (%d patients)" % counts_by_label[1])
    axes[0].axis("off")

    axes[1].imshow(mean_nc, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Mean NC (%d patients)" % counts_by_label[0])
    axes[1].axis("off")

    # Symmetric colour limits around zero so red and blue mean equal magnitudes
    # in opposite directions.
    limit = float(np.abs(difference).max())
    difference_plot = axes[2].imshow(difference, cmap="bwr", vmin=-limit, vmax=limit)
    axes[2].set_title("AD minus NC\n(red: brighter in AD, blue: brighter in NC)")
    axes[2].axis("off")
    figure.colorbar(difference_plot, ax=axes[2], fraction=0.046)

    figure.suptitle("Average middle slice per class, and where they differ", fontsize=12)
    figure.tight_layout()

    output_path = os.path.join(output_dir, "fig3_class_averages.png")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("wrote %s" % output_path)


def figure_intensity_histograms(by_patient, output_dir, max_patients_per_class=40):
    """
    Figure 4: the distribution of pixel values for each class.

    Two things I am checking here.

    First, whether the big spike at zero is as large as I assumed. The measured
    mean of 0.1159 told me these images are mostly black background, and this
    shows exactly how much.

    Second, whether the two classes have noticeably different overall brightness.
    If they do, a model could reach decent accuracy by learning "AD images are
    slightly darker" rather than learning anything about anatomy. That would be
    a dataset artifact masquerading as a result, and the error analysis would
    need to account for it.
    """
    values_by_label = {0: [], 1: []}
    counts_by_label = {0: 0, 1: 0}

    for patient_id in sorted(by_patient.keys()):
        info = by_patient[patient_id]
        label = info["label"]

        if counts_by_label[label] >= max_patients_per_class:
            continue

        paths = info["paths"]
        image = load_image_as_array(paths[len(paths) // 2])
        values_by_label[label].append(image.ravel())
        counts_by_label[label] += 1

    figure, (all_axis, nonzero_axis) = plt.subplots(1, 2, figsize=(11, 4))

    for label, colour in ((1, "tab:red"), (0, "tab:blue")):
        if not values_by_label[label]:
            continue

        pixels = np.concatenate(values_by_label[label])
        class_name = LABEL_TO_CLASS[label]

        all_axis.hist(pixels, bins=64, range=(0, 255), alpha=0.5,
                      label="%s (n=%d)" % (class_name, counts_by_label[label]),
                      color=colour, density=True)

        # Background dominates the full histogram, so the second panel drops it
        # and shows only the actual tissue.
        tissue_pixels = pixels[pixels > 10]
        nonzero_axis.hist(tissue_pixels, bins=64, range=(10, 255), alpha=0.5,
                          label="%s" % class_name, color=colour, density=True)

    all_axis.set_title("All pixels (background included)")
    all_axis.set_xlabel("pixel value")
    all_axis.set_ylabel("density")
    all_axis.legend()
    all_axis.grid(alpha=0.3)

    nonzero_axis.set_title("Tissue only (pixel value > 10)")
    nonzero_axis.set_xlabel("pixel value")
    nonzero_axis.legend()
    nonzero_axis.grid(alpha=0.3)

    figure.suptitle("Pixel intensity distribution by class", fontsize=12)
    figure.tight_layout()

    output_path = os.path.join(output_dir, "fig4_intensity_histograms.png")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("wrote %s" % output_path)


def report_slice_similarity(by_patient, seed=0):
    """
    Put a number on how similar two slices from the same patient are, compared
    to two slices from different patients.

    This backs up figure 2 with a measurement rather than an impression. I use
    plain correlation between the flattened images, which is crude but easy to
    explain and good enough to make the point.

    If within-patient correlation is much higher than between-patient
    correlation, then image-level splitting really does leak, and I can say so
    with a number in the report instead of just asserting it.
    """
    chooser = random.Random(seed)
    patient_ids = sorted(by_patient.keys())

    def correlation(a, b):
        a = a.ravel().astype(np.float64)
        b = b.ravel().astype(np.float64)
        a = a - a.mean()
        b = b - b.mean()
        denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return float((a * b).sum() / denominator) if denominator > 0 else 0.0

    within_patient = []
    between_patient = []

    for _ in range(30):
        patient_id = chooser.choice(patient_ids)
        paths = by_patient[patient_id]["paths"]
        if len(paths) < 2:
            continue

        # Two different slices, same patient. Adjacent ones, since that is the
        # worst case for leakage.
        index = chooser.randrange(len(paths) - 1)
        image_a = load_image_as_array(paths[index])
        image_b = load_image_as_array(paths[index + 1])
        within_patient.append(correlation(image_a, image_b))

        # Same slice position, two different patients of the same class.
        label = by_patient[patient_id]["label"]
        same_class_others = [
            p for p in patient_ids
            if p != patient_id and by_patient[p]["label"] == label
        ]
        if not same_class_others:
            continue

        other_patient = chooser.choice(same_class_others)
        other_paths = by_patient[other_patient]["paths"]
        matched_index = min(index, len(other_paths) - 1)
        image_c = load_image_as_array(other_paths[matched_index])
        between_patient.append(correlation(image_a, image_c))

    print()
    print("SLICE SIMILARITY (plain correlation between flattened images)")
    if within_patient:
        print("  same patient, adjacent slices:  mean %.4f  (min %.4f, max %.4f, n=%d)"
              % (float(np.mean(within_patient)), float(np.min(within_patient)),
                 float(np.max(within_patient)), len(within_patient)))
    if between_patient:
        print("  different patients, same class: mean %.4f  (min %.4f, max %.4f, n=%d)"
              % (float(np.mean(between_patient)), float(np.min(between_patient)),
                 float(np.max(between_patient)), len(between_patient)))
    if within_patient and between_patient:
        print("  gap: %.4f" % (float(np.mean(within_patient)) - float(np.mean(between_patient))))
        print()
        print("  A large gap means two slices from one patient are much more alike")
        print("  than two slices from different patients. That is exactly the")
        print("  shortcut an image-level split would hand the model.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Look at the ADNI images.")
    parser.add_argument("--data-root", default="/home/groups/comp3710/ADNI/AD_NC",
                        help="Path to the AD_NC folder.")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="Which shipped split to draw the pictures from.")
    parser.add_argument("--output-dir", default="figures",
                        help="Where to write the PNG files.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for which patients get picked.")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    os.makedirs(arguments.output_dir, exist_ok=True)

    print("Reading %s split from %s" % (arguments.split, arguments.data_root))
    samples = scan_split_folder(arguments.data_root, arguments.split)
    by_patient = group_samples_by_patient(samples)

    print("Found %d images from %d patients" % (len(samples), len(by_patient)))

    label_counts = {0: 0, 1: 0}
    for info in by_patient.values():
        label_counts[info["label"]] += 1
    print("  AD patients: %d" % label_counts[1])
    print("  NC patients: %d" % label_counts[0])
    print()

    figure_class_comparison(by_patient, arguments.output_dir, seed=arguments.seed)
    figure_one_patient_all_slices(by_patient, arguments.output_dir, seed=arguments.seed)
    figure_class_averages(by_patient, arguments.output_dir)
    figure_intensity_histograms(by_patient, arguments.output_dir)

    report_slice_similarity(by_patient, seed=arguments.seed)


if __name__ == "__main__":
    main()
