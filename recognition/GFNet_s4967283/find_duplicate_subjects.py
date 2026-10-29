"""
find_duplicate_subjects.py

Test whether the same person appears under more than one patient ID.

The question this answers
-------------------------
Both models drop sharply from validation to test:

    BaselineCNN   validation 0.9028   test 0.7421   drop 16 points
    GFNet         validation 0.7907   test 0.6837   drop 11 points

audit_split_difference.py showed the shipped train and test folders are
statistically indistinguishable on slice numbering, head size, head position,
brightness and contrast. So the gap is not a folder difference.

That leaves one explanation that fits a gap this large. ADNI follows people for
years and scans them repeatedly. If one person's sessions were filed under
different patient IDs, then a validation patient can be the SAME PERSON as a
training patient, scanned on another day. My overlap assertion compares IDs, so
it would pass while that leak sat underneath it.

Teaching staff told me the IDs are patient IDs and not session IDs. This script
tests that directly rather than taking it on authority.

The design
----------
For every validation patient, find the most similar TRAINING patient by image
correlation. Then do the same for every test patient, as a control.

    If validation patients have much closer matches in the training set than
    test patients do, the same people are in both splits.

    If the two distributions look the same, the IDs are clean and the
    validation-to-test gap has another cause.

The control is what makes this a test rather than a fishing trip. A high
correlation on its own means nothing, because all brain MRIs look alike. What
matters is whether validation matches are closer than test matches.

Usage:
    python find_duplicate_subjects.py
    python find_duplicate_subjects.py --thumbnail-size 48 --max-train 400
"""

import argparse

import numpy as np
from PIL import Image

from dataset import build_splits


# Each patient is reduced to one small image. Small enough to compare 900 by
# 450 patients quickly, large enough to keep head shape and ventricle size.
DEFAULT_THUMBNAIL = 48


def patient_thumbnail(image_paths, thumbnail_size):
    """
    Reduce one patient's 20 slices to a single small image.

    I average the slices because the average keeps head shape, ventricle size
    and cortical outline, which is what identifies a person, while averaging
    away the per-slice noise that does not.
    """
    accumulator = None
    count = 0

    for image_path in sorted(image_paths):
        with Image.open(image_path) as raw_image:
            small = raw_image.convert("L").resize(
                (thumbnail_size, thumbnail_size), Image.BILINEAR)
        pixels = np.asarray(small, dtype=np.float32)

        if accumulator is None:
            accumulator = pixels
        else:
            accumulator += pixels
        count += 1

    if count == 0:
        return None

    mean_image = (accumulator / count).ravel()

    # Centre and scale, so the comparison measures pattern and not brightness.
    # Two scans of the same head at different exposures should still match.
    mean_image = mean_image - mean_image.mean()
    norm = np.linalg.norm(mean_image)
    if norm == 0:
        return None

    return mean_image / norm


def build_thumbnails(samples, thumbnail_size, keep_patients=None):
    """Return {patient_id: unit vector} for the patients asked for."""
    by_patient = {}
    for image_path, _label, patient_id in samples:
        if keep_patients is not None and patient_id not in keep_patients:
            continue
        by_patient.setdefault(patient_id, []).append(image_path)

    thumbnails = {}
    for patient_id, paths in by_patient.items():
        vector = patient_thumbnail(paths, thumbnail_size)
        if vector is not None:
            thumbnails[patient_id] = vector

    return thumbnails


def nearest_matches(query_thumbnails, reference_thumbnails):
    """
    For each query patient, the best correlation against any reference patient.

    Both sets are unit vectors with the mean removed, so a dot product IS the
    Pearson correlation. 1.0 means identical pattern.
    """
    reference_ids = sorted(reference_thumbnails.keys())
    reference_matrix = np.stack([reference_thumbnails[p] for p in reference_ids])

    results = []
    for query_id in sorted(query_thumbnails.keys()):
        correlations = reference_matrix @ query_thumbnails[query_id]
        best_index = int(np.argmax(correlations))
        results.append({
            "query": query_id,
            "match": reference_ids[best_index],
            "correlation": float(correlations[best_index]),
        })

    return results


def summarise(name, results):
    """Print the distribution of best-match correlations for one group."""
    values = np.array([r["correlation"] for r in results])
    print("  %-12s n=%-5d mean %.4f   median %.4f   p95 %.4f   max %.4f"
          % (name, len(values), values.mean(), np.median(values),
             np.percentile(values, 95), values.max()))
    return values


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Check whether one person appears under several patient IDs."
    )
    parser.add_argument("--data-root", default="/home/groups/comp3710/ADNI/AD_NC")
    parser.add_argument("--thumbnail-size", type=int, default=DEFAULT_THUMBNAIL)
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the training seed, so the validation "
                             "split is the same one the models used.")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--top", type=int, default=10,
                        help="How many closest pairs to print.")
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    print("Reducing each patient to one %d x %d image."
          % (arguments.thumbnail_size, arguments.thumbnail_size))
    print()

    # ---- The exact same split the models used ----
    # build_splits is the function train.py uses, with the same seed, so the
    # validation patients here are the validation patients the models saw.
    train_samples, validation_samples, test_samples = build_splits(
        adni_root=arguments.data_root,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
    )

    def patient_ids_of(samples):
        return sorted({patient_id for _, _, patient_id in samples})

    train_ids = patient_ids_of(train_samples)
    validation_ids = patient_ids_of(validation_samples)
    test_ids = patient_ids_of(test_samples)

    print("Split from the shipped train folder, seed %d:" % arguments.seed)
    print("  %d training patients, %d validation patients"
          % (len(train_ids), len(validation_ids)))
    print("  %d test patients from the shipped test folder" % len(test_ids))
    print()

    print("Building thumbnails. This reads every image once.")
    training = build_thumbnails(train_samples, arguments.thumbnail_size)
    validation = build_thumbnails(validation_samples, arguments.thumbnail_size)
    test = build_thumbnails(test_samples, arguments.thumbnail_size)
    print("  %d training, %d validation, %d test"
          % (len(training), len(validation), len(test)))

    # ---- The comparison, and its control ----
    print()
    print("Matching every validation and test patient to its closest training patient.")
    validation_matches = nearest_matches(validation, training)
    test_matches = nearest_matches(test, training)

    print()
    print("=" * 76)
    print("BEST CORRELATION TO ANY TRAINING PATIENT")
    print("=" * 76)
    validation_values = summarise("validation", validation_matches)
    test_values = summarise("test (control)", test_matches)

    difference = validation_values.mean() - test_values.mean()

    print()
    print("  difference in means: %+.4f" % difference)

    # ---- The closest pairs ----
    print()
    print("=" * 76)
    print("CLOSEST VALIDATION TO TRAINING PAIRS")
    print("=" * 76)
    ranked = sorted(validation_matches,
                    key=lambda r: r["correlation"], reverse=True)
    for record in ranked[:arguments.top]:
        print("  validation %-9s matches training %-9s   correlation %.4f"
              % (record["query"], record["match"], record["correlation"]))

    print()
    print("=" * 76)
    print("CLOSEST TEST TO TRAINING PAIRS, FOR COMPARISON")
    print("=" * 76)
    ranked_test = sorted(test_matches,
                         key=lambda r: r["correlation"], reverse=True)
    for record in ranked_test[:arguments.top]:
        print("  test       %-9s matches training %-9s   correlation %.4f"
              % (record["query"], record["match"], record["correlation"]))

    # ---- The verdict ----
    print()
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)

    if difference > 0.02:
        print("  Validation patients sit CLOSER to the training set than test")
        print("  patients do, by %+.4f on average." % difference)
        print()
        print("  That is consistent with the same people appearing under more")
        print("  than one patient ID. Validation accuracy would then be")
        print("  optimistic, and the validation-to-test gap would be partly an")
        print("  artifact of the ID scheme rather than a property of the models.")
        print()
        print("  Next step: open the closest pairs above side by side and look.")
        print("  Two scans of one person are obvious to the eye.")
    else:
        print("  Validation and test patients are about equally far from the")
        print("  training set. The difference in means is %+.4f." % difference)
        print()
        print("  So the patient IDs are clean, and repeated subjects do NOT")
        print("  explain the validation-to-test gap. Teaching staff were right.")
        print()
        print("  That closes the last cheap explanation. The remaining ones are")
        print("  that the test cohort is genuinely harder, or that the two")
        print("  cohorts differ in something no pixel measure captures, such as")
        print("  scanner model or acquisition site.")

    print()
    print("  NOTE. A high correlation on its own means nothing here, because")
    print("  all brain MRIs resemble each other. Only the DIFFERENCE between")
    print("  the validation and test distributions carries information.")


if __name__ == "__main__":
    main()
