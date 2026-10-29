"""
predict.py

Example usage of a trained model. This is the test driver the assignment asks
for. It loads saved weights, runs inference, prints results, and produces the
figures the report needs.

The assignment says predict.py should:
    "load your saved model weights, run inference on test cases, print out any
     results, and provide visualisations where applicable"

What it does:

  1. Loads a checkpoint written by train.py.
  2. Runs inference on the split you choose.
  3. Prints accuracy, per-class metrics and a confusion matrix.
  4. Saves the per-image probabilities as an .npz file.
  5. Draws a grid of example predictions.
  6. Draws the worst failure cases, for the error autopsy.
  7. For GFNet only, plots the learned frequency filters.

Usage:
    # Test set, with figures
    python predict.py --model gfnet --checkpoint outputs_gfnet_seed42/best_gfnet.pt

    # Validation set, which temperature scaling needs
    python predict.py --model gfnet --checkpoint outputs_gfnet_seed42/best_gfnet.pt \
        --split validation --no-figures

NumPy is used here for the plots and the saved arrays. The assignment allows
that in this script, unlike in modules.py.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import (
    get_dataloaders, DEFAULT_ADNI_ROOT, LABEL_TO_CLASS,
    PIXEL_MEAN, PIXEL_STD,
)
from modules import build_model, count_trainable_parameters


def pick_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU: %s" % torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("No GPU found, running on CPU.")
    return device


def undo_normalisation(image_tensor):
    """
    Turn a normalised tensor back into something I can look at.

    train.py subtracts the mean and divides by the standard deviation, so the
    raw tensor values run from about -0.5 to 3.8. Matplotlib would show that as
    mostly black. This reverses the step so the brain looks like a brain.
    """
    image = image_tensor.squeeze().cpu().numpy()
    image = image * PIXEL_STD + PIXEL_MEAN
    return np.clip(image, 0.0, 1.0)


def run_inference(model, loader, device, use_tta=False):
    """
    Run the model over a whole dataloader and keep everything I need later.

    I keep the images as well as the predictions, because the failure figures
    need to show the actual slices that went wrong. That costs memory, so this
    is fine for 9,000 test slices but would need rethinking for a much larger
    set.

    Test-time augmentation
    ----------------------
    With use_tta, every slice is scored twice: as it is, and mirrored left to
    right. The two probability vectors are averaged.

    Why a mirror and not something else. The training augmentation already
    applies RandomHorizontalFlip, so the model has seen mirrored brains
    throughout training and a mirror is in the distribution it learned. A
    rotation or a shift would be too, but a mirror is the largest change that
    leaves the label untouched, so it is the one most likely to disagree with
    the original and therefore the one most likely to help.

    The honest caveat is the same one the augmentation section carries:
    Alzheimer's atrophy is not always symmetric between hemispheres, so a
    mirrored brain is not perfectly label-preserving. That is an argument for
    measuring the gain rather than assuming it, which is why the plain
    probabilities are kept alongside.

    This costs one extra forward pass per batch and changes no weights.
    """
    model.eval()

    all_probabilities = []
    all_plain = []
    all_labels = []
    all_patients = []
    all_images = []

    with torch.no_grad():
        for images, labels, patient_ids in loader:
            images_on_device = images.to(device, non_blocking=True)

            plain = torch.softmax(model(images_on_device), dim=1)

            if use_tta:
                # dims=[-1] is the width axis, so this is a left-right mirror.
                mirrored_images = torch.flip(images_on_device, dims=[-1])
                mirrored = torch.softmax(model(mirrored_images), dim=1)
                probabilities = (plain + mirrored) / 2.0
            else:
                probabilities = plain

            all_probabilities.append(probabilities.cpu().numpy())
            all_plain.append(plain.cpu().numpy())
            all_labels.append(labels.numpy())
            all_patients.extend(list(patient_ids))
            all_images.append(images.cpu())

    return {
        "probabilities": np.concatenate(all_probabilities),
        # Kept so the gain from averaging can be reported rather than assumed.
        "plain_probabilities": np.concatenate(all_plain),
        "labels": np.concatenate(all_labels),
        "patient_ids": np.array(all_patients),
        "images": torch.cat(all_images),
    }


def print_metrics(probabilities, labels):
    """Print accuracy, a confusion matrix, and per-class precision and recall."""
    predictions = probabilities.argmax(axis=1)

    accuracy = float((predictions == labels).mean())
    print()
    print("RESULTS")
    print("  slices: %d" % len(labels))
    print("  accuracy: %.4f" % accuracy)

    # Confusion matrix. Rows are the truth, columns are the prediction.
    print()
    print("  confusion matrix")
    print("                 predicted NC   predicted AD")
    for true_class in (0, 1):
        row = []
        for predicted_class in (0, 1):
            count = int(((labels == true_class) & (predictions == predicted_class)).sum())
            row.append(count)
        print("  actual %-3s   %12d   %12d" % (LABEL_TO_CLASS[true_class], row[0], row[1]))

    print()
    for class_index in (0, 1):
        class_name = LABEL_TO_CLASS[class_index]
        true_positives = int(((predictions == class_index) & (labels == class_index)).sum())
        false_positives = int(((predictions == class_index) & (labels != class_index)).sum())
        false_negatives = int(((predictions != class_index) & (labels == class_index)).sum())

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0

        print("  %s: precision %.4f  recall %.4f  (missed %d)"
              % (class_name, precision, recall, false_negatives))

    return accuracy


def print_patient_level(probabilities, labels, patient_ids):
    """
    Report accuracy per patient as well as per slice.

    A clinician classifies a patient, not a slice. So this is the number that
    matters clinically. I average each patient's 20 slice probabilities.
    """
    unique_patients = sorted(set(patient_ids.tolist()))

    correct = 0
    for patient in unique_patients:
        mask = patient_ids == patient
        mean_probability = probabilities[mask].mean(axis=0)
        if mean_probability.argmax() == labels[mask][0]:
            correct += 1

    accuracy = correct / len(unique_patients)
    print()
    print("  patient level: %.4f  (%d of %d patients)"
          % (accuracy, correct, len(unique_patients)))
    return accuracy


def figure_example_predictions(results, output_dir, model_name, count=8, seed=0):
    """
    A grid of example predictions, half correct and half wrong.

    This is the "example inputs and outputs" figure the assignment asks for. I
    deliberately include wrong cases. A figure showing only successes tells the
    reader nothing about how the model behaves.
    """
    probabilities = results["probabilities"]
    labels = results["labels"]
    predictions = probabilities.argmax(axis=1)
    was_correct = predictions == labels

    chooser = np.random.default_rng(seed)
    correct_indices = chooser.choice(np.where(was_correct)[0], count // 2, replace=False)
    wrong_indices = chooser.choice(np.where(~was_correct)[0], count // 2, replace=False)
    chosen = list(correct_indices) + list(wrong_indices)

    columns = 4
    rows = (len(chosen) + columns - 1) // columns

    figure, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 3.0 * rows))
    axes = np.atleast_2d(axes)

    for position, index in enumerate(chosen):
        row, column = divmod(position, columns)
        axis = axes[row, column]

        axis.imshow(undo_normalisation(results["images"][index]), cmap="gray", vmin=0, vmax=1)
        axis.axis("off")

        true_name = LABEL_TO_CLASS[int(labels[index])]
        predicted_name = LABEL_TO_CLASS[int(predictions[index])]
        confidence = float(probabilities[index].max())
        mark = "correct" if was_correct[index] else "WRONG"
        colour = "green" if was_correct[index] else "red"

        axis.set_title("true %s / said %s\n%.0f%% confident - %s"
                       % (true_name, predicted_name, 100 * confidence, mark),
                       fontsize=8, color=colour)

    figure.suptitle("%s: example predictions (top row correct, bottom row wrong)" % model_name,
                    fontsize=11)
    figure.tight_layout()

    path = os.path.join(output_dir, "predictions_%s.png" % model_name)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("  wrote %s" % path)


def figure_failure_autopsy(results, output_dir, model_name, count=5):
    """
    The most confident mistakes, which is what the error autopsy needs.

    The assignment asks for 3 to 5 representative failure cases with a diagnosis
    of what triggered each one. I pick the mistakes the model was MOST certain
    about, because those are the dangerous ones. A mistake made at 51%
    confidence would get referred to a human under any sensible threshold. A
    mistake made at 99% confidence would not.

    I split them into false negatives and false positives, because they carry
    different clinical costs. A false negative here means a patient with
    Alzheimer's that the model called normal.
    """
    probabilities = results["probabilities"]
    labels = results["labels"]
    patients = results["patient_ids"]
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    # False negatives: truly AD (1), predicted NC (0).
    false_negative_mask = (labels == 1) & (predictions == 0)
    # False positives: truly NC (0), predicted AD (1).
    false_positive_mask = (labels == 0) & (predictions == 1)

    def most_confident(mask, how_many):
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []
        order = np.argsort(-confidences[indices])
        return list(indices[order][:how_many])

    worst_false_negatives = most_confident(false_negative_mask, count)
    worst_false_positives = most_confident(false_positive_mask, count)

    chosen = worst_false_negatives + worst_false_positives
    if not chosen:
        print("  no failures found, skipping the autopsy figure")
        return

    figure, axes = plt.subplots(2, count, figsize=(2.6 * count, 6.2))
    axes = np.atleast_2d(axes)

    for row, (group, title) in enumerate([
        (worst_false_negatives, "FALSE NEGATIVE: has AD, called normal"),
        (worst_false_positives, "FALSE POSITIVE: normal, called AD"),
    ]):
        for column in range(count):
            axis = axes[row, column]
            axis.axis("off")

            if column >= len(group):
                continue

            index = group[column]
            axis.imshow(undo_normalisation(results["images"][index]), cmap="gray", vmin=0, vmax=1)
            axis.set_title("patient %s\n%.1f%% confident"
                           % (patients[index], 100 * confidences[index]), fontsize=8)

        axes[row, 0].set_ylabel(title, fontsize=9)
        # A visible row label, since axis("off") hides ylabel.
        axes[row, 0].text(-0.08, 0.5, title, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right", fontsize=9,
                          color="red" if row == 0 else "darkorange")

    figure.suptitle("%s: the mistakes the model was most sure about" % model_name, fontsize=11)
    figure.tight_layout()

    path = os.path.join(output_dir, "failures_%s.png" % model_name)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("  wrote %s" % path)

    # Print the details so I can write the autopsy text against real numbers.
    print()
    print("  FAILURE CASES FOR THE AUTOPSY")
    print("  Most confident false negatives (has AD, called normal):")
    for index in worst_false_negatives:
        print("    patient %s  confidence %.4f" % (patients[index], confidences[index]))
    print("  Most confident false positives (normal, called AD):")
    for index in worst_false_positives:
        print("    patient %s  confidence %.4f" % (patients[index], confidences[index]))


def figure_filter_magnitudes(model, output_dir, model_name):
    """
    Plot the frequency filters GFNet learned. Only GFNet has these.

    Each block holds a complex weight of shape (14, 8, embed_dim). I take the
    magnitude and average over the channels, which leaves a 14 by 8 map of how
    strongly that block weights each spatial frequency.

    What to look for. The top-left corner is the lowest frequency, which
    corresponds to large smooth structures. The bottom-right is the highest
    frequency, which corresponds to fine texture and edges. If the low-frequency
    corner dominates, that supports my argument that the model keys on
    large-scale atrophy - enlarged ventricles, thinner cortex - rather than
    fine detail.

    The baseline CNN has no equivalent figure. This is the one thing GFNet gives
    me that a convolutional network cannot.
    """
    if not hasattr(model, "get_filter_magnitudes"):
        return

    magnitudes = model.get_filter_magnitudes()

    columns = 4
    rows = (len(magnitudes) + columns - 1) // columns

    figure, axes = plt.subplots(rows, columns, figsize=(2.8 * columns, 3.4 * rows))
    axes = np.atleast_2d(axes)

    for index in range(rows * columns):
        row, column = divmod(index, columns)
        axis = axes[row, column]

        if index >= len(magnitudes):
            axis.axis("off")
            continue

        magnitude_map = magnitudes[index].numpy()
        image = axis.imshow(magnitude_map, cmap="viridis", aspect="auto")
        axis.set_title("block %d" % index, fontsize=9)
        axis.set_xlabel("frequency (width)", fontsize=7)
        axis.set_ylabel("frequency (height)", fontsize=7)
        axis.tick_params(labelsize=6)
        figure.colorbar(image, ax=axis, fraction=0.046)

    figure.suptitle("%s: learned filter magnitude per block\n"
                    "top-left is low frequency (large structures), "
                    "bottom-right is high frequency (fine texture)" % model_name,
                    fontsize=11)
    figure.tight_layout()

    path = os.path.join(output_dir, "filters_%s.png" % model_name)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("  wrote %s" % path)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a trained ADNI model and produce the report figures."
    )
    parser.add_argument("--model", required=True, choices=["baseline", "gfnet"])
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the .pt file saved by train.py.")
    parser.add_argument("--split", default="test", choices=["test", "validation"],
                        help="Which split to run on. Use validation for temperature scaling.")
    parser.add_argument("--data-root", default=DEFAULT_ADNI_ROOT)
    parser.add_argument("--output-dir", default="figures",
                        help="Where to write the figures and the .npz file.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the seed used for training, or the "
                             "train/validation split will differ.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip the figures and only write the predictions file.")
    parser.add_argument("--slice-context", type=int, default=1, choices=[1, 3],
                        help="Must match the value used for training, or the "
                             "checkpoint will not load.")
    parser.add_argument("--tta", action="store_true",
                        help="Test-time augmentation. Score every slice twice, "
                             "as it is and mirrored left to right, and average "
                             "the two. One extra forward pass per batch, no "
                             "retraining, no change to the weights.")
    parser.add_argument("--depth", type=int, default=8,
                        help="Must match the depth used for training, or the "
                             "checkpoint will not load. 8 is my original, 12 "
                             "is the recipe runs. Ignored by the baseline.")
    parser.add_argument("--patch-size", type=int, default=16,
                        help="Must match the patch size used for training, or "
                             "the checkpoint will not load. Ignored by the "
                             "baseline.")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    os.makedirs(arguments.output_dir, exist_ok=True)

    device = pick_device()

    print()
    print("Loading data from %s" % arguments.data_root)
    # The seed must match training. build_splits uses it to decide which
    # patients go to validation, so a different seed would give me a validation
    # set that overlaps the training patients the model already saw.
    train_loader, validation_loader, test_loader = get_dataloaders(
        adni_root=arguments.data_root,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        num_workers=arguments.num_workers,
        slice_context=arguments.slice_context,
    )

    loader = test_loader if arguments.split == "test" else validation_loader
    print("Running on the %s split: %d batches" % (arguments.split, len(loader)))

    print()
    print("Loading %s from %s" % (arguments.model, arguments.checkpoint))
    # Every shape-affecting argument has to match the training run. Stochastic
    # depth is not among them, because DropPath holds no parameters and is the
    # identity in eval mode, so it never appears in the checkpoint.
    model = build_model(
        arguments.model,
        in_channels=arguments.slice_context,
        patch_size=arguments.patch_size,
        depth=arguments.depth,
    ).to(device)
    model.load_state_dict(torch.load(arguments.checkpoint, map_location=device))
    print("  %d trainable parameters" % count_trainable_parameters(model))

    results = run_inference(model, loader, device, use_tta=arguments.tta)

    print_metrics(results["probabilities"], results["labels"])
    print_patient_level(results["probabilities"], results["labels"], results["patient_ids"])

    # Report what the averaging actually bought, rather than assuming it helped.
    if arguments.tta:
        plain = results["plain_probabilities"]
        labels = results["labels"]
        plain_accuracy = float(((plain[:, 1] >= 0.5).astype(int) == labels).mean())
        tta_accuracy = float(
            ((results["probabilities"][:, 1] >= 0.5).astype(int) == labels).mean()
        )
        print()
        print("  TEST-TIME AUGMENTATION")
        print("    without: %.4f" % plain_accuracy)
        print("    with:    %.4f" % tta_accuracy)
        print("    change:  %+.2f points" % ((tta_accuracy - plain_accuracy) * 100))

    # Save the probabilities. The calibration analysis reads this file.
    predictions_path = os.path.join(
        arguments.output_dir,
        "%s_predictions_%s.npz" % (arguments.split, arguments.model),
    )
    np.savez_compressed(
        predictions_path,
        probabilities=results["probabilities"],
        labels=results["labels"],
        patient_ids=results["patient_ids"],
    )
    print()
    print("  wrote %s" % predictions_path)

    if not arguments.no_figures:
        print()
        print("FIGURES")
        figure_example_predictions(results, arguments.output_dir, arguments.model)
        figure_failure_autopsy(results, arguments.output_dir, arguments.model)
        figure_filter_magnitudes(model, arguments.output_dir, arguments.model)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
