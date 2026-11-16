"""
dataset.py

Data loading for the ADNI brain MRI dataset (AD = Alzheimer's Disease,
NC = Normal Cognitive / Cognitively Normal).

What I found when I audited the data on Rangpur:

    /home/groups/comp3710/ADNI/AD_NC/
        train/AD   10400 images from 520 patients
        train/NC   11120 images from 556 patients
        test/AD     4460 images from 223 patients
        test/NC     4540 images from 227 patients

    File names look like  1003730_100.jpeg
                          ^^^^^^^ ^^^
                          patient slice

    Every patient contributes exactly 20 slices.
    Images are 256 wide by 240 high, greyscale, pixel values 0 to 255.
    No patient appears in both the train and test folders, so the split that
    ships with the dataset is already clean.

The single most important thing this file does
----------------------------------------------
Each patient has 20 slices cut from the same 3D scan, so those 20 images look
very similar to each other. If I split the images randomly, some slices of a
patient land in training and the rest land in validation. The model can then
recognise the patient rather than the disease, and validation accuracy shoots
up for completely the wrong reason. The test score would then collapse and I
would have no idea why.

So every split in this file happens at the PATIENT level. I pick which
patients go where first, then take all 20 of their slices along with them.
There is an assertion at the bottom of build_splits that fails loudly if a
patient ever ends up in two splits, because this is the kind of bug that
silently makes results look good.
"""

import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# Where the data lives on Rangpur.
DEFAULT_ADNI_ROOT = "/home/groups/comp3710/ADNI/AD_NC"

# Class names and the integer label each one maps to.
# NC is 0 and AD is 1, so "positive class" means Alzheimer's. That matters
# later when I compute precision and recall, because those are defined with
# respect to the positive class.
CLASS_TO_LABEL = {"NC": 0, "AD": 1}
LABEL_TO_CLASS = {0: "NC", 1: "AD"}

# The size every image gets resized to before it reaches the model.
# The originals are 256x240, which is not square. GFNet needs a fixed square
# input because it works on a grid of patches, so I resize rather than crop.
# This does stretch the image slightly (240 -> 224 vertically is a bigger
# squeeze than 256 -> 224 horizontally). I chose stretching over cropping
# because cropping could cut off the outer cortex, which is exactly the tissue
# that atrophies in Alzheimer's. The distortion is identical for every image,
# so the model sees a consistent world.
IMAGE_SIZE = 224

# Normalisation values for the greyscale channel.
# These are placeholders of 0.5 until I run compute_dataset_mean_and_std on the
# training split and paste the real numbers in. See the note in that function.
PIXEL_MEAN = 0.1159
PIXEL_STD  = 0.2236
def get_patient_id_from_filename(file_name):
    """
    Pull the patient ID out of a file name like "1003730_100.jpeg".

    The ID is everything before the first underscore. I split on "_" rather
    than using a regular expression because the format is simple and this is
    easier to read. The audit confirmed all 30520 files follow this pattern,
    so I am not handling exceptions that do not exist.
    """
    return file_name.split("_")[0]


def slice_number_from_filename(file_name):
    """
    Return the slice number from a name like 1003730_100.jpeg, or None.

    The number after the underscore is the slice position within that
    patient's 3D scan. The number before it is the ID.
    """
    base = os.path.basename(file_name)
    stem = os.path.splitext(base)[0]
    if "_" not in stem:
        return None
    tail = stem.rsplit("_", 1)[1]
    return int(tail) if tail.isdigit() else None


def scan_class_folder(folder_path, label):
    """
    Look at one folder (for example train/AD) and return a list of
    (image_path, label, patient_id) tuples, one per image.

    Sorting the file names keeps the order identical between runs, which
    matters because the train/validation split uses a seeded shuffle and I want
    the same split every time I run the code.
    """
    samples = []

    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.lower().endswith((".jpeg", ".jpg", ".png")):
            continue

        image_path = os.path.join(folder_path, file_name)
        patient_id = get_patient_id_from_filename(file_name)
        samples.append((image_path, label, patient_id))

    return samples


def scan_split_folder(adni_root, split_name):
    """
    Read both class folders for one of the shipped splits ("train" or "test")
    and return all samples together.
    """
    all_samples = []

    for class_name, label in CLASS_TO_LABEL.items():
        folder_path = os.path.join(adni_root, split_name, class_name)

        if not os.path.isdir(folder_path):
            raise FileNotFoundError("Expected to find a folder at %s" % folder_path)

        all_samples.extend(scan_class_folder(folder_path, label))

    return all_samples


def group_patients_by_label(samples):
    """
    Build a dict of {label: sorted list of unique patient IDs}.

    I keep the two classes apart so the validation split can take a fair share
    of AD patients and a fair share of NC patients. If I split the combined
    pool at random, the validation set could end up lopsided just by luck, and
    then validation accuracy would not mean what I think it means.
    """
    patients_by_label = {}

    for _, label, patient_id in samples:
        if label not in patients_by_label:
            patients_by_label[label] = set()
        patients_by_label[label].add(patient_id)

    # Sort so the ordering is deterministic before shuffling.
    return {label: sorted(ids) for label, ids in patients_by_label.items()}


def build_splits(adni_root=DEFAULT_ADNI_ROOT, validation_fraction=0.15, seed=42):
    """
    Produce the three sample lists: training, validation and test.

    The test set is taken exactly as shipped and is never touched again. The
    audit showed it shares no patients with the training folder, so it is a
    genuine held-out set.

    The validation set is carved out of the training folder by patient. With
    validation_fraction = 0.15 that is roughly 78 AD patients and 83 NC
    patients held back, which is enough to make the validation signal stable
    without starving the training set.

    The seed is fixed so the split is identical on every run. The assignment
    asks about reproducibility of results, and a split that changes between
    runs would make the baseline and GFNet numbers incomparable.

    Returns:
        (train_samples, validation_samples, test_samples)
        where each is a list of (image_path, label, patient_id).
    """
    all_training_pool = scan_split_folder(adni_root, "train")
    test_samples = scan_split_folder(adni_root, "test")

    patients_by_label = group_patients_by_label(all_training_pool)

    # Decide which patients go to validation, one class at a time.
    validation_patient_ids = set()
    shuffler = random.Random(seed)

    for label in sorted(patients_by_label.keys()):
        patient_ids = list(patients_by_label[label])
        shuffler.shuffle(patient_ids)

        how_many_for_validation = int(round(len(patient_ids) * validation_fraction))
        validation_patient_ids.update(patient_ids[:how_many_for_validation])

    # Now send each image to the split its patient belongs to.
    train_samples = []
    validation_samples = []

    for sample in all_training_pool:
        _, _, patient_id = sample
        if patient_id in validation_patient_ids:
            validation_samples.append(sample)
        else:
            train_samples.append(sample)

    # Safety check. If this ever fires, the splits are leaking and every number
    # downstream is worthless, so I would rather crash here than find out from
    # a suspiciously high validation score.
    check_no_patient_overlap(train_samples, validation_samples, test_samples)

    return train_samples, validation_samples, test_samples


def check_no_patient_overlap(train_samples, validation_samples, test_samples):
    """
    Confirm that no patient ID appears in more than one split.

    This is the guard that makes the leakage-free claim in my report something
    I can actually stand behind, rather than something I assume is true.
    """
    def patient_ids_in(samples):
        return {patient_id for _, _, patient_id in samples}

    train_patients = patient_ids_in(train_samples)
    validation_patients = patient_ids_in(validation_samples)
    test_patients = patient_ids_in(test_samples)

    pairs_to_check = [
        ("train", train_patients, "validation", validation_patients),
        ("train", train_patients, "test", test_patients),
        ("validation", validation_patients, "test", test_patients),
    ]

    for name_a, set_a, name_b, set_b in pairs_to_check:
        shared = set_a & set_b
        if shared:
            raise AssertionError(
                "Patient leakage between %s and %s: %d shared patients, e.g. %s"
                % (name_a, name_b, len(shared), sorted(shared)[:5])
            )

def build_transforms(is_training, augmentation_strength="normal", channels=1):
    """
    Build the image preprocessing pipeline.

    augmentation_strength controls how hard the training augmentation is.

      "normal" - what I used for the first runs.
      "strong" - wider rotation and shift, random scaling, and brightness
                 jitter. I added this after the 80-epoch runs showed clear
                 overfitting: validation accuracy reached 0.9326 while test
                 accuracy stayed at 0.7360.

    The brightness jitter matters most. My brightness_baseline.py experiment
    showed a mean-intensity rule alone reaches 0.585 on the test split. So
    average brightness carries real signal, and the model can lean on it.
    Randomising brightness during training removes that shortcut and forces the
    model to use shape instead.
    """
    # With slice_context 3 the image arriving here already has three distinct
    # channels, one per brain slice. Grayscale would average them into one and
    # then copy that average back across all three, destroying exactly the
    # depth information this mode exists to provide. So it is only applied in
    # the single-slice case.
    shared_steps = []
    if channels == 1:
        shared_steps.append(transforms.Grayscale(num_output_channels=1))
    shared_steps.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)))

    if not is_training:
        augmentation_steps = []
    elif augmentation_strength == "strong":
        augmentation_steps = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.10, 0.10), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
        ]
    else:
        augmentation_steps = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        ]

    # The same constants are reused for every channel. Adjacent slices of one
    # scan have almost identical intensity statistics, so measuring a separate
    # mean per channel would produce three nearly equal numbers.
    final_steps = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[PIXEL_MEAN] * channels,
                             std=[PIXEL_STD] * channels),
    ]

    return transforms.Compose(shared_steps + augmentation_steps + final_steps)

class ADNIDataset(Dataset):
    """
    Wraps a list of (image_path, label, patient_id) so PyTorch can iterate it.

    I keep the patient ID on each item even though the model never sees it,
    because the failure analysis later needs to know which patient a wrong
    prediction came from. If several mistakes trace back to one patient, that
    is a very different finding from mistakes scattered across many patients.
    """

    def __init__(self, samples, transform, slice_context=1):
        """
        slice_context 1 returns one slice as one channel, which is the original
        behaviour.

        slice_context 3 returns the slice plus its two neighbours, stacked as
        three channels. One 2D slice shows a cross-section. It cannot show how
        a structure changes as you move through the brain, and ventricular
        enlargement is exactly that kind of change. Three slices give the model
        that gradient.
        """
        if slice_context not in (1, 3):
            raise ValueError(
                "slice_context must be 1 or 3, got %r. Three channels are "
                "stacked with PIL's RGB merge, which takes exactly three bands."
                % slice_context
            )

        self.samples = samples
        self.transform = transform
        self.slice_context = slice_context

        # Only built when it is needed, because it walks every sample.
        self.slices_by_patient = None
        if slice_context > 1:
            self.slices_by_patient = self._index_slices_by_patient(samples)

    @staticmethod
    def _index_slices_by_patient(samples):
        """
        Map each patient to {slice number: path}.

        The neighbours of a slice have to come from the SAME patient. Reaching
        into another patient's scan would mix two people into one input, which
        is worse than the leakage this project exists to avoid.
        """
        index = {}
        for image_path, _label, patient_id in samples:
            slice_number = slice_number_from_filename(image_path)
            if slice_number is None:
                continue
            index.setdefault(patient_id, {})[slice_number] = image_path
        return index

    def _neighbour_paths(self, image_path, patient_id):
        """
        Return the paths of the slice before, the slice itself, and the slice
        after, in that order.

        At the ends of a patient's stack there is no neighbour on one side, so
        the end slice is repeated. That is the standard treatment and it means
        every item still returns exactly three channels.
        """
        available = self.slices_by_patient.get(patient_id, {})
        centre = slice_number_from_filename(image_path)

        if centre is None or centre not in available:
            # Filename did not parse. Fall back to repeating this slice, so a
            # malformed name costs one weak sample instead of crashing a run.
            return [image_path, image_path, image_path]

        ordered = sorted(available.keys())
        position = ordered.index(centre)

        before = ordered[max(position - 1, 0)]
        after = ordered[min(position + 1, len(ordered) - 1)]

        return [available[before], available[centre], available[after]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, patient_id = self.samples[index]

        if self.slice_context == 1:
            # Open and immediately convert, so the file handle does not stay
            # open. "L" is PIL's single channel greyscale mode.
            with Image.open(image_path) as raw_image:
                image = raw_image.convert("L")
                image_tensor = self.transform(image)
            return image_tensor, label, patient_id

        # Three slices, merged into one image BEFORE the transform runs.
        #
        # This ordering is the whole trick. Transforming each slice separately
        # would draw a different random rotation and shift for each one, so the
        # three channels would no longer line up and the depth signal would be
        # destroyed by the augmentation meant to help. Merging first means one
        # random transform is drawn and applied to all three identically.
        bands = []
        reference_size = None
        for path in self._neighbour_paths(image_path, patient_id):
            with Image.open(path) as raw_image:
                band = raw_image.convert("L")
                if reference_size is None:
                    reference_size = band.size
                elif band.size != reference_size:
                    band = band.resize(reference_size)
                bands.append(band.copy())

        merged = Image.merge("RGB", tuple(bands))
        image_tensor = self.transform(merged)

        return image_tensor, label, patient_id


def get_dataloaders(
    adni_root=DEFAULT_ADNI_ROOT,
    batch_size=64,
    validation_fraction=0.15,
    seed=42,
    num_workers=4,
    augmentation_strength="normal",
    slice_context=1,
):
    """
    The one function the training script calls.

    Returns (train_loader, validation_loader, test_loader).

    num_workers defaults to 4 because the Rangpur nodes allow a maximum of 8
    CPUs and I want to leave room for the main process.

    Only the training loader shuffles. Validation and test run in a fixed order
    so results line up between runs and I can trace a prediction back to a
    specific image.
    """
    train_samples, validation_samples, test_samples = build_splits(
        adni_root=adni_root,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    train_dataset = ADNIDataset(
        train_samples,
        build_transforms(
            is_training=True,
            augmentation_strength=augmentation_strength,
            channels=slice_context,
        ),
        slice_context=slice_context,
    )
    validation_dataset = ADNIDataset(
        validation_samples,
        build_transforms(is_training=False, channels=slice_context),
        slice_context=slice_context,
    )
    test_dataset = ADNIDataset(
        test_samples,
        build_transforms(is_training=False, channels=slice_context),
        slice_context=slice_context,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,   # speeds up moving batches to the GPU
        drop_last=True,    # keeps every batch the same size, which batch norm prefers
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, validation_loader, test_loader


def describe_split(split_name, samples):
    """Print a short summary of one split, for the sanity check below."""
    patient_ids = {patient_id for _, _, patient_id in samples}

    label_counts = {}
    for _, label, _ in samples:
        label_counts[label] = label_counts.get(label, 0) + 1

    print("%-12s %6d images   %4d patients   " % (split_name, len(samples), len(patient_ids)), end="")
    for label in sorted(label_counts.keys()):
        print("%s: %d   " % (LABEL_TO_CLASS[label], label_counts[label]), end="")
    print()


def compute_dataset_mean_and_std(adni_root=DEFAULT_ADNI_ROOT, max_images=2000):
    """
    Work out the average pixel value and spread across the TRAINING split only.

    Why training only: validation and test are meant to stand in for data the
    model has never seen. If I calculate the normalisation constants using
    them, a small amount of information about those sets leaks into training.
    It is a mild form of leakage but it is still leakage, and it is free to
    avoid.

    I sample a subset rather than all 21520 training images because the mean
    settles down long before then and I do not want this taking minutes.

    Once this prints real numbers, paste them into PIXEL_MEAN and PIXEL_STD at
    the top of the file.
    """
    train_samples, _, _ = build_splits(adni_root=adni_root)

    sampler = random.Random(0)
    subset = sampler.sample(train_samples, min(max_images, len(train_samples)))

    to_tensor = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    running_sum = 0.0
    running_sum_of_squares = 0.0
    pixel_count = 0

    for image_path, _, _ in subset:
        with Image.open(image_path) as raw_image:
            tensor = to_tensor(raw_image.convert("L"))

        running_sum += tensor.sum().item()
        running_sum_of_squares += (tensor ** 2).sum().item()
        pixel_count += tensor.numel()

    mean = running_sum / pixel_count
    # Standard deviation from the mean of squares minus the square of the mean.
    variance = (running_sum_of_squares / pixel_count) - (mean ** 2)
    std = variance ** 0.5

    print("Measured over %d training images:" % len(subset))
    print("  PIXEL_MEAN = %.4f" % mean)
    print("  PIXEL_STD  = %.4f" % std)

    return mean, std


if __name__ == "__main__":
    # Sanity check. Run this before wiring the loaders into training:
    #     python dataset.py
    #
    # It confirms the splits are the size I expect, that no patient is in two
    # splits, and that a real batch comes out with the right shape.

    print("Building splits...")
    train_samples, validation_samples, test_samples = build_splits()

    print()
    print("SPLIT SUMMARY")
    describe_split("train", train_samples)
    describe_split("validation", validation_samples)
    describe_split("test", test_samples)

    print()
    print("No patient appears in two splits (the assertion in build_splits passed).")

    print()
    print("Loading one batch to check shapes...")
    train_loader, validation_loader, test_loader = get_dataloaders(batch_size=8, num_workers=0)

    images, labels, patient_ids = next(iter(train_loader))

    print("  image batch shape: %s" % (tuple(images.shape),))
    print("  label batch shape: %s" % (tuple(labels.shape),))
    print("  pixel range after normalising: %.3f to %.3f" % (images.min().item(), images.max().item()))
    print("  first few patient IDs in the batch: %s" % list(patient_ids[:4]))

    print()
    print("Now run compute_dataset_mean_and_std() and paste the numbers into the")
    print("constants at the top of this file, then re-run this check.")
