"""LaymanRRG loading and project-specific identifier grouping/preprocessing.

Data source: https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track
Uses Hugging Face Datasets load/filter/select APIs. Source identifiers do not
establish patient-level separation; see README for the audit limitations.
"""

import ast

from datasets import load_dataset
from config import INSTRUCTION

import random



DATASET_ID = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
# Stores the dataset name in case we need it later. 

def get_image_identifiers(example): 
    """Extract source-qualified image or session identifiers from one row.
    example will be a python dictionary representing one dataset row.
    For example:
    {
        "source": "PadChest",
        "images_path": "some_image.png",
        "radiology_report": "No abnormalities.",
        "layman_report": "The scan did not show any abnormalities."
    }
    """
    source = example["source"] # looks up the "source" key and assigns it's value, PadChest, to source
    raw_paths = example["images_path"]

    if source == "OpenI":
        paths = ast.literal_eval(raw_paths) # Converts OpenI's stored list into an actual list
    else:
        paths = [raw_paths] # ALl other sources get their identifiers put into a singleton list

    return {(source, path.strip()) for path in paths}


def load_audited_data():
    """
    Load the dataset and excludes training rows that share validation identifiers.
    Returns both the original splits and the filtered training data.
    Leaves validation and test set unchanged and doesn't create a development split yet.
    """
    dataset = load_dataset(DATASET_ID)
    # Loads the original splits, reusing cached downloads whenever possible

    validation_identifiers = set()

    for example in dataset["validation"]:
        validation_identifiers.update(get_image_identifiers(example)) 
        # Adds identifiers from each validation row and adds them to the set

    def keep_example(example):
        """Helper function that extracts the training row's identifiers and
        returns True when the two sets share no elements, meaning the row should be kept."""
        identifiers = get_image_identifiers(example)
        return identifiers.isdisjoint(validation_identifiers)

    clean_train = dataset["train"].filter(keep_example) # Applies that decision to every training row

    return dataset, clean_train # Returns two objects in a tuple, the original dataset and the filtered split




"""
Create a reusable function that converts a report-summary pair into model inputs and labels. 
"""

def tokenize_example(example, tokenizer):
    """Tokenise a complete report–summary pair without padding or truncation."""
    model_input = INSTRUCTION + example["radiology_report"]

    encoded = tokenizer(
        model_input,
        truncation=False,
        padding=False,
    ) # Produces input IDs and an attention mask, no padding or truncation

    target = tokenizer(
        text_target=example["layman_report"],
        truncation=False,
        padding=False,
    ) # Tokenises the layman summary separately, no padding or truncation

    encoded["labels"] = target["input_ids"]

    return encoded # Returns the prepared example



"""
Finds connected groups of records.
For example, if row A shares an image with B, and B shares another image with C, 
all three must stay together. Grouping only by complete images_path strings would
miss these connections. 
"""

def build_record_groups(split):
    """Group row indices connected by shared image/session identifiers."""
    parents = list(range(len(split))) # Gives every row its own group, each entry points to itself
    identifier_owner = {} # Records the first row where each identifier appears

    def find_root(index): # Finds a row's group representative
        while parents[index] != index: # Follow parent links until reaching a representative that points to itself
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for row_index, example in enumerate(split): # Visits each row along with its position
        for identifier in get_image_identifiers(example):
            if identifier in identifier_owner: # Detects an identifier already seen in another row
                current_root = find_root(row_index)
                previous_root = find_root(identifier_owner[identifier])
                parents[current_root] = previous_root # Finds both group reps and links their groups together
            else:
                identifier_owner[identifier] = row_index # Records first occurence of a new identifier

    groups = {}

    for row_index in range(len(split)):
        root = find_root(row_index)
        groups.setdefault(root, []).append(row_index) 
        # Creates an empty list for that rep if needed, then appends the row index

    return list(groups.values())
    # Returns a list of groups each containing row indices



"""
Reserve about 5 percent of identifier groups for development while keeping each group intact.
"""


def split_train_development(clean_train, development_fraction=0.05, seed=42):
    """Split connected identifier groups into training and development data."""
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between 0 and 1")

    groups = build_record_groups(clean_train) # Gets connected groups we just verified

    rng = random.Random(seed) 
    rng.shuffle(groups) # Shuffles group order based on a random seed

    development_group_count = round(len(groups) * development_fraction)
    # Calculates how many groups to reserve. May vary slightly from 5 percent

    development_indices = sorted(
        index
        for group in groups[:development_group_count]
        for index in group
    ) # Flattens into row-index list

    training_indices = sorted(
        index
        for group in groups[development_group_count:]
        for index in group
    ) # Flattens into row-index list

    training = clean_train.select(training_indices)
    development = clean_train.select(development_indices)
    # Creates the two datasets using those indices

    return training, development