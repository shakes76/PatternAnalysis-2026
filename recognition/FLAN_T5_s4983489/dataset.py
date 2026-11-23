"""Load and tokenize LaymanRRG data for FLAN-T5 training."""

import argparse
import json
import math
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq


DATASET_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
TASK_PREFIX = (
    "Explain this radiology report in plain English for a patient. "
    "Preserve the findings, negation, and uncertainty:\n"
)


def load_data():
    """Load the official train, validation and test splits."""
    return load_dataset(DATASET_NAME)


def audit_overlap(train_data, validation_data):
    """Compare exact, unmodified report texts and targets across the two splits."""
    training_targets = {}
    for row in train_data:
        training_targets.setdefault(row["radiology_report"], set()).add(row["layman_report"])

    shared_reports = set()
    identical_targets = 0
    unseen_indices = []
    for index, row in enumerate(validation_data):
        report = row["radiology_report"]
        if report in training_targets:
            shared_reports.add(report)
            identical_targets += row["layman_report"] in training_targets[report]
        else:
            unseen_indices.append(index)

    overlapping = len(validation_data) - len(unseen_indices)
    summary = {
        "train_reports": len(train_data),
        "validation_reports": len(validation_data),
        "overlapping_validation_reports": overlapping,
        "unseen_validation_reports": len(unseen_indices),
        "unique_shared_reports": len(shared_reports),
        "overlapping_validation_reports_with_identical_target": identical_targets,
        "overlapping_validation_reports_without_identical_target": overlapping - identical_targets,
        "training_unique_reports_with_multiple_targets": sum(
            len(targets) > 1 for targets in training_targets.values()
        ),
    }
    # Indices refer to the supplied validation split and preserve its row order.
    return summary, unseen_indices


def prepare_splits(data, seed=42, dev_fraction=0.05):
    """Keep official validation for final evaluation and split training by report text."""
    original_train, final = data["train"], data["validation"]
    reports = list(original_train["radiology_report"])
    final_reports = set(final["radiology_report"])
    remaining_reports = sorted(set(reports) - final_reports)
    random.Random(seed).shuffle(remaining_reports)

    # Keep every target variant of a source report in the same partition.
    development_count = math.ceil(len(remaining_reports) * dev_fraction)
    development_reports = set(remaining_reports[:development_count])
    training_reports = set(remaining_reports[development_count:])
    train_indices = [index for index, report in enumerate(reports) if report in training_reports]
    development_indices = [
        index for index, report in enumerate(reports) if report in development_reports
    ]
    train = original_train.select(train_indices)
    development = original_train.select(development_indices)

    audit = {
        "seed": seed,
        "development_fraction": dev_fraction,
        "original_train_rows": len(original_train),
        "removed_train_rows": len(original_train) - len(train) - len(development),
        "removed_train_unique_reports": len(set(reports) & final_reports),
        "train_rows": len(train),
        "development_rows": len(development),
        "final_rows": len(final),
        "train_unique_reports": len(training_reports),
        "development_unique_reports": len(development_reports),
        "final_unique_reports": len(final_reports),
        "train_development_shared_reports": len(training_reports & development_reports),
        "train_final_shared_reports": len(training_reports & final_reports),
        "development_final_shared_reports": len(development_reports & final_reports),
    }
    return train, development, final, audit


def tokenize_data(data, tokenizer, max_input_length=512, max_target_length=512):
    """Tokenize one split into input_ids, attention_mask, and labels."""

    def tokenize_batch(batch):
        # The encoder receives the prompt and source report only.
        inputs = [TASK_PREFIX + report for report in batch["radiology_report"]]
        encoded = tokenizer(
            inputs, max_length=max_input_length, truncation=True
        )

        # Reference explanations are decoder targets.
        targets = tokenizer(
            text_target=batch["layman_report"],
            max_length=max_target_length,
            truncation=True,
        )
        encoded["labels"] = targets["input_ids"]
        return encoded

    # Leave padding to the batch collator.
    # Mapping preserves the original full texts for evaluation.
    return data.map(
        tokenize_batch, batched=True, remove_columns=data.column_names
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-output", help="Write an overlap audit to a new JSON file")
    args = parser.parse_args()
    data = load_data()

    if args.audit_output is not None:
        summary, unseen_indices = audit_overlap(data["train"], data["validation"])
        with Path(args.audit_output).open("x", encoding="utf-8") as file:
            json.dump(
                {**summary, "unseen_validation_indices": unseen_indices}, file, indent=2
            )
        print(json.dumps(summary, indent=2))
        return

    for split_name, split in data.items():
        print(f"{split_name}: {len(split)} samples")

    example = data["train"][0]
    print("\nRadiology report (input):")
    print(example["radiology_report"])
    print("\nLayperson report (target):")
    print(example["layman_report"])

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    examples = tokenize_data(data["train"].select(range(2)), tokenizer)

    # Use -100 for target padding so it is excluded from the loss.
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)
    batch = collator([examples[0], examples[1]])
    print("\nTokenized batch:")
    for name, tensor in batch.items():
        print(f"{name}: {tuple(tensor.shape)}")
    print(f"Ignored target padding positions: {(batch['labels'] == -100).sum().item()}")


if __name__ == "__main__":
    main()
