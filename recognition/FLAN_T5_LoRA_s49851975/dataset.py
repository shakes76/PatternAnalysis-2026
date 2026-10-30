"""
Data loading and preprocessing for radiology report lay summarisation.

Dataset: BioLaySumm 2025 LaymanRRG open-source track (Hugging Face).
Each example pairs a radiology report with a lay summary written for patients.

Main steps:
1. Load the official train, validation and test splits.
2. Remove repeated reports from the training set.
3. Remove validation reports that also appear in training (data leakage),
   then remove repeated reports inside the validation set.
4. Add a task prompt to each report and tokenize reports and summaries.
5. Pad each batch to its own longest sequence (dynamic padding) and set the
   padded label positions to -100 so the loss ignores them (target masking).

Run `python dataset.py` to print the data audit numbers and check one batch.
"""

import statistics

import textstat
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

DATASET_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
MODEL_NAME = "google/flan-t5-base"

# Task prompt put in front of every report.
PROMPT = "Simplify this radiology report for a patient: "

# Longer inputs are cut; T5 has no 512-token limit, so the tokenizer warning about 512 can be ignored.
MAX_INPUT_LENGTH = 640

# Longer summaries are cut; 640 cuts fewer than 0.2% of training summaries.
MAX_TARGET_LENGTH = 640

# Label value that the cross-entropy loss skips.
IGNORE_INDEX = -100


def normalise_report(text):
    """Lowercase the text and squeeze repeated spaces so that copies of a report match."""
    words = text.lower().split()
    return " ".join(words)


def remove_duplicates(examples):
    """Keep the first example of each report and return a list of (report, summary) pairs."""
    pairs = []
    seen_reports = set()
    for example in examples:
        key = normalise_report(example["radiology_report"])
        if key not in seen_reports:
            seen_reports.add(key)
            pairs.append((example["radiology_report"], example["layman_report"]))
    return pairs


def remove_leaked_reports(validation_examples, train_pairs):
    """Drop validation examples whose report also appears in the training set."""
    train_reports = set()
    for report, summary in train_pairs:
        train_reports.add(normalise_report(report))

    kept_examples = []
    for example in validation_examples:
        if normalise_report(example["radiology_report"]) not in train_reports:
            kept_examples.append(example)
    return kept_examples


def prepare_splits():
    """Load the official splits and return leakage-free train, validation and test data."""
    raw_splits = load_dataset(DATASET_NAME)

    # Training set: remove repeated reports.
    train_pairs = remove_duplicates(raw_splits["train"])

    # Validation set: remove reports seen in training, then remove repeated reports.
    validation_examples = remove_leaked_reports(raw_splits["validation"], train_pairs)
    validation_pairs = remove_duplicates(validation_examples)

    # The test set has no reference summaries, so only the reports are kept for demo inference.
    test_reports = list(raw_splits["test"]["radiology_report"])

    leaked_count = len(raw_splits["validation"]) - len(validation_examples)
    print(f"Train: {len(raw_splits['train'])} rows, {len(train_pairs)} after removing duplicates")
    print(f"Validation: {len(raw_splits['validation'])} rows, {leaked_count} leaked into train, "
          f"{len(validation_pairs)} after removing leaks and duplicates")
    print(f"Test: {len(test_reports)} reports (no reference summaries)")

    return train_pairs, validation_pairs, test_reports


def add_prompt(report):
    """Put the task prompt in front of a report."""
    return PROMPT + report


class RadiologySummaryDataset(Dataset):
    """Tokenized (report, summary) pairs for a sequence-to-sequence model."""

    def __init__(self, pairs, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

        model_inputs = []
        summaries = []
        for report, summary in pairs:
            model_inputs.append(add_prompt(report))
            summaries.append(summary)

        # Tokenize once and cut long sequences; padding is done later in collate_batch.
        self.input_ids = tokenizer(model_inputs, max_length=MAX_INPUT_LENGTH, truncation=True)["input_ids"]
        self.labels = tokenizer(text_target=summaries, max_length=MAX_TARGET_LENGTH, truncation=True)["input_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index]

    def collate_batch(self, batch):
        """Pad one batch to its longest input and longest label (dynamic padding)."""
        longest_input = 0
        longest_label = 0
        for input_ids, labels in batch:
            longest_input = max(longest_input, len(input_ids))
            longest_label = max(longest_label, len(labels))

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for input_ids, labels in batch:
            input_padding = longest_input - len(input_ids)
            label_padding = longest_label - len(labels)
            batch_input_ids.append(input_ids + [self.pad_token_id] * input_padding)
            # 1 marks a real token and 0 marks padding.
            batch_attention_mask.append([1] * len(input_ids) + [0] * input_padding)
            # Padded label positions get -100 so they add nothing to the loss.
            batch_labels.append(labels + [IGNORE_INDEX] * label_padding)

        return {
            "input_ids": torch.tensor(batch_input_ids),
            "attention_mask": torch.tensor(batch_attention_mask),
            "labels": torch.tensor(batch_labels),
        }


def build_dataloader(pairs, tokenizer, batch_size, shuffle):
    """Create a DataLoader that tokenizes the pairs and pads each batch on the fly."""
    dataset = RadiologySummaryDataset(pairs, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_batch)


def count_tokens(texts, tokenizer):
    """Return the token count of each text, including the end-of-sequence token."""
    lengths = []
    for token_ids in tokenizer(texts)["input_ids"]:
        lengths.append(len(token_ids))
    return lengths


def percent_over_limit(lengths, limit):
    """Return the percentage of sequences longer than the limit (these get truncated)."""
    over_limit = 0
    for length in lengths:
        if length > limit:
            over_limit += 1
    return 100 * over_limit / len(lengths)


def average_flesch_kincaid_grade(texts):
    """Return the mean Flesch-Kincaid Grade Level (FKGL), scoring one text at a time."""
    total = 0.0
    for text in texts:
        total += textstat.flesch_kincaid_grade(text)
    return total / len(texts)


def run_data_audit(train_pairs, validation_pairs, tokenizer):
    """Print token length and reading level statistics for the clean splits."""
    for split_name, pairs in [("Train", train_pairs), ("Validation", validation_pairs)]:
        reports = []
        model_inputs = []
        summaries = []
        for report, summary in pairs:
            reports.append(report)
            model_inputs.append(add_prompt(report))
            summaries.append(summary)

        report_lengths = count_tokens(reports, tokenizer)
        input_lengths = count_tokens(model_inputs, tokenizer)
        summary_lengths = count_tokens(summaries, tokenizer)

        print(f"\n{split_name} audit ({len(pairs)} pairs)")
        print(f"  Median report length: {statistics.median(report_lengths)} tokens")
        print(f"  Median summary length: {statistics.median(summary_lengths)} tokens")
        print(f"  Inputs (prompt + report) over {MAX_INPUT_LENGTH} tokens: "
              f"{percent_over_limit(input_lengths, MAX_INPUT_LENGTH):.3f}%")
        print(f"  Summaries over {MAX_TARGET_LENGTH} tokens: "
              f"{percent_over_limit(summary_lengths, MAX_TARGET_LENGTH):.3f}%")
        print(f"  Mean FKGL of reference summaries: {average_flesch_kincaid_grade(summaries):.2f}")


def check_batches(pairs, tokenizer):
    """Load a few hundred pairs through the DataLoader and print the first batch."""
    loader = build_dataloader(pairs, tokenizer, batch_size=8, shuffle=False)

    number_of_batches = 0
    first_batch = None
    for batch in loader:
        number_of_batches += 1
        if number_of_batches == 1:
            first_batch = batch
    print(f"\nBatch check: {len(pairs)} pairs loaded in {number_of_batches} batches")

    # Keep the real label tokens of the first example and drop the -100 padding.
    first_label_tokens = []
    for token_id in first_batch["labels"][0].tolist():
        if token_id != IGNORE_INDEX:
            first_label_tokens.append(token_id)

    padded_inputs = (first_batch["attention_mask"] == 0).sum().item()
    masked_labels = (first_batch["labels"] == IGNORE_INDEX).sum().item()
    print(f"  input_ids shape: {tuple(first_batch['input_ids'].shape)}")
    print(f"  labels shape: {tuple(first_batch['labels'].shape)}")
    print(f"  Padded input positions: {padded_inputs}, label positions set to -100: {masked_labels}")
    print(f"  First input: {tokenizer.decode(first_batch['input_ids'][0], skip_special_tokens=True)}")
    print(f"  First label: {tokenizer.decode(first_label_tokens, skip_special_tokens=True)}")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_pairs, validation_pairs, test_reports = prepare_splits()
    run_data_audit(train_pairs, validation_pairs, tokenizer)

    # Small run on a few hundred training pairs to check batching.
    check_batches(train_pairs[:200], tokenizer)
