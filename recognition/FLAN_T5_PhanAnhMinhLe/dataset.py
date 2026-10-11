"""Local reports, task prompts, and seq2seq tokenisation."""

from pathlib import Path
import warnings

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "BioLaySumm2025-LaymanRRG-opensource-track"


def load_reports(data_dir=DATA_DIR):
    """Preserve the supplied train, validation, and test splits."""
    data_dir = Path(data_dir)
    splits = DatasetDict()
    for split in ("train", "validation", "test"):
        file = data_dir / "data" / f"{split}-00000-of-00001.parquet"
        splits[split] = Dataset.from_dict(pq.read_table(file).to_pydict())
        columns = ("radiology_report",) if split == "test" else (
            "radiology_report", "layman_report"
        )
        for column in columns:
            if any(not isinstance(text, str) or not text.strip()
                   for text in splits[split][column]):
                raise ValueError(f"Empty or invalid {column} in {split}")
    ids = {split: set(zip(data["source"], data["images_path"]))
           for split, data in splits.items()}
    for first, second in (("train", "validation"), ("train", "test"),
                          ("validation", "test")):
        overlap = len(ids[first] & ids[second])
        if overlap:
            warnings.warn(f"{overlap} shared source/image IDs in {first}/{second}; "
                          "published splits preserved", stacklevel=2)
    return splits


def make_prompt(report):
    return (
        "Rewrite this radiology report in plain English for a patient at a "
        "sixth-grade reading level. Preserve all findings, negations, and "
        "uncertainty. Do not add diagnoses or advice. Explain medical terms "
        "simply.\nRadiology report: " + report.strip() + "\nPlain English summary:"
    )


def get_tokenized_splits(tokenizer, reports=None, max_input_length=512,
                         max_target_length=256):
    """Mask target padding with -100 for the seq2seq loss."""
    if reports is None:
        reports = load_reports()

    def tokenize(batch):
        inputs = tokenizer(
            [make_prompt(text) for text in batch["radiology_report"]],
            max_length=max_input_length, truncation=True, padding="max_length",
        )
        if all(not text for text in batch["layman_report"]):
            return inputs
        targets = tokenizer(
            text_target=batch["layman_report"], max_length=max_target_length,
            truncation=True, padding="max_length",
        )
        inputs["labels"] = [
            [token if token != tokenizer.pad_token_id else -100 for token in ids]
            for ids in targets["input_ids"]
        ]
        return inputs

    return DatasetDict({
        split: data.map(tokenize, batched=True, remove_columns=data.column_names)
        for split, data in reports.items()
    })
