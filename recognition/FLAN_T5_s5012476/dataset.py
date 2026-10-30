"""Dataset loading and inspection for radiology report summarisation."""

import logging

from utils import setup_logging
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq

logger = logging.getLogger(__name__)

DATASET_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"

MODEL_NAME = "google/flan-t5-base"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

PROMPT = (
    "Summarize the following radiology report in simple language "
    "that a patient without medical knowledge can understand:\n\n"
)

EXPECTED_SPLITS = {
    "train": 150454,
    "validation": 10000,
    "test": 10537,
}

REQUIRED_COLUMNS = {
    "source",
    "images_path",
    "radiology_report",
    "layman_report",
}


def load_radiology_dataset():
    """
    Load the official BioLaySumm2025 LaymanRRG dataset.

    The dataset contains radiology reports and their corresponding
    layperson-friendly summaries.

    Returns:
        DatasetDict: Official train, validation, and test splits.
    """
    return load_dataset(DATASET_NAME)

def load_tokenizer():
    """
    Load the tokenizer associated with the FLAN-T5 Base model.

    The same tokenizer is used for both full fine-tuning and LoRA
    to ensure consistent preprocessing across experiments.

    Returns:
        PreTrainedTokenizer: Tokenizer for FLAN-T5 Base.
    """
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def inspect_dataset(dataset):
    """Validate dataset structure and inspect sample records."""

    logger.info("Inspecting dataset structure")

    for split, expected_size in EXPECTED_SPLITS.items():
        if split not in dataset:
            raise ValueError(f"Missing dataset split: {split}")

        split_data = dataset[split]

        if len(split_data) != expected_size:
            raise ValueError(
                f"Unexpected size for {split}: "
                f"{len(split_data)} != {expected_size}"
            )

        missing_columns = REQUIRED_COLUMNS - set(split_data.column_names)

        if missing_columns:
            raise ValueError(
                f"Missing columns in {split}: {missing_columns}"
            )

        logger.info("%s: %s samples", split, f"{len(split_data):,}")
        logger.debug("Columns: %s", split_data.column_names)

        if len(split_data) > 0:
            sample = split_data[0]

            logger.debug(
                "Sample radiology report: %s",
                str(sample["radiology_report"])[:300],
            )

            logger.debug(
                "Sample layperson summary: %s",
                str(sample["layman_report"])[:300],
            )

    logger.info("Dataset inspection completed")

def preprocess_function(examples, tokenizer):
    """
    Convert radiology reports and reference summaries into token IDs.

    Each report is prefixed with a natural-language instruction to
    guide FLAN-T5 towards patient-friendly summarisation.

    Both input and target sequences are truncated to their configured
    maximum lengths. Padding is deferred to the DataLoader.

    Args:
        examples (dict): Batch containing radiology reports and summaries.
        tokenizer: FLAN-T5 tokenizer.

    Returns:
        dict: Tokenized input_ids, attention_mask, and labels.
    """

    inputs = [
        PROMPT + report
        for report in examples["radiology_report"]
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["layman_report"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def prepare_dataset(dataset, tokenizer, splits=("train", "validation")):
    """
    Tokenize the specified dataset splits.

    Uses batched mapping to preprocess multiple examples efficiently.
    Original text columns are removed from the tokenized dataset
    because the DataLoader only requires model input features.

    Args:
        dataset (DatasetDict): Original dataset.
        tokenizer: FLAN-T5 tokenizer.
        splits (tuple): Dataset splits to preprocess.

    Returns:
        dict: Mapping from split names to tokenized datasets.
    """

    tokenized_dataset = {}

    for split in splits:
        if split not in dataset:
            raise ValueError(f"Missing dataset split: {split}")

        tokenized_dataset[split] = dataset[split].map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
            remove_columns=dataset[split].column_names,
            desc=f"Tokenizing {split}",
        )

    return tokenized_dataset

def inspect_preprocessing(dataset, tokenizer):
    """Verify tokenization on three training examples."""

    sample = dataset["train"].select(range(3))

    examples = {
        "radiology_report": list(sample["radiology_report"]),
        "layman_report": list(sample["layman_report"]),
    }

    tokenized = preprocess_function(examples, tokenizer)

    for i in range(len(sample)):
        input_ids = tokenized["input_ids"][i]
        label_ids = tokenized["labels"][i]

        assert len(input_ids) <= MAX_INPUT_LENGTH
        assert len(label_ids) <= MAX_TARGET_LENGTH

        logger.debug(
            "Sample %d | Input tokens: %d | Target tokens: %d",
            i + 1,
            len(input_ids),
            len(label_ids),
        )

        logger.debug(
            "Decoded input: %s",
            tokenizer.decode(
                input_ids,
                skip_special_tokens=True,
            )[:400],
        )

        logger.debug(
            "Decoded target: %s",
            tokenizer.decode(
                label_ids,
                skip_special_tokens=True,
            )[:400],
        )

    logger.info("Preprocessing inspection completed")

def create_dataloader(dataset, tokenizer, batch_size=4, shuffle=False):
    """
    Create a PyTorch DataLoader with dynamic sequence padding.

    Input sequences are padded to the longest input in each batch.
    Target padding positions are assigned -100 so they are ignored
    when calculating the model's cross-entropy loss.

    Args:
        dataset: Tokenized Hugging Face dataset.
        tokenizer: FLAN-T5 tokenizer.
        batch_size (int): Number of examples per batch.
        shuffle (bool): Whether to shuffle examples between epochs.

    Returns:
        DataLoader: Batched model inputs and labels.
    """

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )

def inspect_dataloader(dataset, tokenizer):
    """Check dynamic padding and label masking on a small batch."""

    sample = dataset["train"].select(range(8))

    tokenized = sample.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=sample.column_names,
    )

    dataloader = create_dataloader(
        tokenized,
        tokenizer,
        batch_size=4,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch["input_ids"].shape[0] == 4
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape[0] == 4

    assert batch["input_ids"].shape[1] <= MAX_INPUT_LENGTH
    assert batch["labels"].shape[1] <= MAX_TARGET_LENGTH

    for key, value in batch.items():
        logger.debug("%s shape: %s", key, tuple(value.shape))

    logger.debug(
        "Ignored label padding tokens: %d",
        (batch["labels"] == -100).sum().item(),
    )

    logger.info("DataLoader inspection completed")

if __name__ == "__main__":
    setup_logging()

    dataset = load_radiology_dataset()
    inspect_dataset(dataset)

    tokenizer = load_tokenizer()
    inspect_preprocessing(dataset, tokenizer)
    inspect_dataloader(dataset, tokenizer)