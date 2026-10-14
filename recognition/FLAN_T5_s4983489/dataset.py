"""Load and tokenize LaymanRRG data for FLAN-T5 training.

Tokenization and batch padding reference: https://huggingface.co/docs/transformers/tasks/summarization
"""

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq


DATASET_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
TASK_PREFIX = (
    "Explain this radiology report in plain English for a patient. "
    "Preserve the findings, negation, and uncertainty:\n"
)


def load_data():
    """Load the dataset with its official train, validation, and test splits."""
    return load_dataset(DATASET_NAME)


def tokenize_data(data, tokenizer, max_input_length=512, max_target_length=512):
    """Tokenize one split into input_ids, attention_mask, and labels."""

    def tokenize_batch(batch):
        # The encoder receives only the task prompt and report, not the reference.
        inputs = [TASK_PREFIX + report for report in batch["radiology_report"]]
        encoded = tokenizer(
            inputs, max_length=max_input_length, truncation=True
        )

        # Tokenize reference explanations separately as decoder targets.
        targets = tokenizer(
            text_target=batch["layman_report"],
            max_length=max_target_length,
            truncation=True,
        )
        encoded["labels"] = targets["input_ids"]
        return encoded

    # Pad at batch time so short texts do not each occupy 512 positions.
    # map returns new data, leaving the original full texts available for evaluation.
    return data.map(
        tokenize_batch, batched=True, remove_columns=data.column_names
    )


if __name__ == "__main__":
    data = load_data()

    # Print split sizes to confirm that the dataset loaded.
    for split_name, split in data.items():
        print(f"{split_name}: {len(split)} samples")

    # Each record pairs a radiology report with its layperson explanation.
    example = data["train"][0]
    print("\nRadiology report (input):")
    print(example["radiology_report"])
    print("\nLayperson report (target):")
    print(example["layman_report"])

    # Load only the tokenizer to demonstrate a batch of two real examples.
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    examples = tokenize_data(data["train"].select(range(2)), tokenizer)

    # Pad inputs with the pad token and targets with -100 to exclude padding from loss.
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)
    batch = collator([examples[0], examples[1]])
    print("\nTokenized batch:")
    for name, tensor in batch.items():
        print(f"{name}: {tuple(tensor.shape)}")
    print(f"Ignored target padding positions: {(batch['labels'] == -100).sum().item()}")
