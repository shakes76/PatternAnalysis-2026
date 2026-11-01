"""Loads and batches LaymanRRG reports for plain-language summarisation."""

from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding


class RadiologyData:
    """Prepares official LaymanRRG splits and creates padded batches.

    Call load() before requesting a dataloader.

    Attributes:
        tokenizer: Hugging Face tokenizer shared by all splits.
        config: Dict of dataset, tokenisation and batching settings.
    """

    DATASET_ID = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
    SOURCE_COLUMN = "radiology_report"
    TARGET_COLUMN = "layman_report"
    SMOKE_ROWS = {"train": 16, "validation": 8}

    def __init__(self, tokenizer, config):
        """Stores the tokenizer and configuration without loading data.

        Programmatic callers can set revision to a Hub branch, tag or commit.
        CLI/YAML configuration leaves it unset, using the Hub default revision.

        Args:
            tokenizer: A Hugging Face PreTrainedTokenizerBase instance.
            config: Dict containing dataset_id, smoke, batch_size,
                max_input_length, max_target_length and prompt_template
                (with a {report} placeholder). Optional keys are revision,
                eval_batch_size, max_train_examples and max_validation_examples.
        """
        self.tokenizer = tokenizer
        self.config = config

    def load(self):
        """Loads dataset rows and stores their tokenised forms.

        Smoke mode selects up to 16 training and 8 validation rows.
        Optional example limits also select prefixes of those splits. Both
        subset modes omit test data; full loading excludes test references from
        tokenisation. Split membership and row order are preserved.

        Stores selected rows in _raw_splits and tokens in _tokenised_splits.
        """
        from datasets import load_dataset

        self._raw_splits = load_dataset(
            self.config["dataset_id"],
            revision=self.config.get("revision"),
        )
        limits = self._row_limits()
        if limits is not None:
            # Select before tokenisation; never process test rows here.
            self._raw_splits = DatasetDict(
                {
                    split: _head(self._raw_splits[split], count)
                    for split, count in limits.items()
                }
            )

        self._tokenised_splits = DatasetDict()
        for split, rows in self._raw_splits.items():
            self._tokenised_splits[split] = rows.map(
                self._tokenise_batch,
                batched=True,
                remove_columns=rows.column_names,
                # Test references are withheld; never turn them into labels.
                fn_kwargs={"with_labels": split != "test"},
                desc=f"Tokenising {split}",
            )

    def _row_limits(self):
        """Returns train/validation limits, or None to keep all splits."""
        if self.config["smoke"]:
            return dict(self.SMOKE_ROWS)

        limits = {
            "train": self.config.get("max_train_examples"),
            "validation": self.config.get("max_validation_examples"),
        }
        return limits if any(n is not None for n in limits.values()) else None

    def _tokenise_batch(self, examples, with_labels=None):
        """Tokenises and truncates prompted reports and optional targets.

        Args:
            examples: Dict mapping column names to lists of strings. Requires
                radiology_report; layman_report is optional.
            with_labels: Whether to tokenise targets. None infers this from
                the presence of layman_report, preserving direct-call usage.

        Returns:
            A dict of unpadded input IDs and attention masks, plus labels when
            requested. Padding and -100 label masking happen later
            in the dataloader's collator.
        """
        prompts = [
            self.config["prompt_template"].format(report=report)
            for report in examples[self.SOURCE_COLUMN]
        ]
        encoded = self.tokenizer(
            prompts,
            max_length=self.config["max_input_length"],
            truncation=True,
            padding=False,
        )
        if with_labels is None:
            with_labels = self.TARGET_COLUMN in examples
        if with_labels:
            targets = self.tokenizer(
                text_target=examples[self.TARGET_COLUMN],
                max_length=self.config["max_target_length"],
                truncation=True,
                padding=False,
            )
            encoded["labels"] = targets["input_ids"]
        # Label padding is introduced and masked with -100 by the collator.
        return dict(encoded)

    def references(self, split):
        """Returns the full reference texts from a loaded split.

        Args:
            split: Name of a loaded dataset split.

        Returns:
            The reference column, without target-length truncation.
        """
        return self._raw_splits[split][self.TARGET_COLUMN]

    def num_examples(self, split):
        """Returns the number of selected rows in a loaded split.

        Args:
            split: Name of a loaded dataset split.

        Returns:
            The split's row count as an integer.
        """
        return len(self._raw_splits[split])

    def dataloader(self, split, shuffle=None):
        """Creates a padded DataLoader for a previously loaded split.

        Uses batch_size for training. Other splits use eval_batch_size when
        provided, falling back to batch_size.
        Label padding uses -100; test batches contain no labels.

        Args:
            split: Split name (str): "train", "validation" or "test".
            shuffle: Whether to shuffle rows (bool). None shuffles only train.

        Returns:
            A torch.utils.data.DataLoader yielding batches of PyTorch tensors.

        Raises:
            KeyError: If the requested split was not loaded.
        """
        if split == "test":
            # Keep test batches input-only, without even a labels=None field.
            collator = DataCollatorWithPadding(
                self.tokenizer,
                return_tensors="pt",
            )
        else:
            collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding=True,
                label_pad_token_id=-100,
                return_tensors="pt",
            )

        batch_size = self.config["batch_size"]
        if split != "train":
            batch_size = self.config.get("eval_batch_size") or batch_size

        return DataLoader(
            self._tokenised_splits[split],
            batch_size=batch_size,
            shuffle=(split == "train") if shuffle is None else shuffle,
            collate_fn=collator,
        )


def _head(rows, limit):
    """Returns up to limit leading rows, or all rows when limit is None."""
    return rows if limit is None else rows.select(range(min(limit, len(rows))))
