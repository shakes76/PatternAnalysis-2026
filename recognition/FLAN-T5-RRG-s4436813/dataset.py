"""
COMP3710 Pattern Analysis - Final Assessment Project
Title: Clinical Radiology Report Summarisation with FLAN-T5 and Custom LoRA
Author: Aidan Fassy
Student ID: 44368133
File: dataset.py
Description: Data loading, task conditioning, SentencePiece tokenisation,
             loss-masking, and SHA-256 deduplication for LaymanRRG.
"""

import hashlib
import re
from functools import reduce
from typing import Dict, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import Dataset as HFDataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding

# ----------------- Constants -----------------

# Model parameters & Details
MODEL: str = "google/flan-t5-base"
DATASET_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
TASK_PREFIX: str = "Translate this radiology report into plain language " \
"for the patient: "

# Constants
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128
IGNORE_INDEX = -100

# String Constants
SOURCE_COL: str = "radiology_report"
TARGET_COL: str = "layman_report"

# ------------------ Helpers ------------------

def normalise_text(text: str) -> str:
    """
    Cleans text to normalise the input to the model.
    """
    # lower -> remove punc. -> remove spaces, newlines -> strip leading/trailing
    transforms: tuple[Callable, ...] = (
        str.lower, 
        lambda s: re.sub(r"[^\w\s]", "", s), # punctuation
        lambda s: re.sub(r"\s+", " ", s), # spaces and newlines
        str.strip 
    )
    return reduce(lambda acc, fn: fn(acc), transforms, text or "")

def hash_text(text: str) -> str:
    """
    Creates a text digest, used for fast deduplication.
    """
    # normalise -> encode -> hash -> digest
    transforms: tuple[Callable, ...] = (
        normalise_text,
        str.encode,
        hashlib.sha256,
        lambda h: h.hexdigest()
    )
    return reduce(lambda acc, fn: fn(acc), transforms, text or "")

def load_and_split_dataset(dataset: str = DATASET_NAME) \
      -> Tuple[HFDataset, HFDataset, HFDataset]:
    """
    Loads and splits the dataset.
    """
    # Splits
    raw_dataset: DatasetDict = load_dataset(DATASET_NAME)
    train_split: HFDataset = raw_dataset["train"]
    validation_split: HFDataset = raw_dataset["validation"]

    # Hashes for training purposes
    train_hashes: set[str] = {hash_text(report) for report 
                              in train_split[SOURCE_COL]}
    val_unseen: HFDataset = validation_split.filter(
        lambda row: hash_text(row[SOURCE_COL]) not in train_hashes
    )

    return train_split, validation_split, val_unseen

# ---------------- Classes ----------------

class LaymanRRGDataSet(Dataset):
    """Translates raw text to Torch Tensors."""

    def __init__(self, hf_dataset: HFDataset, tokeniser: PreTrainedTokenizerBase,
                 max_source_length: int = MAX_SOURCE_LENGTH,
                 max_target_length: int = MAX_TARGET_LENGTH):
        self._hf_dataset: HFDataset = hf_dataset
        self._tokeniser: PreTrainedTokenizerBase = tokeniser
        self._max_source_length: int = max_source_length
        self._max_target_length: int = max_target_length

    def __len__(self):
        return len(self._hf_dataset)

    def __getitem__(self, idx: int):
        """
        Returns usable tensors.
        """
        row = self._hf_dataset[idx]
        # Prepend the task prefix to the input
        source: str = TASK_PREFIX + row[SOURCE_COL]
        target: str = row[TARGET_COL]

        # Tokenise the input radiology report
        source_encoding: BatchEncoding = self._tokeniser(
            source,
            max_length=self._max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenise the summary 
        target_encoding: BatchEncoding = self._tokeniser(
            target,
            max_length=self._max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # [8, 1, 2] -> [8, 512] using squeeze(0)
        input_ids: torch.Tensor = source_encoding["input_ids"].squeeze(0)
        attention_mask: torch.Tensor = source_encoding["attention_mask"].squeeze(0)
        labels: torch.Tensor = target_encoding["input_ids"].squeeze(0)
        # Sets the padding to the predetermined ignore index
        labels[labels == self._tokeniser.pad_token_id] = IGNORE_INDEX

        return {"input_ids": input_ids, "attention_mask": attention_mask,
                 "labels": labels}

# -------------- Loading & Splits --------------

def create_dataloaders(B: int = 8, distributed: bool = False,
                       num_workers: int = 2, model: str = MODEL,
                       dataset: str = DATASET_NAME) -> \
                       Tuple[DataLoader, DataLoader,
                              DataLoader, PreTrainedTokenizerBase]:
    """
    Creates PyTorch DataLoaders. Outputs train, val_full,
    val_unseen splits.
    """
    tokeniser: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model)
    train_hf, val_full_hf, val_unseen_hf = load_and_split_dataset(dataset)

    train_ds: LaymanRRGDataSet = LaymanRRGDataSet(train_hf, tokeniser)
    val_full_ds: LaymanRRGDataSet = LaymanRRGDataSet(val_full_hf, tokeniser)
    val_unseen_ds: LaymanRRGDataSet = LaymanRRGDataSet(val_unseen_hf, tokeniser)

    train_sampler: Optional[DistributedSampler] = (
        DistributedSampler(train_ds, shuffle=True) if distributed else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=B,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_full_loader = DataLoader(
        val_full_ds,
        batch_size=B,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_unseen_loader = DataLoader(
        val_unseen_ds,
        batch_size=B,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_full_loader, val_unseen_loader, tokeniser