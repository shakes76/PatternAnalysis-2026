"""
dataset.py - data loading and preprocessing for BioLaySumm 2025 LaymanRRG (open-source track).

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
Course : COMP3710 Pattern Recognition, UQ, Semester 2 2026 - Project 2.6
         (FLAN-T5 lay summarisation of radiology reports, LoRA vs full fine-tuning)

Description
    Turns (radiology report, lay summary) pairs into the tensors FLAN-T5 needs:
      1. a task prompt is prepended to the report (FLAN-T5 is instruction-tuned);
      2. the prompted report and the lay summary are tokenised and truncated;
      3. when a batch is padded, padded *label* positions are set to -100 so the loss
         ignores them (target sequence masking).
    It also defines the evaluation subsets and the development split (see "Data leakage").

Data leakage (measured, see README): the official splits are kept, but 45.1% of validation
reports also occur verbatim in train and 39.0% of validation rows share a study/session id
with train. Reporting one ROUGE number on the whole validation set would therefore mix
memorisation with real summarisation, so ROUGE is reported on three nested tiers:
    full           all 10,000 validation rows (main, reproducible number)
    unseen_report  report text not present in train (5,494 rows)
    clean          unseen report AND study/session not present in train (4,037 rows)
Hyper-parameters and checkpoints are chosen on a small development split cut from TRAIN
(whole studies, so no sibling image stays in training), keeping validation untouched.

Key components
    load_laysumm       - load the three official splits from the Hugging Face Hub
    build_source_text  - prepend the task prompt to a radiology report
    LaySummDataset     - tokenise one example (truncation) as lists of token ids
    make_collate_fn    - pad a batch; padded labels become -100
    normalise_report / study_key - the keys used to detect overlap between splits
    evaluation_tiers   - validation row indices of the full / unseen_report / clean tiers
    make_dev_split     - (train_indices, dev_indices) cut from train by whole study
    session_duplicate_rows - BIMCV rows whose session id occurs more than once

Self-check (needs `transformers`, run on Rangpur; downloads nothing if the cache is warm):
    python dataset.py            # synthetic checks
    python dataset.py --real     # also loads LaymanRRG, checks tiers (10,000 / 5,494 / 4,037)

References
    BioLaySumm 2025 LaymanRRG: https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track
    T5 (Raffel et al., 2020) and FLAN-T5 (Chung et al., 2024), spec references [19], [20].
    Hugging Face T5 documentation: label positions equal to -100 are ignored by the loss.

AI assistance: drafted with Claude (Anthropic) and reviewed by the author, see the README
section "Artificial Intelligence Usage Disclosure".
"""

import argparse
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset

DATASET_ID = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
SOURCE_COLUMN = "radiology_report"
TARGET_COLUMN = "layman_report"
IMAGE_COLUMN = "images_path"   # BIMCV: session id "ses-XXXXX"; others: "<study-uid>_<code>.png"
ORIGIN_COLUMN = "source"       # PadChest / BIMCV-COVID19 / OpenI
BIMCV = "BIMCV-COVID19"

# Truncation lengths come from the data audit (feasibility review, section 3): with the
# FLAN-T5 tokenizer only 0.19% of train reports and 0.24% of train summaries exceed 512
# tokens, so truncating at 512 affects at most ~0.4% of rows.
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 512

# PyTorch's CrossEntropyLoss (and Hugging Face T5's loss) skips targets equal to -100.
IGNORE_INDEX = -100

# The prompt goes BEFORE the report so truncation can only cut the end of the report,
# never the instruction.
TASK_PROMPT = ("Rewrite the following radiology report in plain language that a patient can "
               "understand:")

# Equivalent instructions used as the only data augmentation (training only). Changing words
# inside a clinical report could flip a negation or a finding, so the report text is never
# touched. TASK_PROMPT (variant 0) is the fixed prompt used for evaluation.
TASK_PROMPT_VARIANTS = (
    TASK_PROMPT,
    "Explain the following radiology report in simple words for a patient:",
    "Translate this radiology report into plain, easy-to-understand language for a patient:",
    "Summarize the following radiology report in plain language for a layperson:",
)


def load_laysumm(cache_dir: Optional[str] = None) -> DatasetDict:
    """Load the official train / validation / test splits (no re-splitting, no shuffling).

    The test split has an empty `layman_report` for every row (blind shared-task test
    set), so only train and validation can be used to compute ROUGE.
    """
    return load_dataset(DATASET_ID, cache_dir=cache_dir)


def build_source_text(report: str, prompt: str = TASK_PROMPT) -> str:
    """Return the model input: the task prompt followed by the radiology report."""
    return f"{prompt} {report.strip()}"


class LaySummDataset(Dataset):
    """One split of LaymanRRG as tokenised (input_ids, attention_mask[, labels]) examples.

    Tokenisation is done on the fly in `__getitem__` so no tokenised copy of the 150k
    training pairs is kept in memory. Items are plain lists of ints; padding to a common
    length is done per batch by `make_collate_fn`.

    Args:
        split: a Hugging Face split (or any mapping of column name -> list of str).
        tokenizer: the FLAN-T5 tokenizer.
        prompt: task prompt prepended to every report.
        max_source_len / max_target_len: truncation lengths (in tokens).
        with_targets: False for the test split, which has no reference summaries.
        augment: training only. Cycle through `prompt_variants` instead of using `prompt`;
            variant = (index + epoch) % len(variants), so it is deterministic. Call
            `set_epoch` before creating each epoch's DataLoader iterator.
        prompt_variants: the equivalent task prompts used when `augment` is True.
    """

    def __init__(self, split, tokenizer, prompt: str = TASK_PROMPT,
                 max_source_len: int = MAX_SOURCE_LEN, max_target_len: int = MAX_TARGET_LEN,
                 with_targets: bool = True, augment: bool = False,
                 prompt_variants: Sequence[str] = TASK_PROMPT_VARIANTS) -> None:
        """See the class docstring for the arguments."""
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.augment = augment
        self.prompt_variants = tuple(prompt_variants)
        self.epoch = 0
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        # list(...) materialises the column: datasets>=5 returns lazy Column objects.
        self.reports: List[str] = [r or "" for r in list(split[SOURCE_COLUMN])]
        self.targets: Optional[List[str]] = None
        if with_targets:
            self.targets = [t or "" for t in list(split[TARGET_COLUMN])]
            if not any(t.strip() for t in self.targets):
                raise ValueError("every reference summary is empty (this is the blind test "
                                 "split); build the dataset with with_targets=False")

    def __len__(self) -> int:
        """Number of reports."""
        return len(self.reports)

    def set_epoch(self, epoch: int) -> None:
        """Shift the prompt variant assignment so each report sees different phrasings."""
        self.epoch = epoch

    def prompt_for(self, index: int) -> str:
        """Task prompt used for example `index` (the fixed prompt unless augmenting)."""
        if self.augment:
            return self.prompt_variants[(index + self.epoch) % len(self.prompt_variants)]
        return self.prompt

    def __getitem__(self, index: int) -> Dict[str, object]:
        """Tokenise report `index` (and its summary when the dataset has targets)."""
        source = build_source_text(self.reports[index], self.prompt_for(index))
        # truncation=True cuts the text but the tokenizer still appends </s> afterwards.
        encoded = self.tokenizer(source, max_length=self.max_source_len, truncation=True)
        item: Dict[str, object] = {"input_ids": encoded["input_ids"],
                                   "attention_mask": encoded["attention_mask"],
                                   "index": index}
        if self.targets is not None:
            # The </s> token stays in the labels (it is NOT masked): the model must learn
            # to emit it, otherwise generation would never stop on its own.
            target = self.tokenizer(self.targets[index], max_length=self.max_target_len,
                                    truncation=True)
            item["labels"] = target["input_ids"]
        return item


def make_collate_fn(pad_token_id: int):
    """Build a DataLoader `collate_fn` that pads each batch to its longest sequence.

    Inputs are padded with the tokenizer's pad id (and attention_mask 0). Labels are
    padded with -100 instead of the pad id, so padded positions contribute nothing to the
    loss. Hugging Face T5 then derives the decoder inputs by shifting the labels right
    and replacing -100 with the pad id, so no extra step is needed.
    """

    def pad(sequences: List[List[int]], value: int) -> torch.Tensor:
        """Right-pad every sequence to the longest one in the batch with `value`."""
        width = max(len(seq) for seq in sequences)
        return torch.tensor([seq + [value] * (width - len(seq)) for seq in sequences],
                            dtype=torch.long)

    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        """Turn a list of items into one batch of tensors."""
        out = {"input_ids": pad([b["input_ids"] for b in batch], pad_token_id),
               "attention_mask": pad([b["attention_mask"] for b in batch], 0),
               "index": torch.tensor([b["index"] for b in batch], dtype=torch.long)}
        if "labels" in batch[0]:
            out["labels"] = pad([b["labels"] for b in batch], IGNORE_INDEX)
        return out

    return collate


def normalise_report(text: Optional[str]) -> str:
    """Lower-case and collapse whitespace so trivially different copies of a report match."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def study_key(images_path: str) -> str:
    """Id shared by all images of one study (BIMCV: the session id).

    `images_path` looks like "<study-uid>_<image-code>.png"; images of the same study share
    the part before "_" and one report. The dataset has no patient id, so this is a strong
    proxy for "same exam", not proof of the same patient.
    """
    return images_path.rsplit(".", 1)[0].split("_")[0]


def evaluation_tiers(train, validation) -> Dict[str, List[int]]:
    """Validation row indices of the three nested evaluation tiers (see module docstring)."""
    seen_reports = {normalise_report(r) for r in list(train[SOURCE_COLUMN])}
    seen_studies = {study_key(p) for p in list(train[IMAGE_COLUMN])}
    reports = [normalise_report(r) for r in list(validation[SOURCE_COLUMN])]
    studies = [study_key(p) for p in list(validation[IMAGE_COLUMN])]
    unseen = [i for i, r in enumerate(reports) if r not in seen_reports]
    clean = [i for i in unseen if studies[i] not in seen_studies]
    return {"full": list(range(len(reports))), "unseen_report": unseen, "clean": clean}


def make_dev_split(train, size: int = 1000, seed: int = 0) -> Tuple[List[int], List[int]]:
    """Cut a small development set out of TRAIN; returns (train_indices, dev_indices).

    Used to choose learning rate / checkpoint so that validation stays untouched. Whole
    studies are moved together (otherwise sibling images with a near-identical report would
    stay in training and make the dev score optimistic), and only studies whose report
    texts occur nowhere else in train are eligible. BIMCV sessions with two rows are also
    skipped (their report/summary pairs are often mismatched, see `session_duplicate_rows`).
    Deterministic for a given seed. The dev split is therefore comparable to the `clean`
    validation tier, not to the full one.
    """
    images = list(train[IMAGE_COLUMN])
    origins = list(train[ORIGIN_COLUMN])
    reports = [normalise_report(r) for r in list(train[SOURCE_COLUMN])]
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, path in enumerate(images):
        groups[study_key(path)].append(i)
    total = Counter(reports)
    eligible = sorted(
        key for key, members in groups.items()
        if all(total[r] == c for r, c in Counter(reports[i] for i in members).items())
        and not (len(members) > 1 and origins[members[0]] == BIMCV))
    random.Random(seed).shuffle(eligible)
    dev: List[int] = []
    for key in eligible:
        if len(dev) >= size:
            break
        dev.extend(groups[key])
    dev_set = set(dev)
    return [i for i in range(len(reports)) if i not in dev_set], sorted(dev)


def session_duplicate_rows(splits) -> Dict[str, List[int]]:
    """Row indices (per split) of BIMCV sessions that occur in more than one row.

    In the data every such session repeats the same report text in both rows while the lay
    summaries differ, so one of the two (report, summary) pairs is often mismatched. These
    rows are excluded from the clinical-safety autopsy pool (data errors, not model errors).
    """
    sessions = {name: list(zip(list(part[IMAGE_COLUMN]), list(part[ORIGIN_COLUMN])))
                for name, part in splits.items()}      # (images_path, source) per row
    counts = Counter(p for rows in sessions.values() for p, o in rows if o == BIMCV)
    return {name: [i for i, (p, o) in enumerate(rows) if o == BIMCV and counts[p] > 1]
            for name, rows in sessions.items()}


# --------------------------------------------------------------------------------------
# Self-check. Run on Rangpur; it only needs the tokenizer (a few MB, no model weights).
# --------------------------------------------------------------------------------------
def _self_check(model_name: str, real: bool) -> None:
    """Synthetic checks of the data pipeline; with `real`, also the real splits."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    prompt_ids = tokenizer(TASK_PROMPT)["input_ids"][:-1]  # drop </s>

    split = {SOURCE_COLUMN: ["No pneumothorax.", "Mild bibasilar atelectasis. " * 4,
                             "pneumothorax " * 3000],
             TARGET_COLUMN: ["There is no collapsed lung.", "Small areas at the bottom of "
                             "both lungs are not fully open.", "x"]}
    dataset = LaySummDataset(split, tokenizer)
    items = [dataset[i] for i in range(len(dataset))]

    for item in items:  # prompt first, length cap, </s> kept
        assert item["input_ids"][:len(prompt_ids)] == prompt_ids, "prompt must come first"
        assert len(item["input_ids"]) <= MAX_SOURCE_LEN and item["input_ids"][-1] == eos
        assert item["labels"][-1] == eos
    assert len(items[2]["input_ids"]) == MAX_SOURCE_LEN, "long report must be truncated"

    batch = make_collate_fn(pad)(items)
    padding = batch["attention_mask"] == 0
    assert (batch["input_ids"][padding] == pad).all(), "inputs padded with the pad id"
    label_padding = batch["labels"] == IGNORE_INDEX
    expected = sum(batch["labels"].shape[1] - len(it["labels"]) for it in items)
    assert int(label_padding.sum()) == expected, "only padded label positions are -100"
    assert not (batch["labels"] == pad).any(), "no pad id may leak into labels"

    assert "labels" not in LaySummDataset(split, tokenizer, with_targets=False)[0]
    try:
        LaySummDataset({SOURCE_COLUMN: ["a"], TARGET_COLUMN: [""]}, tokenizer)
    except ValueError:
        pass
    else:
        raise AssertionError("an all-empty reference column must be rejected")
    # augmentation: deterministic cycling over equivalent prompts, evaluation prompt fixed
    aug = LaySummDataset(split, tokenizer, augment=True)
    assert aug.prompt_for(0) == TASK_PROMPT_VARIANTS[0]
    aug.set_epoch(1)
    assert aug.prompt_for(0) == TASK_PROMPT_VARIANTS[1]
    variant_ids = tokenizer(TASK_PROMPT_VARIANTS[1])["input_ids"][:-1]
    assert aug[0]["input_ids"][:len(variant_ids)] == variant_ids
    assert dataset.prompt_for(0) == TASK_PROMPT, "evaluation prompt must stay fixed"

    # overlap keys and tiers
    assert normalise_report("  No   Pneumothorax. ") == "no pneumothorax."
    assert study_key("2168_03-150-151.png") == "2168" and study_key("ses-E26946") == "ses-E26946"
    tiny_train = {SOURCE_COLUMN: ["Report A.", "Report B."], IMAGE_COLUMN: ["s1_x.png", "s2_y.png"]}
    tiny_val = {SOURCE_COLUMN: ["report a.", "Report C.", "Report D."],
                IMAGE_COLUMN: ["s9_q.png", "s1_z.png", "s7_w.png"]}
    tiers = evaluation_tiers(tiny_train, tiny_val)
    assert tiers == {"full": [0, 1, 2], "unseen_report": [1, 2], "clean": [2]}, tiers

    # dev split: whole studies, unique report texts only, deterministic, nothing left over
    tiny = {SOURCE_COLUMN: ["Unique one.", "unique   one.", "Common.", "Common.", "Unique two."],
            IMAGE_COLUMN: ["g1_a.png", "g1_b.png", "g2_a.png", "g3_a.png", "g4_a.png"],
            ORIGIN_COLUMN: ["PadChest"] * 5}
    rest, dev = make_dev_split(tiny, size=1, seed=0)
    assert dev and sorted(rest + dev) == list(range(5)) and not set(rest) & set(dev)
    assert (rest, dev) == make_dev_split(tiny, size=1, seed=0), "must be deterministic"
    assert not {study_key(tiny[IMAGE_COLUMN][i]) for i in dev} & \
        {study_key(tiny[IMAGE_COLUMN][i]) for i in rest}, "no study may straddle train and dev"
    assert all(tiny[SOURCE_COLUMN][i].lower() != "common." for i in dev)
    print("synthetic checks passed")

    if real:
        splits = load_laysumm()
        print({name: len(part) for name, part in splits.items()})
        tiers = evaluation_tiers(splits["train"], splits["validation"])
        sizes = {name: len(idx) for name, idx in tiers.items()}
        print("evaluation tiers:", sizes)
        assert sizes == {"full": 10000, "unseen_report": 5494, "clean": 4037}, sizes
        rest, dev = make_dev_split(splits["train"])
        train_studies = [study_key(p) for p in list(splits["train"][IMAGE_COLUMN])]
        dev_studies = {train_studies[i] for i in dev}
        assert len(rest) + len(dev) == len(train_studies) and len(dev) >= 1000
        assert not dev_studies & {train_studies[i] for i in rest}
        print(f"dev split: {len(dev)} rows ({len(dev_studies)} studies); train kept: {len(rest)}")
        dup = {name: len(idx) for name, idx in session_duplicate_rows(splits).items()}
        print("BIMCV duplicate-session rows:", dup)
        assert dup == {"train": 2226, "validation": 146, "test": 30}, dup
        validation = LaySummDataset(splits["validation"].select(range(1000)), tokenizer)
        truncated = sum(len(validation[i]["input_ids"]) == MAX_SOURCE_LEN for i in range(1000))
        print(f"validation rows truncated at {MAX_SOURCE_LEN} tokens (first 1,000): {truncated}")
        test = LaySummDataset(splits["test"].select(range(8)), tokenizer, with_targets=False)
        assert "labels" not in test[0]
        print("real-data checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-check for dataset.py")
    parser.add_argument("--model_name", default="google/flan-t5-base")
    parser.add_argument("--real", action="store_true", help="also load LaymanRRG")
    args = parser.parse_args()
    _self_check(args.model_name, args.real)
