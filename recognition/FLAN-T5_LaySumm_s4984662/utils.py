"""
utils.py - evaluation helpers shared by train.py and predict.py: ROUGE, batched generation,
and resource profiling (peak VRAM, tokens/s, latency).

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
Course : COMP3710 Pattern Recognition, UQ, Semester 2 2026 - Project 2.6
         (FLAN-T5 lay summarisation of radiology reports, LoRA vs full fine-tuning)

Description
    The spec asks for ROUGE-1/2/L/Lsum plus peak VRAM and inference throughput for every
    configuration. The generation loop and the aggregation around the metric are written
    by hand (teaching staff: no HF Trainer); only the ROUGE formula itself comes from the
    `rouge_score` package (Google's reference implementation).

Key components
    split_sentences    - put one sentence per line (needed by ROUGE-Lsum)
    rouge_per_example  - ROUGE F-measure of every (prediction, reference) pair
    mean_rouge         - average x100 over all pairs or over a subset (an evaluation tier)
    compute_rouge      - the two above in one call
    generate_summaries - beam-search generation for a LaySummDataset + timing statistics
    reset_peak_vram / peak_vram_gib - GPU memory high-water mark

Definitions used for the resource table (written into the README as well)
    peak VRAM    torch.cuda.max_memory_allocated() / 2**30, in GiB (not GB).
    tokens/s     generated tokens (up to and including the first </s>) divided by the seconds
                 spent inside model.generate, with cuda synchronised around every call.
                 One batch is generated first and discarded (warm-up).
    latency      the same seconds divided by the number of reports (seconds per report,
                 amortised over a batch, so it depends on batch_size and num_beams).

Self-check (a CPU node on Rangpur is enough; loads ~1 GB of weights):
    python utils.py [--model_name google/flan-t5-base]

References
    ROUGE: Lin (2004), "ROUGE: A package for automatic evaluation of summaries".
    rouge_score: https://github.com/google-research/google-research/tree/master/rouge
    ROUGE-Lsum splits text into sentences on newlines (rouge_score documentation).

AI assistance: drafted with Claude (Anthropic) and reviewed by the author, see the README
section "Artificial Intelligence Usage Disclosure".
"""

import argparse
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from rouge_score import rouge_scorer

from dataset import MAX_TARGET_LEN, LaySummDataset, make_collate_fn

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")
_SCORER = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> str:
    """Return `text` with one sentence per line.

    rouge_score computes ROUGE-Lsum per line: a text without newlines is one single
    "sentence" and ROUGE-Lsum would silently equal ROUGE-L. The other ROUGE types ignore
    newlines, so the same string can be used for all four.
    """
    return "\n".join(s for s in _SENTENCE_END.split((text or "").strip()) if s)


def rouge_per_example(predictions: Sequence[str], references: Sequence[str]
                      ) -> List[Dict[str, float]]:
    """ROUGE F-measure (0..1) of each pair; keys are ROUGE_TYPES.

    Per-example scores let one generation pass be scored on every evaluation tier (by
    index) and let the autopsy pick high-ROUGE cases without re-scoring.
    """
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions but {len(references)} references")
    rows = []
    for prediction, reference in zip(predictions, references):
        # rouge_score takes the reference ("target") first and the prediction second.
        scores = _SCORER.score(split_sentences(reference), split_sentences(prediction))
        rows.append({name: scores[name].fmeasure for name in ROUGE_TYPES})
    return rows


def mean_rouge(per_example: Sequence[Dict[str, float]],
               indices: Optional[Sequence[int]] = None) -> Dict[str, float]:
    """Mean ROUGE x100 over all rows, or only over the rows listed in `indices`."""
    rows = list(per_example) if indices is None else [per_example[i] for i in indices]
    if not rows:
        raise ValueError("no examples to average")
    return {name: 100.0 * sum(r[name] for r in rows) / len(rows) for name in ROUGE_TYPES}


def compute_rouge(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    """Mean ROUGE-1/2/L/Lsum (x100) of the whole list."""
    return mean_rouge(rouge_per_example(predictions, references))


def reset_peak_vram() -> None:
    """Start a new peak-memory measurement (no-op without a GPU)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gib() -> Optional[float]:
    """Peak GPU memory allocated since the last reset, in GiB (None without a GPU)."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**30


def count_generated(sequences: torch.Tensor, eos_id: int) -> int:
    """Tokens generated in a batch of decoder outputs: from after the start token up to and
    including the first </s>; a sequence without </s> (cut by max_new_tokens) counts in full.
    Counting "not equal to the pad id" instead over-counts, because finished beams are not
    always padded with the pad id."""
    body = sequences[:, 1:]
    is_eos = body == eos_id
    length = torch.where(is_eos.any(dim=1), is_eos.int().argmax(dim=1) + 1, body.shape[1])
    return int(length.sum())


def _sync(device: torch.device) -> None:
    """Wait for queued GPU work so that wall-clock timing is meaningful."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def generate_summaries(model, tokenizer, dataset: LaySummDataset, batch_size: int = 32,
                       num_beams: int = 4, max_new_tokens: int = MAX_TARGET_LEN
                       ) -> Tuple[List[str], Dict[str, float]]:
    """Generate one summary per example of `dataset`; returns (summaries, statistics).

    Summaries are in dataset order. Internally, reports are processed from shortest to
    longest so that a batch contains similar lengths (less padding, faster); the same
    ordering is used for every configuration, so tokens/s stays comparable. The model is
    put in eval mode and restored afterwards, so this can be called between training
    epochs. bf16 autocast is used on GPU, as in training.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError("empty dataset")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    collate = make_collate_fn(tokenizer.pad_token_id)
    order = sorted(range(n), key=lambda i: len(dataset.reports[i]))
    batches = [order[i:i + batch_size] for i in range(0, n, batch_size)]

    def run(indices: List[int]) -> torch.Tensor:
        """Generate for the reports in `indices` (one batch) and return the token ids."""
        batch = collate([dataset[i] for i in indices])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return model.generate(input_ids=batch["input_ids"].to(device),
                                  attention_mask=batch["attention_mask"].to(device),
                                  num_beams=num_beams, max_new_tokens=max_new_tokens)

    run(batches[0])                                   # warm-up, result discarded
    reset_peak_vram()
    summaries: List[Optional[str]] = [None] * n
    seconds, generated = 0.0, 0
    for indices in batches:
        _sync(device)
        start = time.perf_counter()
        out = run(indices)
        _sync(device)
        seconds += time.perf_counter() - start
        generated += count_generated(out, tokenizer.eos_token_id)
        for i, text in zip(indices, tokenizer.batch_decode(out, skip_special_tokens=True)):
            summaries[i] = text.strip()
    model.train(was_training)
    stats = {"n_reports": n, "batch_size": batch_size, "num_beams": num_beams,
             "max_new_tokens": max_new_tokens, "generated_tokens": generated,
             "seconds": seconds, "tokens_per_s": generated / seconds,
             "seconds_per_report": seconds / n, "peak_vram_gib": peak_vram_gib()}
    return summaries, stats


# --------------------------------------------------------------------------------------
# Self-check. The ROUGE part needs no model; the generation part loads `model_name`.
# --------------------------------------------------------------------------------------
def _self_check(model_name: str) -> None:
    """Check ROUGE on hand-computed examples, then generation with `model_name`."""
    close = lambda a, b: abs(a - b) < 1e-9  # noqa: E731

    # 1. The dilemma of spec 2.6 in numbers: an inverted meaning still scores 0.80, a correct
    #    paraphrase scores 0.40 (ROUGE counts shared words, it does not read negation).
    reference = "there is no collapsed lung"
    wrong = rouge_per_example(["there is a collapsed lung"], [reference])[0]
    right = rouge_per_example(["the lung has not collapsed"], [reference])[0]
    assert close(wrong["rouge1"], 0.8) and close(wrong["rouge2"], 0.5), wrong
    assert close(wrong["rougeL"], 0.8), wrong
    assert close(right["rouge1"], 0.4) and close(right["rouge2"], 0.0), right
    assert close(rouge_per_example(["there is no collapsed lung"], [reference])[0]["rouge1"], 1.0)
    assert close(rouge_per_example([""], [reference])[0]["rouge1"], 0.0), "empty output scores 0"

    # 2. ROUGE-Lsum needs one sentence per line: with swapped sentences ROUGE-L is penalised
    #    (0.5) but summary-level ROUGE-Lsum is 1.0; newlines must not change the other types.
    pred, ref = "Delta epsilon zeta. Alpha beta gamma.", "Alpha beta gamma. Delta epsilon zeta."
    assert split_sentences(pred) == "Delta epsilon zeta.\nAlpha beta gamma."
    split = rouge_per_example([pred], [ref])[0]
    plain = _SCORER.score(ref, pred)
    assert close(split["rougeL"], plain["rougeL"].fmeasure) and close(split["rougeL"], 0.5), split
    assert close(split["rougeLsum"], 1.0), split
    assert close(plain["rougeLsum"].fmeasure, 0.5), "without newlines Lsum would equal L"

    # 3. Aggregation over tiers: mean x100, subset by index.
    rows = [{k: v for k, v in zip(ROUGE_TYPES, (1.0, 0.5, 0.5, 0.5))},
            {k: v for k, v in zip(ROUGE_TYPES, (0.0, 0.0, 0.0, 0.0))}]
    assert close(mean_rouge(rows)["rouge1"], 50.0) and close(mean_rouge(rows, [0])["rouge2"], 50.0)
    # Token counting: <s>=0 start, </s>=1; rows end with </s> then padding of either kind,
    # or never end (cut by max_new_tokens).
    rows = torch.tensor([[0, 5, 6, 1, 1, 1], [0, 7, 1, 0, 0, 0], [0, 8, 9, 10, 11, 12]])
    assert count_generated(rows, eos_id=1) == 3 + 2 + 5, count_generated(rows, eos_id=1)
    print("rouge checks passed")

    # 4. Generation: shape, order, and agreement between batch sizes (greedy, so that the
    #    comparison does not depend on beam bookkeeping).
    from modules import build_model, load_tokenizer

    tokenizer = load_tokenizer(model_name)
    reports = ["No pneumothorax.", "Mild bibasilar atelectasis, pneumonia cannot be excluded. "
               "No pleural effusion.", "Heart size is normal.", "Right lower lobe opacity. " * 6,
               "Within normal limits."]
    split_data = {"radiology_report": reports, "layman_report": ["x"] * len(reports)}
    dataset = LaySummDataset(split_data, tokenizer, with_targets=False)
    for mode in ("zeroshot", "lora"):
        model = build_model(mode, model_name)
        training_before = model.training
        texts1, _ = generate_summaries(model, tokenizer, dataset, batch_size=1, num_beams=1,
                                       max_new_tokens=16)
        texts3, stats = generate_summaries(model, tokenizer, dataset, batch_size=3, num_beams=1,
                                           max_new_tokens=16)
        assert len(texts3) == len(reports) and all(isinstance(t, str) for t in texts3)
        assert texts1 == texts3, "results must not depend on batching/ordering"
        assert stats["generated_tokens"] >= len(reports) and stats["tokens_per_s"] > 0
        assert stats["peak_vram_gib"] is None or stats["peak_vram_gib"] > 0
        assert model.training == training_before, "the train/eval mode must be restored"
        print(f"{mode}: {stats['tokens_per_s']:.1f} tokens/s, "
              f"{1000 * stats['seconds_per_report']:.0f} ms/report, example -> {texts3[1]!r}")
    model.train()
    generate_summaries(model, tokenizer, dataset, batch_size=3, num_beams=1, max_new_tokens=4)
    assert model.training, "generate_summaries must restore train mode"
    print("utils self-check passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-check for utils.py")
    parser.add_argument("--model_name", default="google/flan-t5-base")
    _self_check(parser.parse_args().model_name)
