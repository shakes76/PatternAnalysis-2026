"""
predict.py - example usage of a trained model: load the saved weights, summarise example
radiology reports, print the results and draw a comparison figure.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
Course : COMP3710 Pattern Recognition, UQ, Semester 2 2026 - Project 2.6
         (FLAN-T5 lay summarisation of radiology reports, LoRA vs full fine-tuning)

Description
    Takes a few example reports, generates the lay summary with the chosen model and shows
    SOURCE / OUTPUT / REFERENCE side by side with the ROUGE scores of the output.
      * validation examples come with a reference summary, so ROUGE can be printed;
      * test examples are inputs only (the test references are withheld), and a report typed
        on the command line has no reference either: they show source and output only.
    Examples are drawn with a fixed seed, not hand-picked, so the scores are not cherry-picked.
    Validation rows whose BIMCV session id repeats (their report/summary pairs are often
    mismatched, see dataset.py) are left out of the draw.
    Decoding is the same as in evaluation (beam search, utils.generate_summaries).

    Without --checkpoint the untuned zero-shot FLAN-T5 is used, so the script always runs;
    it then prints how to create a trained checkpoint. The checkpoint type (LoRA adapter or
    full model) is detected from its files. Weights are not part of the repository; they are
    produced by train.py.

Usage
    python predict.py                                         # zero-shot baseline
    python predict.py --checkpoint runs/lora_r8_lr0.0003/best # a model trained with train.py
    python predict.py --checkpoint runs/full_lr5e-05/best --n_val 6 --n_test 3
    python predict.py --report "Mild bibasilar atelectasis. No pneumothorax."
    Output: the examples on screen and <output_dir>/predictions.png (default runs/predict).
    Run it on a GPU node for speed (slurm/train.sh is for training; for a few examples a
    CPU is also fine).

References
    Same model, decoding and metrics as train.py / utils.py; examples drawn from the
    BioLaySumm 2025 LaymanRRG open-source track (see dataset.py).

AI assistance: drafted with Claude (Anthropic) and reviewed by the author, see the README
section "Artificial Intelligence Usage Disclosure".
"""

import argparse
import json
import os
import random
import sys
import textwrap
from typing import Dict, List, Optional

import numpy as np
import torch

from dataset import (SOURCE_COLUMN, TARGET_COLUMN, LaySummDataset, load_laysumm,
                     session_duplicate_rows)
from modules import MODEL_NAME, build_model, load_checkpoint, load_tokenizer
from utils import ROUGE_TYPES, generate_summaries, rouge_per_example

TRAIN_HINT = """To create a trained checkpoint (weights are not stored in the repository):
    python train.py --mode lora --lora_r 8        # or --mode full
    (on Rangpur: sbatch slurm/train.sh --mode lora --lora_r 8)
Then run:  python predict.py --checkpoint runs/<run name>/best"""


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line options (see the module docstring for usage)."""
    p = argparse.ArgumentParser(description="Summarise example radiology reports")
    p.add_argument("--checkpoint", default=None, help="folder 'best' written by train.py")
    p.add_argument("--model_name", default=MODEL_NAME, help="base model (LoRA needs it)")
    p.add_argument("--n_val", type=int, default=4, help="validation examples (with reference)")
    p.add_argument("--n_test", type=int, default=2, help="test examples (input only)")
    p.add_argument("--report", default=None, help="summarise this text instead of the examples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_dir", default=os.path.join("runs", "predict"))
    return p.parse_args(argv)


def load_model(args, device: torch.device):
    """Zero-shot model, or the LoRA / full checkpoint in --checkpoint (type auto-detected)."""
    if args.checkpoint is None:
        print("No checkpoint was given, so the untuned zero-shot FLAN-T5 is used.\n" + TRAIN_HINT,
              flush=True)
        return build_model("zeroshot", args.model_name).to(device), "zero-shot"
    if not os.path.isdir(args.checkpoint):
        sys.exit(f"checkpoint folder not found: {args.checkpoint}\n{TRAIN_HINT}")
    if os.path.exists(os.path.join(args.checkpoint, "adapter_config.json")):
        mode = "lora"
    elif os.path.exists(os.path.join(args.checkpoint, "config.json")):
        mode = "full"
    else:
        sys.exit(f"{args.checkpoint} holds neither a LoRA adapter nor a full model.\n{TRAIN_HINT}")
    model = load_checkpoint(mode, args.checkpoint, args.model_name).to(device)
    return model, f"{mode} ({args.checkpoint})"


def pick_examples(args) -> List[Dict[str, Optional[str]]]:
    """A typed report, or random validation (with reference) and test (input only) rows."""
    if args.report:
        return [{"split": "custom input", "source": args.report, "reference": None}]
    splits = load_laysumm()
    rng = random.Random(args.seed)
    skip = set(session_duplicate_rows(splits)["validation"])
    val_pool = [i for i in range(len(splits["validation"])) if i not in skip]
    examples = []
    for split, count, pool in (("validation", args.n_val, val_pool),
                               ("test", args.n_test, list(range(len(splits["test"]))))):
        for i in rng.sample(pool, min(count, len(pool))):
            row = splits[split][i]
            examples.append({"split": f"{split} #{i}", "source": row[SOURCE_COLUMN],
                             "reference": row[TARGET_COLUMN] if split == "validation" else None})
    return examples


def plot_examples(examples: List[Dict[str, object]], title: str, path: str) -> None:
    """One row per example: source / output / reference as text, ROUGE of the output as bars."""
    import matplotlib
    matplotlib.use("Agg")                       # no display on compute nodes
    import matplotlib.pyplot as plt

    bodies = []
    for ex in examples:
        parts = [("SOURCE", ex["source"]), ("OUTPUT", ex["output"])]
        if ex["reference"]:
            parts.append(("REFERENCE", ex["reference"]))
        bodies.append(f"[{ex['split']}]\n" + "\n".join(
            textwrap.fill(f"{name}: " + textwrap.shorten(str(text), 330), 112)
            for name, text in parts))
    heights = [max(body.count("\n") + 2, 12) for body in bodies]   # text lines, at least 12
    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 0.16 * sum(heights) + 0.8),
                             squeeze=False, gridspec_kw={"width_ratios": [3.4, 1],
                                                         "height_ratios": heights})
    fig.suptitle(textwrap.fill(title, 110), fontsize=10)
    labels = ["R-1", "R-2", "R-L", "R-Lsum"]
    for (text_ax, bar_ax), ex, body in zip(axes, examples, bodies):
        text_ax.axis("off")
        text_ax.text(0, 1, body, va="top", ha="left", fontsize=7.5, family="monospace",
                     transform=text_ax.transAxes)
        if ex["rouge"] is None:
            bar_ax.axis("off")
            bar_ax.text(0.5, 0.5, "no reference\n(blind test / typed input)", ha="center",
                        va="center", fontsize=8, transform=bar_ax.transAxes)
            continue
        scores = np.array([100 * ex["rouge"][name] for name in ROUGE_TYPES])
        bars = bar_ax.bar(np.arange(len(scores)), scores, color="tab:blue")
        bar_ax.bar_label(bars, fmt="%.0f", fontsize=7)
        bar_ax.set(ylim=(0, 105), xticks=np.arange(len(labels)), xticklabels=labels,
                   ylabel="ROUGE F1 x100")
        bar_ax.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    """Load the model, summarise the examples, print the results and save the figure."""
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, description = load_model(args, device)
    tokenizer = load_tokenizer(args.model_name)
    examples = pick_examples(args)

    sources = [ex["source"] for ex in examples]
    dataset = LaySummDataset({SOURCE_COLUMN: sources, TARGET_COLUMN: [""] * len(sources)},
                             tokenizer, with_targets=False)
    outputs, stats = generate_summaries(model, tokenizer, dataset, args.batch_size,
                                        args.num_beams, args.max_new_tokens)
    with_ref = [i for i, ex in enumerate(examples) if ex["reference"]]
    scores = rouge_per_example([outputs[i] for i in with_ref],
                               [examples[i]["reference"] for i in with_ref])
    for ex, output in zip(examples, outputs):
        ex["output"], ex["rouge"] = output, None
    for i, score in zip(with_ref, scores):
        examples[i]["rouge"] = score

    for ex in examples:
        print(f"\n=== {ex['split']} ===\nSOURCE   : {ex['source']}\nOUTPUT   : {ex['output']}")
        if ex["reference"]:
            print(f"REFERENCE: {ex['reference']}\nROUGE x100: "
                  + " ".join(f"{k} {100 * ex['rouge'][k]:.1f}" for k in ROUGE_TYPES))
    print(f"\nmodel: {description} | {stats['n_reports']} reports | {stats['seconds']:.1f} s in "
          f"generate | {stats['tokens_per_s']:.0f} tokens/s | beams {args.num_beams}")

    os.makedirs(args.output_dir, exist_ok=True)
    figure = os.path.join(args.output_dir, "predictions.png")
    plot_examples(examples, f"FLAN-T5 lay summaries: {description}", figure)
    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        json.dump(examples, f, indent=2)
    print("wrote", figure)


if __name__ == "__main__":
    main()
