"""Shared evaluation for untuned, full and LoRA FLAN-T5 models.

ROUGE is calculated by the rouge-score library, not reimplemented here:
https://github.com/google-research/google-research/tree/master/rouge
Metric reference: Lin (2004), https://aclanthology.org/W04-1013/.
Generation uses Transformers/PEFT APIs referenced in modules.py and README.
"""

from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import MODEL_ID, INSTRUCTION
from dataset import load_audited_data, split_train_development
from modules import RadiologySummarizer

import nltk
import numpy as np
from rouge_score import rouge_scorer

import argparse
import hashlib
import json


def generate_summaries(
    model_path, reports, device, max_new_tokens, progress_path
):
    generation_options = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 0,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
    }

    reports_hash = hashlib.sha256(
        json.dumps(list(reports), ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    expected_settings = {
        "progress_version": 2,
        "model_path": str(model_path),
        "instruction": INSTRUCTION,
        "reports_hash": reports_hash,
        "generation_options": generation_options,
    }

    summaries = []
    diagnostics = []

    if progress_path.exists():
        with progress_path.open() as file:
            saved = json.load(file)

        if saved["settings"] != expected_settings:
            raise ValueError("Saved generation settings do not match this run")

        summaries = saved["summaries"]
        diagnostics = saved["diagnostics"]
        if not isinstance(diagnostics, list) or len(diagnostics) != len(summaries):
            raise ValueError("Saved diagnostics do not match saved summaries")
        if not isinstance(summaries, list) or not all(
            isinstance(summary, str) for summary in summaries
        ):
            raise ValueError("Saved summaries must be a list of strings")
        if len(summaries) > len(reports):
            raise ValueError("Saved progress contains too many summaries")

        print(
            f"{model_path}: recovered {len(summaries)} summaries",
            flush=True,
        )

    if len(summaries) == len(reports):
        return summaries

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RadiologySummarizer(str(model_path)).to(device)
    model.eval() # Define tokenizer, model, and set to evaluation mode

    def save_progress():
        # Replace the previous snapshot only after the new file is fully written
        temporary_path = progress_path.with_suffix(".tmp")
        with temporary_path.open("w") as file:
            json.dump(
                {
                    "settings": expected_settings,
                    "summaries": summaries,
                    "diagnostics": diagnostics,
                },
                file,
            )
        temporary_path.replace(progress_path)

    # Process one report at a time, starting after the saved predictions
    for index in range(len(summaries), len(reports)):
        inputs = tokenizer(
            INSTRUCTION + reports[index],
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_options)

        # T5 outputs begin with one decoder-start token; exclude it from the count
        sequence = output_ids[0]
        generated_tokens = sequence.numel() - 1
        ended_with_eos = sequence[-1].item() == tokenizer.eos_token_id
        diagnostics.append({
            "input_tokens": inputs["input_ids"].shape[-1],
            "generated_tokens": generated_tokens,
            "ended_with_eos": ended_with_eos,
            "reached_cap_without_eos": (
                generated_tokens >= max_new_tokens and not ended_with_eos
            ),
        })

        summaries.append(
            tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
            ) # Decodes each output and adds it to the result
        )

        if len(summaries) % 25 == 0 or len(summaries) == len(reports):
            save_progress()
            print(
                f"{model_path}: generated {len(summaries)}/{len(reports)} summaries",
                flush=True,
            )

    return summaries # Returns the list of summary results



def calculate_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True,
    ) # Configure four metrics
#     # ROUGE-1	Individual words
      # ROUGE-2	Consecutive two-word phrases
      # ROUGE-L	The longest matching sequence of words in the same order; they don’t need to be adjacent
      # ROUGE-Lsum	A summary-level version of ROUGE-L that accounts for sentence boundaries

    scores = {
        name: []
        for name in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    } # Creates empty list for each metric

    for prediction, reference in zip(predictions, references):
        prediction_sentences = "\n".join(nltk.sent_tokenize(prediction))
        # Splits prediction into sentences, then joins them with newline characters.

        reference_sentences = "\n".join(nltk.sent_tokenize(reference))
        # Splits references into sentences, then joins them with newline characters.

        example_scores = scorer.score(
            reference_sentences,
            prediction_sentences,
        ) # Calculate scores with reference first and prediction second

        for name, score in example_scores.items():
            scores[name].append(score.fmeasure)
            # Stores each metric's F1 score, which combines overlap precision and recall

    return { # Returns arithmetic mean of each metric across samples, on a 0 to 1 scale
        name: float(np.mean(values))
        for name, values in scores.items()
    }


def summarize_generation(diagnostics):
    """Describe lengths and stopping behaviour without treating EOS as accuracy."""
    count = len(diagnostics)
    capped = sum(row["reached_cap_without_eos"] for row in diagnostics)
    return {
        "count": count,
        "mean_generated_tokens": sum(
            row["generated_tokens"] for row in diagnostics
        ) / count,
        "max_generated_tokens": max(row["generated_tokens"] for row in diagnostics),
        "ended_with_eos_count": sum(row["ended_with_eos"] for row in diagnostics),
        "reached_cap_without_eos_count": capped,
        "reached_cap_without_eos_fraction": capped / count,
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output-name", required=True)
    parser.add_argument(
        "--split",
        choices=["development", "final"],
        default="development",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.samples <= 0 or args.max_new_tokens <= 0:
        parser.error("Sample count and output token limit must be positive")

    project_dir = Path(__file__).resolve().parent
    checkpoint_dir = project_dir / "checkpoints" / args.checkpoint
    output_dir = project_dir / "results" / args.output_name

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_dir}")

    if args.resume:
        if not output_dir.is_dir():
            raise FileNotFoundError(
                f"No evaluation directory to resume: {output_dir}"
            )
    elif output_dir.exists():
        raise FileExistsError(
            f"Choose a new output name or use --resume: {output_dir}"
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Evaluation device:", device, flush=True)

    dataset, clean_train = load_audited_data()

    if args.split == "development":
        # Filter data and split into training/development
        _, evaluation_data = split_train_development(clean_train)
    else:
        # The supplied validation split is reserved for final evaluation
        evaluation_data = dataset["validation"]

    if args.samples > len(evaluation_data):
        parser.error("Requested more examples than the selected split contains")

    # Select a reproducible subset of the requested size from the chosen split
    examples = evaluation_data.shuffle(seed=42).select(range(args.samples))
    reports = examples["radiology_report"]
    references = examples["layman_report"]

    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summaries = generate_summaries(
        MODEL_ID, reports, device, args.max_new_tokens,
        output_dir / "baseline_progress.json",
    )
    trained_summaries = generate_summaries(
        checkpoint_dir, reports, device, args.max_new_tokens,
        output_dir / "trained_progress.json",
    )
    # Generates from the original and trained checkpoints using the same reports

    # Read per-example diagnostics from the same recoverable generation snapshots
    with (output_dir / "baseline_progress.json").open() as file:
        baseline_diagnostics = json.load(file)["diagnostics"]
    with (output_dir / "trained_progress.json").open() as file:
        trained_diagnostics = json.load(file)["diagnostics"]

    metrics = {
        "baseline": calculate_rouge(baseline_summaries, references),
        "trained": calculate_rouge(trained_summaries, references),
        "by_source": {},
        "generation": {
            "baseline": summarize_generation(baseline_diagnostics),
            "trained": summarize_generation(trained_diagnostics),
        },
    }

    # Report each source separately so aggregate scores do not hide differences
    for source in sorted(set(examples["source"])):
        indices = [
            index
            for index, value in enumerate(examples["source"])
            if value == source
        ]

        source_references = [references[index] for index in indices]

        metrics["by_source"][source] = {
            "count": len(indices),
            "baseline": calculate_rouge(
                [baseline_summaries[index] for index in indices],
                source_references,
            ),
            "trained": calculate_rouge(
                [trained_summaries[index] for index in indices],
                source_references,
            ),
        }

    with (output_dir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)

    settings = {
        "arguments": vars(args),
        "checkpoint_path": str(checkpoint_dir),
        "baseline_model": MODEL_ID,
        "instruction": INSTRUCTION,
        "split": args.split,
        "subset_seed": 42,
        "do_sample": False,
        "num_beams": 1,
        "min_new_tokens": 0,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "device": str(device),
    }

    with (output_dir / "evaluation_config.json").open("w") as file:
        json.dump(settings, file, indent=2)

    # Preserve every report, reference, and prediction for later error analysis
    with (output_dir / "predictions.jsonl").open("w") as file:
        for index, example in enumerate(examples):
            record = {
                "subset_index": index,
                "source": example["source"],
                "images_path": example["images_path"],
                "report": example["radiology_report"],
                "reference": example["layman_report"],
                "baseline": baseline_summaries[index],
                "trained": trained_summaries[index],
                "baseline_diagnostics": baseline_diagnostics[index],
                "trained_diagnostics": trained_diagnostics[index],
            }
            file.write(json.dumps(record) + "\n")

    print(json.dumps(metrics, indent=2), flush=True)
    print("Saved evaluation results to:", output_dir, flush=True)


if __name__ == "__main__":
    main()
