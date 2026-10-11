"""Zero-shot prediction and held-out baseline evaluation."""

import argparse
import json
import platform
import random
import re
import time
from importlib.metadata import version
from pathlib import Path

import torch
from rouge_score import rouge_scorer

from dataset import PROJECT_DIR, load_reports, make_prompt
from modules import MODEL_ID, count_params, load_baseline


@torch.inference_mode()
def predict(report, tokenizer, model, max_input_length=512, max_new_tokens=256):
    inputs = tokenizer(make_prompt(report), return_tensors="pt",
                       truncation=True, max_length=max_input_length).to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                            do_sample=False, num_beams=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def evaluate_baseline(tokenizer, model, reports=None, limit=None, seed=42,
                      max_input_length=512, max_new_tokens=256, safety_cases=5,
                      output_dir=PROJECT_DIR / "outputs" / "baseline"):
    """Score validation; export separate test cases for human safety review."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")
    if not 3 <= safety_cases <= 5:
        raise ValueError("Select 3-5 safety cases")
    reports = load_reports() if reports is None else reports
    validation = reports["validation"]
    indices = list(range(len(validation)))
    if limit is not None:
        indices = sorted(random.Random(seed).sample(indices, min(limit, len(indices))))
    if not indices:
        raise ValueError("Validation split is empty")
    model.eval()
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )
    scores = {metric: 0.0 for metric in scorer.rouge_types}
    predictions = []
    generation_seconds, generated_tokens, truncated_inputs = 0.0, 0, 0
    gpu = model.device.type == "cuda"
    predict(validation[indices[0]]["radiology_report"], tokenizer, model,
            max_input_length, max_new_tokens)
    if gpu:
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)

    for number, index in enumerate(indices, 1):
        row = validation[index]
        prompt = make_prompt(row["radiology_report"])
        truncated_inputs += len(tokenizer(prompt)["input_ids"]) > max_input_length
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_input_length).to(model.device)
        if gpu:
            torch.cuda.synchronize(model.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=False, num_beams=1)
        if gpu:
            torch.cuda.synchronize(model.device)
        generation_seconds += time.perf_counter() - start
        tokens = output[0, 1:]
        generated_tokens += int(((tokens != tokenizer.pad_token_id) &
                                 (tokens != tokenizer.eos_token_id)).sum().item())
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        reference = row["layman_report"]
        # ROUGE-Lsum expects sentence boundaries as newlines.
        result = scorer.score(re.sub(r"(?<=[.!?])\s+", "\n", reference.strip()),
                              re.sub(r"(?<=[.!?])\s+", "\n", summary.strip()))
        for metric in scores:
            scores[metric] += result[metric].fmeasure
        predictions.append({"index": index, "source": row["source"],
                            "images_path": row["images_path"],
                            "report": row["radiology_report"],
                            "reference": reference, "prediction": summary})
        print(f"Validation {number}/{len(indices)}", flush=True)

    peak_vram = torch.cuda.max_memory_allocated(model.device) / 2**20 if gpu else None
    test_indices = sorted(random.Random(seed).sample(
        range(len(reports["test"])), safety_cases
    ))
    cases = []
    for index in test_indices:
        row = reports["test"][index]
        cases.append({
            "index": index, "source": row["source"],
            "report": row["radiology_report"], "reference": row["layman_report"],
            "prediction": predict(row["radiology_report"], tokenizer, model,
                                  max_input_length, max_new_tokens),
            "hallucinations": None, "critical_omissions": None,
            "certainty_changes": None, "readability": None,
            "review_status": "pending human review",
        })
    metrics = {
        "model": MODEL_ID,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "split": "validation", "evaluated_reports": len(indices),
        "full_validation": len(indices) == len(validation), "seed": seed,
        "rouge_f1": {metric: score / len(indices) for metric, score in scores.items()},
        **count_params(model), "peak_gpu_vram_mib": peak_vram,
        "generation_seconds": generation_seconds, "generated_tokens": generated_tokens,
        "inference_tokens_per_second": generated_tokens / generation_seconds,
        "throughput_scope": "generation only; batch size 1; excludes warmup and safety cases",
        "max_input_length": max_input_length, "max_new_tokens": max_new_tokens,
        "truncated_inputs": truncated_inputs, "decoding": "greedy",
        "training_seconds": 0, "training_gpu_resources": "none; zero-shot baseline",
        "device": str(model.device),
        "hardware": torch.cuda.get_device_name(model.device) if gpu else platform.processor(),
        "cpu_threads": torch.get_num_threads(), "python": platform.python_version(),
        "packages": {name: version(name) for name in
                     ("torch", "transformers", "datasets", "sentencepiece", "rouge-score")},
        "rouge_convention": "mean per-report F1, stemming, punctuation sentence splitting",
        "clinical_safety": "Test cases require qualitative human review; ROUGE is not a safety score",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("metrics", metrics), ("predictions", predictions),
                        ("safety_cases", cases)):
        (output_dir / f"{name}.json").write_text(
            json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", help="Predict one report instead of evaluating")
    parser.add_argument("--limit", type=int, help="Seeded validation sample; omit for full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "baseline")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    tokenizer, model = load_baseline()
    if args.report:
        print(predict(args.report, tokenizer, model, max_new_tokens=args.max_new_tokens))
    else:
        metrics = evaluate_baseline(tokenizer, model, limit=args.limit, seed=args.seed,
                                    max_new_tokens=args.max_new_tokens,
                                    output_dir=args.output_dir)
        print(json.dumps({key: value for key, value in metrics.items()
                          if key != "trainable_layers"}, indent=2))
