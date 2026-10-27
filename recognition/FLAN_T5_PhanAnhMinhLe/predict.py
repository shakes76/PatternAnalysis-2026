"""Inference, validation evaluation, and matched model comparisons."""

import argparse
import json
import platform
import re
import time
from importlib.metadata import version
from pathlib import Path

import torch
from rouge_score import rouge_scorer

from dataset import PROJECT_DIR, load_reports, make_prompt, sample_indices
from modules import MODEL_ID, count_params, load_baseline, load_lora


@torch.inference_mode()
def predict(report, tokenizer, model, max_input_length=512, max_new_tokens=256):
    model.eval()
    inputs = tokenizer(make_prompt(report), return_tensors="pt",
                       truncation=True, max_length=max_input_length).to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                            do_sample=False, num_beams=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def evaluate_baseline(tokenizer, model, reports=None, limit=None, seed=42,
                      max_input_length=512, max_new_tokens=256, safety_cases=5,
                      output_dir=PROJECT_DIR / "outputs" / "baseline",
                      training_info=None):
    """Score validation; export separate test cases for human safety review."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")
    if not 3 <= safety_cases <= 5:
        raise ValueError("Select 3-5 safety cases")
    output_dir = Path(output_dir)
    if any((output_dir / f"{name}.json").exists() for name in
           ("metrics", "predictions", "safety_cases")):
        raise ValueError("Choose a new evaluation output directory")
    reports = load_reports() if reports is None else reports
    validation = reports["validation"]
    indices = sample_indices(len(validation), limit, seed)
    if not indices:
        raise ValueError("Validation split is empty")
    model.eval()
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )
    scores = {metric: 0.0 for metric in scorer.rouge_types}
    predictions = []
    generation_seconds, generated_tokens, truncated_inputs, capped_outputs = 0.0, 0, 0, 0
    if len(reports["test"]) < safety_cases:
        raise ValueError("Test split has fewer rows than requested safety cases")
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
        capped_outputs += len(tokens) >= max_new_tokens and tokenizer.eos_token_id not in tokens.tolist()
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
    test_indices = sample_indices(len(reports["test"]), safety_cases, seed)
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
        "outputs_reaching_token_limit_without_eos": capped_outputs,
        "training_seconds": 0, "training_gpu_resources": "none; zero-shot baseline",
        "device": str(model.device),
        "hardware": torch.cuda.get_device_name(model.device) if gpu else platform.processor(),
        "cpu_threads": torch.get_num_threads(), "python": platform.python_version(),
        "packages": {name: version(name) for name in
                     ("torch", "transformers", "datasets", "sentencepiece", "rouge-score")},
        "rouge_convention": "mean per-report F1, stemming, punctuation sentence splitting",
        "clinical_safety": "Test cases require qualitative human review; ROUGE is not a safety score",
    }
    if training_info is not None:
        # Keep training metadata separate: it must not overwrite generation timers/settings.
        metrics["training"] = training_info
        metrics["training_seconds"] = training_info.get("training_seconds")
        metrics["training_gpu_resources"] = training_info.get("training_gpu_resources", "see training metadata")
        metrics["packages"]["peft"] = version("peft")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("metrics", metrics), ("predictions", predictions),
                        ("safety_cases", cases)):
        (output_dir / f"{name}.json").write_text(
            json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return metrics


def compare_runs(run_dirs, output_dir):
    """Match validation and safety examples across all runs."""
    output_dir = Path(output_dir)
    if not run_dirs:
        raise ValueError("Select at least one evaluation directory")
    runs = {str(Path(path).resolve()): {
        name: json.loads((Path(path) / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("metrics", "predictions", "safety_cases")
    } for path in run_dirs}
    baseline = next(iter(runs.values()))
    settings = ("seed", "max_input_length", "max_new_tokens", "decoding",
                "model_revision", "rouge_convention")
    for run in runs.values():
        if any(run["metrics"][key] != baseline["metrics"][key] for key in settings):
            raise ValueError("Comparison settings differ")
        for name in ("predictions", "safety_cases"):
            if [(r["index"], r["report"], r["reference"]) for r in run[name]] != [
                (r["index"], r["report"], r["reference"]) for r in baseline[name]
            ]:
                raise ValueError("Comparison examples differ")
    comparison = {
        "metrics": {name: run["metrics"] for name, run in runs.items()},
        "validation_examples": [], "safety_examples": [],
    }
    for source, target in (("predictions", "validation_examples"),
                           ("safety_cases", "safety_examples")):
        for position, row in enumerate(baseline[source]):
            comparison[target].append({
                "index": row["index"], "report": row["report"],
                "reference": row["reference"],
                "outputs": {name: run[source][position]["prediction"]
                            for name, run in runs.items()},
            })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = ["# Matched model comparison", "",
             f"Validation sample: {baseline['metrics']['evaluated_reports']} reports; "
             f"seed {baseline['metrics']['seed']}. Validation estimates, not clinical safety scores.", "",
             "| Run | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Train seconds | Tokens/sec |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, run in runs.items():
        metrics = run["metrics"]
        values = " | ".join(f"{value:.4f}" for value in metrics["rouge_f1"].values())
        lines.append(f"| {name} | {values} | {metrics['training_seconds']:.2f} | "
                     f"{metrics['inference_tokens_per_second']:.2f} |")
    lines += ["", "## Matched test cases", "",
              "No reference summaries are supplied for test. Review each output against "
              "the source for hallucinations, omissions, certainty, and readability."]
    for case in comparison["safety_examples"]:
        lines += ["", f"### Test {case['index']}", "", f"Source: {case['report']}"]
        for name, summary in case["outputs"].items():
            lines += ["", f"**{name}:** {summary}"]
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", help="Predict one report instead of evaluating")
    parser.add_argument("--compare", type=Path, nargs="+", help="Compare saved evaluation directories without loading a model")
    parser.add_argument("--adapter", type=Path, help="Saved LoRA adapter directory")
    parser.add_argument("--checkpoint", type=Path, help="Full fine-tuned model directory")
    parser.add_argument("--limit", type=int, help="Seeded validation sample; omit for full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "baseline")
    args = parser.parse_args()
    if args.threads < 1 or args.max_new_tokens < 1:
        parser.error("threads and max-new-tokens must be positive")
    torch.set_num_threads(args.threads)
    if args.compare:
        compare_runs(args.compare, args.output_dir)
        print(f"Wrote comparison to {args.output_dir}")
        raise SystemExit(0)
    if args.adapter and args.checkpoint:
        parser.error("Choose either --adapter or --checkpoint")
    tokenizer, model = (load_lora(adapter_path=args.adapter) if args.adapter
                        else load_baseline(model_name=str(args.checkpoint) if args.checkpoint else None))
    if args.report:
        print(predict(args.report, tokenizer, model, max_new_tokens=args.max_new_tokens))
    else:
        checkpoint = args.adapter or args.checkpoint
        info_file = checkpoint / "training_info.json" if checkpoint else None
        training_info = (json.loads(info_file.read_text(encoding="utf-8"))
                         if info_file and info_file.exists() else None)
        metrics = evaluate_baseline(tokenizer, model, limit=args.limit, seed=args.seed,
                                    max_new_tokens=args.max_new_tokens,
                                    output_dir=args.output_dir, training_info=training_info)
        print(json.dumps(metrics, indent=2))
