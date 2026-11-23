"""Profile one model on a fixed development subset without tuning or scoring."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from config import INSTRUCTION, MODEL_ID
from dataset import load_audited_data, split_train_development
from modules import RadiologySummarizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Local checkpoint folder name; omit for the untuned baseline")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    if args.samples <= 0 or args.warmup < 0:
        parser.error("Samples must be positive and warmup must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this benchmark inside a Slurm GPU allocation")

    project_dir = Path(__file__).resolve().parents[2]
    output_path = project_dir / "results" / "inference_profiles" / f"{args.output_name}.json"
    if output_path.exists():
        raise FileExistsError(f"Profile already exists: {output_path}")

    model_path = MODEL_ID
    if args.checkpoint:
        checkpoint_dir = project_dir / "checkpoints" / args.checkpoint
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(checkpoint_dir)
        model_path = str(checkpoint_dir)

    torch.manual_seed(42)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RadiologySummarizer(model_path).to(device)
    model.eval()

    _, clean_train = load_audited_data()
    _, development = split_train_development(clean_train)
    if args.samples > len(development):
        parser.error("Requested more samples than the development split contains")
    reports = list(development.shuffle(seed=42).select(range(args.samples))["radiology_report"])

    generation_options = {
        "max_new_tokens": 1024,
        "min_new_tokens": 0,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
    }
    # Tokenisation and host-to-device transfer are outside generation timing.
    inputs = [
        tokenizer(INSTRUCTION + report, return_tensors="pt").to(device)
        for report in reports
    ]

    with torch.inference_mode():
        for index in range(args.warmup):
            warmup_output = model.generate(
                **inputs[index % len(inputs)], **generation_options
            )
            del warmup_output
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        measurements = []
        for index, example_inputs in enumerate(inputs):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = model.generate(**example_inputs, **generation_options)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started

            token_count = output.shape[-1] - 1  # Exclude T5 decoder-start token, include EOS.
            ended_with_eos = output[0, -1].item() == tokenizer.eos_token_id
            measurements.append({
                "development_subset_index": index,
                "input_tokens": example_inputs["input_ids"].shape[-1],
                "generated_tokens": token_count,
                "generation_seconds": elapsed,
                "ended_with_eos": ended_with_eos,
                "reached_cap_without_eos": token_count >= 1024 and not ended_with_eos,
            })
            del output
            print(f"Profiled {index + 1}/{len(inputs)}", flush=True)

    elapsed_total = sum(row["generation_seconds"] for row in measurements)
    tokens_total = sum(row["generated_tokens"] for row in measurements)
    latencies = [row["generation_seconds"] for row in measurements]
    result = {
        "model_path": model_path,
        "split": "development",
        "seed": 42,
        "samples": args.samples,
        "warmup_calls": args.warmup,
        "batch_size": 1,
        "reports_sha256": hashlib.sha256(
            json.dumps(reports, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "instruction": INSTRUCTION,
        "generation_options": generation_options,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "lora_merged": False,
        "total_generated_tokens": tokens_total,
        "total_generation_seconds": elapsed_total,
        "generated_tokens_per_second": tokens_total / elapsed_total,
        "mean_latency_seconds": float(np.mean(latencies)),
        "median_latency_seconds": float(np.median(latencies)),
        "p95_latency_seconds": float(np.percentile(latencies, 95)),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024 ** 3,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024 ** 3,
        "timing_scope": (
            "Synchronized model.generate calls only; excludes loading, tokenisation, "
            "transfer, decoding, warmup, and scoring. EOS counts as an output token. "
            "Peak memory includes resident model and pretokenised input tensors."
        ),
        "measurements": measurements,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    with temporary_path.open("w") as file:
        json.dump(result, file, indent=2)
    temporary_path.replace(output_path)
    print(json.dumps({key: value for key, value in result.items() if key != "measurements"}, indent=2))
    print("Saved profile to:", output_path, flush=True)


if __name__ == "__main__":
    main()
