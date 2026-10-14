"""Load a saved run and explain custom reports or official dataset examples.

LoRA loading: https://huggingface.co/docs/peft/developer_guides/checkpoint
Generation settings: https://huggingface.co/docs/transformers/main_classes/text_generation
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from peft import PeftModel
from transformers import AutoTokenizer, GenerationConfig

from dataset import TASK_PREFIX, load_data
from modules import load_model
from train import calculate_rouge


def load_saved_model(run_dir, device):
    """Reload the baseline, full model or base model with its saved LoRA adapter."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    model_dir = run_dir / "model"
    if config["mode"] == "baseline":
        model = load_model(config["model_name"])
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        # Restore baseline decoding settings from the run configuration.
        model.generation_config.max_length = config["max_target_length"] + 1
        model.generation_config.num_beams = 1
        model.generation_config.do_sample = False
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if config["mode"] == "lora":
            # Attach the saved adapter to the original pretrained base model.
            base_model = load_model(config["model_name"])
            model = PeftModel.from_pretrained(base_model, model_dir)
        else:
            model = load_model(model_dir)
        model.generation_config = GenerationConfig.from_pretrained(model_dir)
    model.to(device).eval()
    return model, tokenizer, config


@torch.inference_mode()
def generate_report(model, tokenizer, report, max_input_length):
    """Generate an explanation and measure generation time and output tokens."""
    inputs = tokenizer(
        TASK_PREFIX + report,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    # Synchronize asynchronous CUDA work before and after timing generation.
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    start = perf_counter()
    # Pass saved settings explicitly so the underlying model also uses them with LoRA.
    output = model.generate(**inputs, generation_config=model.generation_config)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = perf_counter() - start

    # Exclude the decoder start token and padding; include EOS.
    generated_tokens = (output[0, 1:] != tokenizer.pad_token_id).sum().item()
    return {
        "prediction": tokenizer.decode(output[0], skip_special_tokens=True),
        "generated_tokens": generated_tokens,
        "generation_seconds": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run directory saved by train.py")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="An English radiology report to explain")
    inputs.add_argument("--split", choices=["validation", "test"])
    parser.add_argument("--samples", type=int, help="Sample with the run's seed; omit to use the full split")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", help="New JSON output file; omit to print the results")
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer, config = load_saved_model(args.run_dir, args.device)
    if args.text is not None:
        records = [{"radiology_report": args.text}]
    else:
        data = load_data()[args.split]
        if args.samples is not None:
            data = data.shuffle(seed=config["seed"]).select(range(args.samples))
        records = []
        for row in data:
            record = {
                "source": row["source"],
                "images_path": row["images_path"],
                "radiology_report": row["radiology_report"],
            }
            # Test references are not public, so only validation supports ROUGE scoring.
            if args.split == "validation":
                record["reference"] = row["layman_report"]
            records.append(record)

    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    for record in records:
        record.update(generate_report(
            model, tokenizer, record["radiology_report"], config["max_input_length"]
        ))

    seconds = sum(record["generation_seconds"] for record in records)
    tokens = sum(record["generated_tokens"] for record in records)
    metrics = {
        "reports": len(records),
        "generated_tokens": tokens,
        "generation_seconds": seconds,
        "seconds_per_report": seconds / len(records),
        "tokens_per_second": tokens / seconds,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(model.device) / 1024**2
            if model.device.type == "cuda" else None
        ),
    }
    if args.split == "validation":
        metrics.update(calculate_rouge(
            [record["prediction"] for record in records],
            [record["reference"] for record in records],
        ))
    result = {
        "run_dir": str(Path(args.run_dir).resolve()),
        "mode": config["mode"],
        "split": args.split,
        "device": str(model.device),
        "seed": config["seed"],
        "metrics": metrics,
        "predictions": records,
    }
    if args.output is not None:
        # Exclusive creation preserves any previously saved predictions.
        with Path(args.output).open("x", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        print(json.dumps(metrics, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
