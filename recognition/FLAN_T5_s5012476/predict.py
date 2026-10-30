"""
Inference pipeline for FLAN-T5 radiology report summarisation.

Supports full fine-tuning and LoRA fine-tuning checkpoints.
"""

import argparse
import csv
import json
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

from dataset import MODEL_NAME, PROMPT, load_radiology_dataset
from evaluation import (
    plot_dataset_overlap,
    audit_dataset_overlap,
    calculate_rouge,
    plot_rouge_scores,
    get_training_reports,
    normalize_report,
    calculate_grouped_rouge,
    plot_seen_unseen_rouge,
)

def load_evaluation_dataset(max_eval_samples=0):
    """
    Load the official validation split for model evaluation.

    Args:
        max_eval_samples (int): Maximum samples to evaluate.
            A value of 0 uses the entire validation split.

    Returns:
        Dataset: Validation dataset.
    """
    dataset = load_radiology_dataset()
    validation_dataset = dataset["validation"]

    if max_eval_samples > 0:
        sample_count = min(
            max_eval_samples,
            len(validation_dataset),
        )

        validation_dataset = validation_dataset.select(
            range(sample_count)
        )

    print("\n=== Evaluation Dataset ===")
    print(f"Validation samples: {len(validation_dataset):,}")
    print(f"Columns: {validation_dataset.column_names}")

    return validation_dataset



def parse_args():
    """Parse inference arguments."""
    parser = argparse.ArgumentParser(
        description="Generate layman summaries using fine-tuned FLAN-T5."
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["zero_shot", "full", "lora"],
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=None,
        help="Path to the fine-tuned checkpoint (required for full/lora).",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Radiology report for single prediction.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model on the validation dataset.",
    )

    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=0,
        help="Maximum evaluation samples; 0 uses the full validation split.",
    )

    parser.add_argument(
        "--audit_overlap",
        action="store_true",
        help="Audit exact report overlap between training and validation.",
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation inference.",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="smoke",
        choices=["smoke", "final"],
        help="Experiment category for evaluation outputs.",
    )

    return parser.parse_args()

def load_model_and_tokenizer(method, checkpoint, device):
    """Load the selected model and its tokenizer."""

    if method == "zero_shot":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    elif method == "full":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        model = PeftModel.from_pretrained(
            base_model,
            checkpoint,
        )

    model = model.to(device)
    model.eval()

    return model, tokenizer

def generate_summary(
    model,
    tokenizer,
    report,
    device,
    max_new_tokens=128,
    num_beams=4,
):
    """Generate a patient-friendly summary of a radiology report."""
    input_text = PROMPT + report

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )

    summary = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return summary

def generate_batch_summaries(
    model,
    tokenizer,
    reports,
    device,
    max_new_tokens=128,
    num_beams=4,
):
    """
    Generate layman summaries for a batch of radiology reports.
    """
    input_texts = [PROMPT + report for report in reports]

    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )

    summaries = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )

    return summaries

def evaluate_predictions(
    model,
    tokenizer,
    validation_dataset,
    device,
    args,
):
    """
    Generate and save predictions for the validation dataset.

    Args:
        model: Fine-tuned FLAN-T5 model.
        tokenizer: FLAN-T5 tokenizer.
        validation_dataset: Official validation dataset.
        device (torch.device): Computation device.
        args (argparse.Namespace): Inference configuration.
    """
    output_dir = (
        Path("outputs")
        / args.run_name
        / args.method
        / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_config = {
        "method": args.method,
        "run_name": args.run_name,
        "checkpoint": str(args.checkpoint),
        "evaluation_split": "validation",
        "num_samples": len(validation_dataset),
        "eval_batch_size": args.eval_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "device": str(device),
    }

    config_path = output_dir / "evaluation_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_config, f, indent=4)

    print(f"Evaluation configuration saved to: {config_path}")

    predictions = []

    print("\n=== Generating Evaluation Predictions ===")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    training_reports = get_training_reports()
    inference_start = time.perf_counter()

    total_samples = len(validation_dataset)

    for start_index in range(0, total_samples, args.eval_batch_size):
        end_index = min(
            start_index + args.eval_batch_size,
            total_samples,
        )

        batch_examples = [
            validation_dataset[index]
            for index in range(start_index, end_index)
        ]

        reports = [
            example["radiology_report"]
            for example in batch_examples
        ]

        generated_summaries = generate_batch_summaries(
            model=model,
            tokenizer=tokenizer,
            reports=reports,
            device=device,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )

        for index, (example, generated_summary) in enumerate(
            zip(batch_examples, generated_summaries),
            start=start_index,
        ):
            report = example["radiology_report"]

            predictions.append(
                {
                    "sample_id": index,
                    "source": example["source"],
                    "split_group": (
                        "seen"
                        if normalize_report(report) in training_reports
                        else "unseen"
                    ),
                    "radiology_report": report,
                    "reference_summary": example["layman_report"],
                    "generated_summary": generated_summary,
                }
            )

        print(
            f"Generated {end_index}/{total_samples} summaries"
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    total_inference_time = time.perf_counter() - inference_start

    num_samples = len(predictions)

    average_latency = total_inference_time / num_samples
    throughput = num_samples / total_inference_time

    peak_gpu_memory = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda"
        else None
    )

    csv_path = output_dir / "predictions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "source",
                "split_group",
                "radiology_report",
                "reference_summary",
                "generated_summary",
            ],
        )

        writer.writeheader()
        writer.writerows(predictions)

    print(f"\nPredictions saved to: {csv_path}")

    # Calculate ROUGE scores for the generated summaries.
    rouge_scores = calculate_rouge(predictions)

    rouge_path = output_dir / "rouge_scores.json"

    with open(rouge_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "method": args.method,
                "num_samples": len(predictions),
                "scores": rouge_scores,
            },
            file,
            indent=4,
        )

    print("\n=== ROUGE Evaluation ===")

    for metric, score in rouge_scores.items():
        print(f"{metric}: {score:.4f}")

    print(f"ROUGE scores saved to: {rouge_path}")

    plot_rouge_scores(
        rouge_scores=rouge_scores,
        output_dir=output_dir,
        method=args.method,
    )

    grouped_rouge = calculate_grouped_rouge(predictions)

    grouped_path = output_dir / "seen_unseen_rouge.json"

    with open(grouped_path, "w", encoding="utf-8") as file:
        json.dump(grouped_rouge, file, indent=4)

    print("\n=== Seen vs Unseen Evaluation ===")

    for group, result in grouped_rouge.items():
        print(f"{group.capitalize()} samples: {result['num_samples']}")

        if result["scores"] is not None:
            for metric, score in result["scores"].items():
                print(f"  {metric}: {score:.4f}")
        else:
            print("  No samples available.")

    print(f"Grouped ROUGE saved to: {grouped_path}")

    plot_seen_unseen_rouge(
        grouped_rouge=grouped_rouge,
        output_dir=output_dir,
    )

    inference_metrics = {
        "method": args.method,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "num_samples": num_samples,
        "total_inference_seconds": total_inference_time,
        "average_latency_seconds": average_latency,
        "throughput_samples_per_second": throughput,
        "peak_gpu_memory_gb": peak_gpu_memory,
    }

    metrics_path = output_dir / "inference_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(inference_metrics, file, indent=4)

    print("\n=== Inference Performance ===")
    print(f"Total inference time: {total_inference_time:.2f} seconds")
    print(f"Average latency: {average_latency:.4f} seconds/sample")
    print(f"Throughput: {throughput:.4f} samples/second")

    if peak_gpu_memory is not None:
        print(f"Peak GPU memory: {peak_gpu_memory:.3f} GB")
    else:
        print("Peak GPU memory: N/A (CPU execution)")

    print(f"Inference metrics saved to: {metrics_path}")

    return predictions

def main():
    """Run inference from the command line."""
    args = parse_args()

    if args.method in ["full", "lora"] and not args.checkpoint:
        raise ValueError(
            "--checkpoint is required for full and lora methods."
        )

    if args.audit_overlap:
        audit_dataset_overlap()
        return

    if args.max_eval_samples < 0:
        raise ValueError("max_eval_samples must be non-negative.")

    if args.evaluate and args.text is not None:
        raise ValueError(
            "Use either --evaluate or --text, not both."
        )

    if not args.evaluate and not args.text:
        raise ValueError(
            "Provide --text for prediction or --evaluate for evaluation."
        )
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    if args.num_beams <= 0:
        raise ValueError("num_beams must be positive.")

    if args.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n=== Inference Configuration ===")
    print(f"Method: {args.method}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    model, tokenizer = load_model_and_tokenizer(
        method=args.method,
        checkpoint=args.checkpoint,
        device=device,
    )

    if args.evaluate:
        validation_dataset = load_evaluation_dataset(
            max_eval_samples=args.max_eval_samples
        )

        evaluate_predictions(
            model=model,
            tokenizer=tokenizer,
            validation_dataset=validation_dataset,
            device=device,
            args=args,
        )

        print("\nEvaluation predictions completed successfully.")
        return

    summary = generate_summary(
        model=model,
        tokenizer=tokenizer,
        report=args.text,
        device=device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    print("\n=== Radiology Report ===")
    print(args.text)

    print("\n=== Generated Layman Summary ===")
    print(summary)


if __name__ == "__main__":
    main()