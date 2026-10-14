"""Run baseline evaluation, full fine-tuning or LoRA and save the results.

Trainer API: https://huggingface.co/docs/transformers/main_classes/trainer
ROUGE implementation: https://github.com/google-research/google-research/tree/master/rouge
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from dataset import load_data, tokenize_data
from modules import count_parameters, load_model

matplotlib.use("Agg")  # Save plots without requiring a display.
import matplotlib.pyplot as plt


ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")


def calculate_rouge(predictions, references):
    """Return mean per-example F1 on a 0–100 scale; Lsum uses English sentence splitting."""
    scorer = rouge_scorer.RougeScorer(
        ROUGE_TYPES, use_stemmer=True, split_summaries=True
    )
    scores = [
        scorer.score(reference, prediction)
        for reference, prediction in zip(references, predictions)
    ]
    return {
        name: 100 * sum(score[name].fmeasure for score in scores) / len(scores)
        for name in ROUGE_TYPES
    }


def plot_history(history, output_path):
    """Plot training/validation loss and four validation ROUGE scores by epoch."""
    training = [entry for entry in history if "loss" in entry]
    validation = [entry for entry in history if "eval_loss" in entry]
    epochs = [entry["epoch"] for entry in validation]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        [entry["epoch"] for entry in training],
        [entry["loss"] for entry in training],
        marker="o", label="Training",
    )
    axes[0].plot(
        epochs, [entry["eval_loss"] for entry in validation],
        marker="o", label="Validation",
    )
    axes[0].set_ylabel("Loss")
    for name in ROUGE_TYPES:
        axes[1].plot(
            epochs, [entry[f"eval_{name}"] for entry in validation],
            marker="o", label=name,
        )
    axes[1].set_ylabel("ROUGE F1 (0–100)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "full", "lora"], required=True)
    parser.add_argument("--output-dir", required=True, help="New directory for this run")
    parser.add_argument("--model-name", default="google/flan-t5-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=512)
    parser.add_argument("--train-samples", type=int, help="Limit training samples for a small test run")
    parser.add_argument("--eval-samples", type=int, help="Limit evaluation samples for a small test run")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        logging_strategy="epoch",
        # Save weights, optimizer and random states after validation; keep the latest checkpoint.
        save_strategy="epoch",
        save_total_limit=1,
        predict_with_generate=True,
        optim="adamw_torch",
        report_to=[],
        seed=args.seed,
    )
    # Only the main process creates the new run directory; other processes wait.
    with training_args.main_process_first(local=False):
        if training_args.process_index == 0:
            output_dir.mkdir(parents=True)
            # Save settings before training so they survive an interrupted job.
            (output_dir / "run_config.json").write_text(
                json.dumps(vars(args), indent=2), encoding="utf-8"
            )

    data = load_data()
    raw_train, raw_eval = data["train"], data["validation"]
    # Sample within each official split to keep training and validation separate.
    if args.train_samples is not None:
        raw_train = raw_train.shuffle(seed=args.seed).select(range(args.train_samples))
    if args.eval_samples is not None:
        raw_eval = raw_eval.shuffle(seed=args.seed).select(range(args.eval_samples))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    eval_data = tokenize_data(
        raw_eval, tokenizer, args.max_input_length, args.max_target_length
    )
    train_data = None
    if args.mode != "baseline":
        train_data = tokenize_data(
            raw_train, tokenizer, args.max_input_length, args.max_target_length
        )
    model = load_model(args.model_name, use_lora=args.mode == "lora")
    if args.mode == "baseline":
        model.requires_grad_(False)
    # max_length includes the decoder start token, so allow one extra token.
    model.generation_config.max_length = args.max_target_length + 1
    model.generation_config.num_beams = 1
    model.generation_config.do_sample = False
    total, trainable = count_parameters(model)
    predictions = []

    def compute_metrics(result):
        nonlocal predictions
        # Replace Trainer's -100 padding with pad tokens before decoding.
        ids = np.where(result.predictions == -100, tokenizer.pad_token_id, result.predictions)
        predictions = tokenizer.batch_decode(ids, skip_special_tokens=True)
        # Score against complete references, rather than truncated training labels.
        return calculate_rouge(predictions, raw_eval["layman_report"])

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics,
    )
    device = training_args.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.mode == "baseline":
        metrics = trainer.evaluate()
    else:
        result = trainer.train()
        # Reuse the final validation results to avoid generating the full set again.
        evaluations = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
        metrics = {**result.metrics, **evaluations[-1]}
        trainer.save_model(str(output_dir / "model"))
        trainer.save_state()

    if trainer.is_world_process_zero():
        metrics.update(
            total_parameters=total,
            trainable_parameters=trainable,
            device=str(device),
            peak_gpu_memory_mb=(
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda" else None
            ),
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        # Keep source reports, complete references and predictions for error analysis.
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
            for row, prediction in zip(raw_eval, predictions):
                record = {
                    "source": row["source"],
                    "images_path": row["images_path"],
                    "radiology_report": row["radiology_report"],
                    "reference": row["layman_report"],
                    "prediction": prediction,
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if args.mode != "baseline":
            # Save generation settings separately because adapter files do not include them.
            model.generation_config.save_pretrained(output_dir / "model")
            plot_history(trainer.state.log_history, output_dir / "training_curves.png")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
