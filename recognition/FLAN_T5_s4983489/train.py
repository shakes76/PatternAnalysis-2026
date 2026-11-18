"""Train and evaluate FLAN-T5 with an explicit PyTorch loop."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import matplotlib
import torch
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed,
)

from dataset import load_data, prepare_splits, tokenize_data
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


def train_one_epoch(model, loader, optimizer, scheduler, accumulation_steps):
    """Train one epoch and return the sample-weighted batch loss and update count."""
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    total_loss, reports, updates = 0.0, 0, 0

    for batch_index, batch in enumerate(loader):
        batch = {name: tensor.to(model.device) for name, tensor in batch.items()}
        if batch_index % accumulation_steps == 0:
            # Average the actual number of microbatches, including the final short group.
            group_size = min(accumulation_steps, len(loader) - batch_index)
        loss = model(**batch).loss
        (loss / group_size).backward()
        batch_size = batch["input_ids"].shape[0]
        total_loss += loss.item() * batch_size
        reports += batch_size

        if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if updates % 250 == 0 or batch_index + 1 == len(loader):
                print(json.dumps({
                    "optimizer_updates": updates,
                    "training_reports": reports,
                    "train_loss": total_loss / reports,
                }), flush=True)

    return total_loss / reports, updates


@torch.inference_mode()
def evaluate(model, tokenizer, loader, references, unseen_indices=None):
    """Evaluate loss and source-only generation in loader order, without gradients."""
    model.eval()
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    start = perf_counter()
    total_loss, reports = 0.0, 0
    predictions = []

    for batch in loader:
        batch = {name: tensor.to(model.device) for name, tensor in batch.items()}
        batch_size = batch["input_ids"].shape[0]
        total_loss += model(**batch).loss.item() * batch_size
        reports += batch_size
        # Labels and teacher-forced decoder inputs are used for loss only, never generation.
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            generation_config=model.generation_config,
        )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    metrics = {
        "eval_loss": total_loss / reports,
        "eval_reports": reports,
        **{f"eval_{name}": score for name, score in calculate_rouge(predictions, references).items()},
    }
    if unseen_indices is not None:
        metrics["eval_unseen_reports"] = len(unseen_indices)
    # An empty unseen subset has no ROUGE score; its count is still recorded above.
    if unseen_indices:
        unseen_scores = calculate_rouge(
            [predictions[index] for index in unseen_indices],
            [references[index] for index in unseen_indices],
        )
        metrics.update({f"eval_unseen_{name}": score for name, score in unseen_scores.items()})
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    metrics["eval_runtime"] = perf_counter() - start
    return metrics, predictions


def plot_history(history, output_path):
    """Plot training/development loss and development ROUGE scores by epoch."""
    training = [entry for entry in history if "loss" in entry]
    development = [entry for entry in history if "dev_loss" in entry]
    epochs = [entry["epoch"] for entry in development]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        [entry["epoch"] for entry in training],
        [entry["loss"] for entry in training],
        marker="o", label="Training",
    )
    axes[0].plot(
        epochs, [entry["dev_loss"] for entry in development],
        marker="o", label="Development",
    )
    axes[0].set_ylabel("Loss")
    for name in ROUGE_TYPES:
        axes[1].plot(
            epochs, [entry[f"dev_{name}"] for entry in development],
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
    parser.add_argument("--dev-fraction", type=float, default=0.05)
    parser.add_argument("--train-samples", type=int, help="Limit training samples for a small test run")
    parser.add_argument("--eval-samples", type=int, help="Limit each evaluation split for a small test run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True)
    config = {
        **vars(args), "training_loop": "manual_pytorch",
        "evaluation_protocol": "grouped_development_final_validation",
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    data = load_data()
    raw_train, raw_dev, raw_eval, audit = prepare_splits(data, args.seed, args.dev_fraction)
    # Optional small runs sample only after the disjoint partitions are fixed.
    if args.train_samples is not None:
        raw_train = raw_train.shuffle(seed=args.seed).select(range(args.train_samples))
    if args.eval_samples is not None:
        raw_dev = raw_dev.shuffle(seed=args.seed).select(range(args.eval_samples))
        raw_eval = raw_eval.shuffle(seed=args.seed).select(range(args.eval_samples))

    audit.update(
        used_train_rows=len(raw_train), used_development_rows=len(raw_dev),
        used_final_rows=len(raw_eval),
    )
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = load_model(args.model_name, use_lora=args.mode == "lora")
    device = torch.device(args.device)
    model.to(device)
    if args.mode == "baseline":
        model.requires_grad_(False)
    # max_length includes the decoder start token, so allow one extra token.
    model.generation_config.max_length = args.max_target_length + 1
    model.generation_config.num_beams = 1
    model.generation_config.do_sample = False
    total, trainable = count_parameters(model)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    training_metrics = {}
    if args.mode != "baseline":
        train_data = tokenize_data(
            raw_train, tokenizer, args.max_input_length, args.max_target_length
        )
        dev_data = tokenize_data(
            raw_dev, tokenizer, args.max_input_length, args.max_target_length
        )
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
            generator=torch.Generator().manual_seed(args.seed),
        )
        dev_loader = DataLoader(
            dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
        )
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
        total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / total_steps),
        )
        history, global_step = [], 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()

        for epoch in range(1, args.epochs + 1):
            print(json.dumps({"epoch": epoch, "phase": "training"}), flush=True)
            loss, updates = train_one_epoch(
                model, train_loader, optimizer, scheduler, args.gradient_accumulation_steps,
            )
            global_step += updates
            # Save trained weights before generation, which can be slow on long reports.
            model_dir = output_dir / "model"
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            model.generation_config.save_pretrained(model_dir)
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                output_dir / "checkpoint.pt",
            )
            print(json.dumps({"epoch": epoch, "phase": "development"}), flush=True)
            dev_metrics, _ = evaluate(model, tokenizer, dev_loader, raw_dev["layman_report"])
            dev_metrics = {name.replace("eval_", "dev_", 1): value
                           for name, value in dev_metrics.items()}
            dev_metrics.update(epoch=epoch, global_step=global_step)
            history.extend([{"epoch": epoch, "loss": loss}, dev_metrics])
            (output_dir / "training_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            print(json.dumps(dev_metrics, indent=2), flush=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_metrics = {
            "train_runtime": perf_counter() - start, "train_loss": loss,
            "epoch": args.epochs, "global_step": global_step,
        }
        plot_history(history, output_dir / "training_curves.png")

    # Official validation is evaluated once, after the fixed training schedule ends.
    print(json.dumps({"phase": "final_validation"}), flush=True)
    eval_data = tokenize_data(
        raw_eval, tokenizer, args.max_input_length, args.max_target_length
    )
    eval_loader = DataLoader(
        eval_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
    )
    metrics, predictions = evaluate(model, tokenizer, eval_loader, raw_eval["layman_report"])
    metrics.update(training_metrics)
    metrics.update(
        evaluation_split="official_validation",
        total_parameters=total,
        trainable_parameters=trainable,
        device=str(device),
        peak_gpu_memory_mb=(
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else None
        ),
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
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
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
