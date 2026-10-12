"""Supervised LoRA experiments with a matched zero-shot comparison."""

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from dataset import PROJECT_DIR, get_tokenized_splits, load_reports, remove_train_overlap
from modules import count_params, load_baseline, load_lora
from predict import compare_runs, evaluate_baseline

# Hold alpha/rank at 2; isolate rank and learning-rate effects.
HYPOTHESES = [
    {"name": "rank8", "rank": 8, "alpha": 16, "dropout": 0.05, "lr": 3e-4,
     "hypothesis": "Low-rank adaptation improves task following with few weights."},
    {"name": "rank16", "rank": 16, "alpha": 32, "dropout": 0.05, "lr": 3e-4,
     "hypothesis": "Higher rank improves capacity at the same alpha/rank scaling."},
    {"name": "rank8_fast", "rank": 8, "alpha": 16, "dropout": 0.05, "lr": 1e-3,
     "hypothesis": "A higher learning rate adapts faster but may hurt fidelity."},
]


def train_lora(tokenizer, model, reports, config, output_dir, epochs=1,
               batch_size=4, seed=42):
    """Teacher forcing; only non-padding target tokens contribute to loss."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    tokens = get_tokenized_splits(tokenizer, reports, padding=False)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(tokens["train"], batch_size=batch_size, shuffle=True,
                       collate_fn=collator, generator=generator)
    validation = DataLoader(tokens["validation"], batch_size=batch_size,
                            collate_fn=collator)
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert parameters and all("lora_" in name for name, p in
                              model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=0.0)
    gpu = model.device.type == "cuda"
    if gpu:
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    start = time.perf_counter()
    history, step_losses = [], []
    model.config.use_cache = False
    for epoch in range(epochs):
        model.train()
        loss_sum, token_count = 0.0, 0
        for step, batch in enumerate(train, 1):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            assert not torch.any(batch["labels"] == tokenizer.pad_token_id)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            count = int((batch["labels"] != -100).sum())
            loss_sum += loss.item() * count
            token_count += count
            step_losses.append(loss.item())
            print(f"{config['name']} epoch {epoch + 1} step {step}/{len(train)} "
                  f"loss {loss.item():.4f}", flush=True)
        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.inference_mode():
            for batch in validation:
                batch = {key: value.to(model.device) for key, value in batch.items()}
                count = int((batch["labels"] != -100).sum())
                val_sum += model(**batch).loss.item() * count
                val_count += count
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / token_count,
                        "validation_loss": val_sum / val_count})
    if gpu:
        torch.cuda.synchronize(model.device)
    info = {
        "adaptation": "LoRA", "lora_config": config, "epochs": epochs,
        "optimizer": "AdamW", "weight_decay": 0.0, "gradient_clip_norm": 1.0,
        "max_input_length": 512, "max_target_length": 256,
        "batch_size": batch_size, "train_reports": len(reports["train"]),
        "training_seconds": time.perf_counter() - start,
        "training_gpu_resources": torch.cuda.get_device_name(model.device) if gpu else "CPU only",
        "training_peak_gpu_vram_mib": torch.cuda.max_memory_allocated(model.device) / 2**20 if gpu else None,
        "training_scope": "optimisation and validation loss; excludes tokenisation and generation",
        "loss_history": history, "seed": seed, **count_params(model),
    }
    model.config.use_cache = True
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    plt.plot(range(1, len(step_losses) + 1), step_losses, label="Training batch loss")
    plt.plot([row["epoch"] * len(train) for row in history],
             [row["validation_loss"] for row in history], "o-", label="Validation loss")
    plt.xlabel("Optimiser step")
    plt.ylabel("Token cross-entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()
    return info


def run_experiments(train_limit=128, eval_limit=20, epochs=1, batch_size=4,
                    seed=42, output_dir=PROJECT_DIR / "outputs" / "lora"):
    if train_limit is not None and train_limit < 1:
        raise ValueError("train_limit must be positive or None")
    if eval_limit is not None and eval_limit < 1:
        raise ValueError("eval_limit must be positive or None")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    published = load_reports()
    clean = remove_train_overlap(published)
    train_indices = list(range(len(clean["train"])))
    if train_limit is not None:
        train_indices = sorted(random.Random(seed).sample(
            train_indices, min(train_limit, len(train_indices))))
    val_indices = list(range(len(clean["validation"])))
    if eval_limit is not None:
        val_indices = sorted(random.Random(seed).sample(
            val_indices, min(eval_limit, len(val_indices))))
    training = DatasetDict({"train": clean["train"].select(train_indices),
                            "validation": clean["validation"].select(val_indices)})
    audit = {"published_train": len(published["train"]), "clean_train": len(clean["train"]),
             "removed_train": len(published["train"]) - len(clean["train"]),
             "policy": "remove training source/image and normalised report matches with validation/test",
             "patient_separation": "unverified; no patient IDs supplied",
             "sample_train_indices_after_filter": train_indices,
             "validation_indices": val_indices}
    (output_dir / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    tokenizer, model = load_baseline()
    run_dirs = [output_dir / "baseline"]
    evaluate_baseline(tokenizer, model, reports=published, limit=eval_limit,
                      seed=seed, output_dir=run_dirs[0])
    del model
    gc.collect()
    for config in HYPOTHESES:
        random.seed(seed)
        torch.manual_seed(seed)
        tokenizer, model = load_lora(config["rank"], config["alpha"], config["dropout"])
        adapter = output_dir / config["name"] / "adapter"
        info = train_lora(tokenizer, model, training, config, adapter,
                          epochs=epochs, batch_size=batch_size, seed=seed)
        run_dirs.append(output_dir / config["name"])
        evaluate_baseline(tokenizer, model, reports=published, limit=eval_limit,
                          seed=seed, output_dir=run_dirs[-1], training_info=info)
        del model
        gc.collect()
    comparison = compare_runs(run_dirs, output_dir)
    for name, metrics in comparison["metrics"].items():
        print(name, metrics["rouge_f1"], flush=True)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=128, help="0 for all clean training rows")
    parser.add_argument("--eval-limit", type=int, default=20, help="0 for full validation")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "lora")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run_experiments(args.train_limit or None, args.eval_limit or None, args.epochs,
                    args.batch_size, args.seed, args.output_dir)
