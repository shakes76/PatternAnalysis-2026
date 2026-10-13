"""Supervised LoRA and full fine-tuning with matched comparisons."""

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
from modules import load_full
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


def full_optimizers(model, learning_rate, optimizer_name, weight_decay=0.01):
    """Muon: hidden Linear weights; AdamW: embeddings, norms, and other weights.

    https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html
    """
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if optimizer_name not in ("adamw", "adamw_torch_fused", "muon_adamw"):
        raise ValueError("Use adamw, adamw_torch_fused, or muon_adamw")
    hidden = {id(layer.weight) for name, layer in model.named_modules()
              if isinstance(layer, torch.nn.Linear) and name != "lm_head"}
    embeddings = {id(layer.weight) for layer in model.modules()
                  if isinstance(layer, torch.nn.Embedding)}
    hidden -= embeddings | {id(model.get_output_embeddings().weight)}
    muon = [(name, p) for name, p in named
            if optimizer_name == "muon_adamw" and id(p) in hidden and p.ndim == 2]
    muon_ids = {id(p) for _, p in muon}
    adam = [(name, p) for name, p in named if id(p) not in muon_ids]
    groups = [{"params": [p for _, p in adam if p.ndim >= 2], "weight_decay": weight_decay},
              {"params": [p for _, p in adam if p.ndim < 2], "weight_decay": 0.0}]
    fused = optimizer_name == "adamw_torch_fused"
    optimizers = [torch.optim.AdamW(groups, lr=learning_rate, foreach=False, fused=fused)]
    if muon:
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("Muon requires a newer PyTorch build (2.9 or later)")
        optimizers.append(torch.optim.Muon(
            [p for _, p in muon], lr=learning_rate, weight_decay=weight_decay,
            momentum=0.95, adjust_lr_fn="match_rms_adamw",
        ))
    partition = {"muon": [name for name, _ in muon],
                 "adamw": [name for name, _ in adam]}
    assert len(named) == len(partition["muon"]) + len(partition["adamw"])
    assert not muon_ids & {id(p) for _, p in adam}
    return optimizers, partition


def fine_tune(tokenizer, model, reports, tokens, output_dir, learning_rate=5e-5,
              optimizer_name="adamw", epoch_options=(1, 3), batch_size=16,
              eval_every=500, eval_limit=150, seed=42, generation_every=None):
    """Check validation loss periodically; generate at selected steps/checkpoints."""
    if not epoch_options or min(epoch_options) < 1 or batch_size < 1 or eval_every < 1:
        raise ValueError("Epochs, batch size, and evaluation interval must be positive")
    if not len(tokens["train"]) or not len(tokens["validation"]):
        raise ValueError("Training and validation must be nonempty")
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("Full fine-tuning requires all parameters trainable")
    generation_every = eval_every if generation_every is None else generation_every
    if generation_every < 0:
        raise ValueError("generation_every must be nonnegative; 0 means checkpoints only")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu = model.device.type == "cuda"
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    train = DataLoader(tokens["train"], batch_size=batch_size, shuffle=True,
                       collate_fn=collator, generator=torch.Generator().manual_seed(seed),
                       pin_memory=gpu)
    validation = DataLoader(tokens["validation"], batch_size=batch_size,
                            collate_fn=collator)
    optimizers, partition = full_optimizers(model, learning_rate, optimizer_name)
    (output_dir / "optimizer_partition.json").write_text(
        json.dumps(partition, indent=2), encoding="utf-8")
    bf16 = gpu and torch.cuda.is_bf16_supported()

    def synchronize():
        if gpu:
            torch.cuda.synchronize(model.device)

    synchronize()
    wall_start = time.perf_counter()
    training_seconds, validation_seconds, step, peak_training_vram = 0.0, 0.0, 0, 0
    loss_seconds, generation_seconds, generation_steps = 0.0, 0.0, []
    history, step_losses, results = [], [], []
    pending_losses, pending_norms = [], []
    segment_start = None
    training_segments = 0
    model.config.use_cache = False

    def collect_losses():
        if not pending_losses:
            return
        values = torch.stack((torch.stack(pending_losses), torch.stack(pending_norms))).cpu()
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite training loss or gradient norm")
        step_losses.extend(values[0].tolist())
        pending_losses.clear()
        pending_norms.clear()

    def finish_training():
        nonlocal training_seconds, peak_training_vram, segment_start
        if segment_start is None:
            return
        collect_losses()
        synchronize()
        training_seconds += time.perf_counter() - segment_start
        if gpu:
            peak_training_vram = max(peak_training_vram,
                                     torch.cuda.max_memory_allocated(model.device))
        segment_start = None

    def info(epoch):
        return {
            "adaptation": "full", "optimizer": optimizer_name, "learning_rate": learning_rate,
            "optimizer_fused": optimizers[0].defaults["fused"],
            "model_revision": getattr(model.config, "_commit_hash", None),
            "epochs": epoch, "batch_size": batch_size, "eval_every_steps": eval_every,
            "generation_every_steps": generation_every, "generation_steps": generation_steps.copy(),
            "optimizer_steps": step, "train_reports": len(tokens["train"]),
            "training_seconds": training_seconds, "validation_seconds": validation_seconds,
            "validation_loss_seconds": loss_seconds, "generation_seconds": generation_seconds,
            "wall_seconds": time.perf_counter() - wall_start,
            "training_scope": "training segments including data collation/logging; excludes validation/save",
            "training_gpu_sync": "training/evaluation boundaries",
            "training_segments": training_segments, "scalar_check_every_steps": 25,
            "pinned_training_batches": gpu, "nonblocking_training_transfers": gpu,
            "peak_memory_reset_scope": "training segment boundaries",
            "training_gpu_resources": torch.cuda.get_device_name(model.device) if gpu else "CPU",
            "training_peak_gpu_vram_mib": peak_training_vram / 2**20 if gpu else None,
            "rocm_version": torch.version.hip, "torch_version": torch.__version__,
            "precision": "bfloat16 autocast" if bf16 else "float32",
            "gradient_checkpointing": model.is_gradient_checkpointing,
            "weight_decay": 0.01, "gradient_clip_norm": 1.0,
            "learning_rate_schedule": "constant", "muon_lr_adjustment": "match_rms_adamw",
            "loss_history": history.copy(), **count_params(model),
        }

    def validate(epoch, generate=False):
        nonlocal validation_seconds, loss_seconds, generation_seconds
        synchronize()
        start = time.perf_counter()
        model.eval()
        evaluation_dir = output_dir / "evaluations" / f"step_{step}"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        if not history or history[-1]["step"] != step:
            total_loss, total_tokens = 0.0, 0
            with torch.inference_mode():
                for batch in validation:
                    batch = {key: value.to(model.device) for key, value in batch.items()}
                    count = int((batch["labels"] != -100).sum())
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                        loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite validation loss")
                    total_loss += loss.item() * count
                    total_tokens += count
            history.append({"epoch": epoch, "step": step,
                            "validation_loss": total_loss / total_tokens})
            synchronize()
            loss_seconds += time.perf_counter() - start
            (evaluation_dir / "validation_loss.json").write_text(
                json.dumps(history[-1], indent=2), encoding="utf-8")
            print(f"Validation loss step={step}: {history[-1]['validation_loss']:.4f}", flush=True)
        metrics = None
        if generate:
            generation_start = time.perf_counter()
            generation_steps.append(step)
            model.config.use_cache = True
            metrics = evaluate_baseline(
                tokenizer, model, reports=reports, limit=eval_limit, seed=seed,
                output_dir=evaluation_dir, training_info=info(epoch),
            )
            synchronize()
            generation_seconds += time.perf_counter() - generation_start
        model.config.use_cache = False
        synchronize()
        validation_seconds += time.perf_counter() - start
        model.train()
        if metrics is not None:
            metrics.update(info(epoch))
            (evaluation_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    for epoch in range(1, max(epoch_options) + 1):
        model.train()
        metrics = None
        for batch in train:
            if segment_start is None:
                synchronize()
                if gpu:
                    torch.cuda.reset_peak_memory_stats(model.device)
                segment_start = time.perf_counter()
                training_segments += 1
            assert not torch.any(batch["labels"] == tokenizer.pad_token_id)
            batch = {key: value.to(model.device, non_blocking=gpu)
                     for key, value in batch.items()}
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                loss = model(**batch).loss
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                 error_if_nonfinite=False)
            for optimizer in optimizers:
                optimizer.step()
            step += 1
            pending_losses.append(loss.detach())
            pending_norms.append(norm.detach())
            if step % 25 == 0:
                collect_losses()
                print(f"{optimizer_name} lr={learning_rate} epoch={epoch} "
                      f"step={step} loss={step_losses[-1]:.4f} "
                      f"train_seconds={training_seconds + time.perf_counter() - segment_start:.1f}", flush=True)
            generation_due = generation_every > 0 and step % generation_every == 0
            metrics = None
            if step % eval_every == 0 or generation_due:
                finish_training()
                metrics = validate(epoch, generate=generation_due)
        finish_training()
        if epoch in epoch_options:
            if metrics is None:
                metrics = validate(epoch, generate=True)
            checkpoint = output_dir / f"epoch_{epoch}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            model.config.use_cache = True
            save_start = time.perf_counter()
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            metadata = info(epoch)
            metadata["checkpoint_save_seconds"] = time.perf_counter() - save_start
            (checkpoint / "training_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            metrics.update(metadata)
            (checkpoint / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            for name in ("predictions", "safety_cases"):
                source = output_dir / "evaluations" / f"step_{step}" / f"{name}.json"
                (checkpoint / f"{name}.json").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            results.append(checkpoint)
            model.config.use_cache = False
            print(f"Saved {checkpoint}; training {training_seconds:.1f}s", flush=True)
    plt.plot(range(1, step + 1), step_losses, label="Training batch loss")
    plt.plot([row["step"] for row in history], [row["validation_loss"] for row in history],
             "o-", label="Validation loss")
    plt.xlabel("Optimizer step")
    plt.ylabel("Token cross-entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()
    model.config.use_cache = True
    model.eval()
    return results


def run_full_experiments(train_limit=8000, eval_limit=150, output_dir=PROJECT_DIR / "outputs" / "full_finetuning",
                         learning_rates=(5e-5, 1e-4), optimizer_names=("adamw", "muon_adamw"),
                         epoch_options=(1, 3), seed=42, batch_size=16,
                         eval_every=500, generation_every=None):
    """Compare selected optimizers, rates, and epoch checkpoints."""
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("AMD ROCm PyTorch is required for measured GPU experiments")
    if not hasattr(torch.optim, "Muon") and "muon_adamw" in optimizer_names:
        raise RuntimeError("Install an AMD PyTorch build with torch.optim.Muon")
    published = load_reports()
    clean = remove_train_overlap(published)
    train_indices = list(range(len(clean["train"])))
    if train_limit is not None:
        train_indices = sorted(random.Random(seed).sample(train_indices, min(train_limit, len(train_indices))))
    val_indices = list(range(len(clean["validation"])))
    if eval_limit is not None:
        val_indices = sorted(random.Random(seed).sample(val_indices, min(eval_limit, len(val_indices))))
    selected = DatasetDict({"train": clean["train"].select(train_indices),
                            "validation": clean["validation"].select(val_indices)})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_audit.json").write_text(json.dumps({
        "train_indices_after_filter": train_indices, "validation_indices": val_indices,
        "removed_train": len(published["train"]) - len(clean["train"]),
        "patient_separation": "unverified; patient IDs absent",
    }, indent=2), encoding="utf-8")
    tokenizer, baseline = load_baseline()
    tokens = get_tokenized_splits(tokenizer, selected, padding=False)
    runs = [output_dir / "baseline"]
    evaluate_baseline(tokenizer, baseline, reports=published, limit=eval_limit,
                      seed=seed, output_dir=runs[0])
    del baseline
    gc.collect()
    torch.cuda.empty_cache()
    for optimizer_name in optimizer_names:
        for learning_rate in learning_rates:
            random.seed(seed)
            torch.manual_seed(seed)
            tokenizer, model = load_full()
            name = f"{optimizer_name}_lr{learning_rate:g}"
            runs.extend(fine_tune(tokenizer, model, published, tokens, output_dir / name,
                                  learning_rate, optimizer_name, epoch_options, seed=seed,
                                  eval_limit=eval_limit, batch_size=batch_size,
                                  eval_every=eval_every, generation_every=generation_every))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    return compare_runs(runs, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Full fine-tuning on AMD ROCm")
    parser.add_argument("--train-limit", type=int, help="0 for all clean training rows")
    parser.add_argument("--eval-limit", type=int, help="0 for full validation")
    parser.add_argument("--epochs", type=int, help="Default: LoRA 1; full sweep 1 and 3")
    parser.add_argument("--optimizer", choices=("adamw", "adamw_torch_fused", "muon_adamw"))
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--eval-every", type=int, default=500, help="Validation loss interval")
    parser.add_argument("--generation-every", type=int, help="ROUGE interval; 0 for checkpoints only")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.full:
        run_full_experiments(
            (8000 if args.train_limit is None else args.train_limit) or None,
            (150 if args.eval_limit is None else args.eval_limit) or None,
            args.output_dir or PROJECT_DIR / "outputs" / "full_finetuning", seed=args.seed,
            batch_size=args.batch_size if args.batch_size is not None else 16,
            learning_rates=(args.learning_rate,) if args.learning_rate is not None else (5e-5, 1e-4),
            optimizer_names=(args.optimizer,) if args.optimizer else ("adamw", "muon_adamw"),
            epoch_options=(args.epochs,) if args.epochs is not None else (1, 3),
            eval_every=args.eval_every, generation_every=args.generation_every,
        )
    else:
        run_experiments((128 if args.train_limit is None else args.train_limit) or None,
                        (20 if args.eval_limit is None else args.eval_limit) or None,
                        args.epochs if args.epochs is not None else 1, args.batch_size or 4, args.seed,
                        args.output_dir or PROJECT_DIR / "outputs" / "lora")
