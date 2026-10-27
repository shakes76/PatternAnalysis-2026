"""LoRA/full fine-tuning, rank presets, loss plots, and a small CPU self-test."""

import argparse
import gc
import json
from pathlib import Path
import random
import time

import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from dataset import (PROJECT_DIR, get_tokenized_splits, load_reports,
                     remove_train_overlap, sample_indices)
from modules import MODEL_REVISION, count_params, load_baseline, load_full, load_lora
from predict import compare_runs, evaluate_baseline


def train_model(tokenizer, model, reports, output_dir, *, full=False, epochs=3,
                batch_size=16, learning_rate=0.001, optimizer_name="adamw",
                eval_every=0, seed=42):
    """Train with teacher forcing; generate only after the optimiser is released."""
    if min(epochs, batch_size) < 1 or learning_rate <= 0 or eval_every < 0:
        raise ValueError("Invalid epochs, batch size, learning rate, or evaluation interval")
    if not len(reports["train"]) or not len(reports["validation"]):
        raise ValueError("Training and validation must be nonempty")
    if full != all(p.requires_grad for p in model.parameters()):
        raise ValueError("Model trainability does not match adaptation mode")
    if not full and any("lora_" not in n for n, p in model.named_parameters() if p.requires_grad):
        raise ValueError("Only LoRA parameters may be trainable in adapter mode")
    tokens = get_tokenized_splits(tokenizer, reports, padding=False)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    training = DataLoader(tokens["train"], batch_size=batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(seed), collate_fn=collator)
    validation = DataLoader(tokens["validation"], batch_size=batch_size, collate_fn=collator)
    parameters = [p for p in model.parameters() if p.requires_grad]
    decay = 0.01 if full else 0.0
    groups = [{"params": [p for p in parameters if p.ndim >= 2], "weight_decay": decay},
              {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0}]
    gpu = model.device.type == "cuda"
    bf16 = gpu and torch.cuda.is_bf16_supported()
    if optimizer_name not in ("adamw", "adamw_torch_fused"):
        raise ValueError("Use adamw or adamw_torch_fused")
    if optimizer_name == "adamw_torch_fused" and not gpu:
        raise ValueError("Select adamw for CPU training")
    optimizer = torch.optim.AdamW(groups, lr=learning_rate,
                                 fused=optimizer_name == "adamw_torch_fused", foreach=False)

    def synchronize():
        if gpu:
            torch.cuda.synchronize(model.device)

    def validation_loss():
        model.eval()
        total, count = 0.0, 0
        with torch.inference_mode():
            for batch in validation:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite validation loss")
                size = int((batch["labels"] != -100).sum())
                total += loss.item() * size
                count += size
        model.train()
        return total / count

    model.config.use_cache = False
    synchronize()
    if gpu:
        torch.cuda.reset_peak_memory_stats(model.device)
    start = time.perf_counter()
    history, batch_losses, step = [], [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in training:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            if torch.any(batch["labels"] == tokenizer.pad_token_id):
                raise ValueError("Target padding must be masked with -100")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            size = int((batch["labels"] != -100).sum())
            total += loss.item() * size
            count += size
            batch_losses.append(loss.item())
            if step % 25 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}", flush=True)
            if eval_every and step % eval_every == 0:
                history.append({"epoch": epoch, "step": step, "validation_loss": validation_loss()})
        if history and history[-1]["step"] == step:
            history[-1]["train_loss"] = total / count
        else:
            history.append({"epoch": epoch, "step": step, "train_loss": total / count,
                            "validation_loss": validation_loss()})
    synchronize()
    info = {
        "adaptation": "full" if full else "LoRA", "epochs": epochs,
        "train_reports": len(reports["train"]), "batch_size": batch_size,
        "optimizer_steps": step, "optimizer": optimizer_name, "learning_rate": learning_rate,
        "weight_decay": decay, "gradient_clip_norm": 1.0, "seed": seed,
        "precision": "BF16 forward/loss; FP32 weights/generation" if bf16 else "FP32",
        "gradient_checkpointing": model.is_gradient_checkpointing,
        "max_input_length": 512, "max_target_length": 256,
        "training_seconds": time.perf_counter() - start,
        "training_scope": "optimisation and validation loss; excludes tokenisation, saving and generation",
        "training_gpu_resources": torch.cuda.get_device_name(model.device) if gpu else "CPU",
        "torch_version": torch.__version__, "torch_hip": torch.version.hip,
        "training_peak_gpu_vram_mib": torch.cuda.max_memory_allocated(model.device) / 2**20 if gpu else None,
        "training_parameter_counts": count_params(model), "loss_history": history,
        "implementation": "Consolidated loop; historical timings/schedules are not reproduced exactly",
    }
    if not full:
        config = model.peft_config["default"]
        info["lora"] = {"rank": config.r, "alpha": config.lora_alpha,
                        "dropout": config.lora_dropout, "targets": sorted(config.target_modules)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    model.config.use_cache = True
    model.eval()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "download_revision.txt").write_text(MODEL_REVISION, encoding="utf-8")
    (output_dir / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot(range(1, step + 1), batch_losses, label="Training batch loss", alpha=0.6)
    plt.plot([r["step"] for r in history], [r["validation_loss"] for r in history], "o-", label="Validation loss")
    plt.xlabel("Optimizer step")
    plt.ylabel("Target-token cross-entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()
    return info


def run_experiment(config):
    """Compare a baseline and fresh adaptations on exactly matched held-out rows."""
    output = Path(config["output_dir"])
    if output.exists():
        raise ValueError("Choose a new output directory to preserve existing experiments")
    published = load_reports()
    clean = remove_train_overlap(published)
    train_ids = sample_indices(len(clean["train"]), config["train_limit"], config["seed"])
    val_ids = sample_indices(len(clean["validation"]), config["eval_limit"], config["seed"])
    selected = DatasetDict({"train": clean["train"].select(train_ids),
                            "validation": clean["validation"].select(val_ids)})
    output.mkdir(parents=True)
    (output / "split_audit.json").write_text(json.dumps({
        "train_indices_after_filter": train_ids, "validation_indices": val_ids,
        "removed_train": len(published["train"]) - len(clean["train"]),
        "patient_separation": "Unverified; patient IDs unavailable", "configuration": config,
    }, indent=2), encoding="utf-8")
    tokenizer, model = load_baseline()
    runs = [output / "baseline"]
    evaluate_baseline(tokenizer, model, published, limit=config["eval_limit"],
                      seed=config["seed"], output_dir=runs[0])
    del model
    gc.collect()
    for rank in ([None] if config["full"] else config["ranks"]):
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tokenizer, model = load_full() if rank is None else load_lora(rank, 2 * rank, 0.05)
        run_dir = output / ("full" if rank is None else f"rank{rank}")
        checkpoint = run_dir / ("checkpoint" if rank is None else "adapter")
        info = train_model(tokenizer, model, selected, checkpoint, **{
            key: config[key] for key in ("full", "epochs", "batch_size", "learning_rate",
                                        "optimizer_name", "eval_every", "seed")})
        runs.append(run_dir)
        evaluate_baseline(tokenizer, model, published, limit=config["eval_limit"],
                          seed=config["seed"], output_dir=run_dir, training_info=info)
        del model
        gc.collect()
    return compare_runs(runs, output / "comparison")


def self_test():
    """Real CPU optimisation/reload on a tiny synthetic model; no downloads needed."""
    import tempfile
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration
    from predict import predict

    vocabulary = {word: i for i, word in enumerate(["<pad>", "</s>", "<unk>", "No", "fluid", "."])}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                       eos_token="</s>", unk_token="<unk>",
                                       model_input_names=["input_ids", "attention_mask"])
    def rows(texts, images):
        return Dataset.from_dict({"source": ["synthetic"] * len(texts), "images_path": images,
                                  "radiology_report": texts, "layman_report": ["No fluid ."] * len(texts)})
    reports = DatasetDict({"train": rows(["train one", " SAME ", "id match", "long training report"], ["a", "b", "held", "c"]),
                           "validation": rows(["same"], ["v"]),
                           "test": rows(["test one", "test two", "test three"], ["held", "t2", "t3"])})
    reports["test"] = reports["test"].map(lambda row: {"layman_report": ""})
    clean = remove_train_overlap(reports)
    assert len(clean["train"]) == 2
    tokens = get_tokenized_splits(tokenizer, clean, max_input_length=32, max_target_length=8)
    assert -100 in tokens["train"][0]["labels"] and "labels" not in tokens["test"].column_names
    selected = DatasetDict({k: clean[k] for k in ("train", "validation")})
    cfg = T5Config(vocab_size=len(tokenizer), d_model=16, d_kv=8, d_ff=32,
                   num_layers=1, num_decoder_layers=1, num_heads=2,
                   decoder_start_token_id=0, eos_token_id=1, pad_token_id=0)
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base = T5ForConditionalGeneration(cfg)
        base.save_pretrained(root / "base")
        model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                    r=2, lora_alpha=4, target_modules=["q", "v"], lora_dropout=0.05))
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        train_model(tokenizer, model, selected, root / "adapter", epochs=1, batch_size=2)
        assert all(torch.equal(before[n], p) for n, p in model.named_parameters() if not p.requires_grad)
        assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
        reloaded = PeftModel.from_pretrained(T5ForConditionalGeneration.from_pretrained(root / "base"), root / "adapter")
        reloaded.eval()
        assert predict("No fluid .", tokenizer, model, max_new_tokens=4) == predict("No fluid .", tokenizer, reloaded, max_new_tokens=4)
        train_model(tokenizer, reloaded.merge_and_unload().requires_grad_(True), selected,
                    root / "full", full=True, epochs=1, batch_size=2)
        full = T5ForConditionalGeneration.from_pretrained(root / "full")
        (root / "evaluation/adapter").mkdir(parents=True)
        metrics = evaluate_baseline(tokenizer, full, reports, limit=1, max_new_tokens=4,
                                    safety_cases=3, output_dir=root / "evaluation",
                                    training_info={"training_seconds": 1, "generation_seconds": 999})
        assert metrics["evaluated_reports"] == 1 and metrics["generation_seconds"] > 0
        assert metrics["generation_seconds"] != 999 and metrics["training"]["generation_seconds"] == 999
        compare_runs([root / "evaluation"], root / "comparison")
        import shutil
        shutil.copytree(root / "evaluation", root / "mismatch")
        modified = json.loads((root / "mismatch/metrics.json").read_text())
        modified["seed"] = 99
        (root / "mismatch/metrics.json").write_text(json.dumps(modified))
        try:
            compare_runs([root / "evaluation", root / "mismatch"], root / "bad")
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched evaluation settings were accepted")
    print("Self-test passed: filtering, masks, LoRA/full updates, reload, evaluation and mismatch rejection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Update all base parameters")
    parser.add_argument("--preset", choices=("pilot", "sweep", "full"), default="pilot")
    parser.add_argument("--ranks", type=int, nargs="+")
    parser.add_argument("--train-limit", type=int, help="0 means all eligible rows")
    parser.add_argument("--eval-limit", type=int, help="0 means full validation")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--optimizer", choices=("adamw", "adamw_torch_fused"))
    parser.add_argument("--eval-every", type=int, help="0 means epoch checks only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "experiment")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.self_test:
        self_test()
        return
    pilot = args.preset == "pilot"
    train_limit = args.train_limit if args.train_limit is not None else (128 if pilot else 8000 if args.preset == "sweep" else 0)
    eval_limit = args.eval_limit if args.eval_limit is not None else (20 if pilot else 150)
    config = {"full": args.full, "ranks": args.ranks or ([2, 4, 8, 16, 32, 64] if args.preset == "sweep" else [8 if pilot else 64]),
              "train_limit": train_limit or None, "eval_limit": eval_limit or None,
              "epochs": args.epochs if args.epochs is not None else (1 if pilot else 3),
              "batch_size": args.batch_size if args.batch_size is not None else (4 if pilot else 16),
              "learning_rate": args.learning_rate if args.learning_rate is not None else (1e-4 if args.full else 1e-3),
              "optimizer_name": args.optimizer or ("adamw_torch_fused" if args.full and torch.cuda.is_available() else "adamw"),
              "eval_every": args.eval_every if args.eval_every is not None else (500 if args.full else 0),
              "seed": args.seed, "output_dir": str(args.output_dir)}
    if min(train_limit, eval_limit) < 0 or min(config["epochs"], config["batch_size"], *config["ranks"]) < 1 or config["learning_rate"] <= 0 or config["eval_every"] < 0:
        parser.error("Limits must be nonnegative; epochs, batch size, ranks and LR positive")
    print(json.dumps(config, indent=2))
    if not args.dry_run:
        run_experiment(config)


if __name__ == "__main__":
    main()
