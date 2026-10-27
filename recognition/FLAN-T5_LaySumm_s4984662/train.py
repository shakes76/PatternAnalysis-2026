"""
train.py - train, validate, test and save FLAN-T5 for lay summarisation (zero-shot, LoRA, full FT).

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
Course : COMP3710 Pattern Recognition, UQ, Semester 2 2026 - Project 2.6
         (FLAN-T5 lay summarisation of radiology reports, LoRA vs full fine-tuning)

Description
    One script runs all three configurations through the same pipeline, so they differ only
    in `--mode`:
        zeroshot : no training, the untuned baseline
        lora     : base frozen, low-rank adapters trained
        full     : every parameter trained
    The training loop and the evaluation are written by hand (no Hugging Face Trainer):
    zero_grad -> forward -> loss -> backward -> clip -> step, as in the lectures (W6).

    Data (dataset.py): training uses the official train split MINUS a 1,000-row development
    set cut from train by whole study. The development set is used each epoch for the loss,
    ROUGE and checkpoint choice, so the official validation split is never used to tune
    anything. After training, the best checkpoint is reloaded and evaluated ONCE on the
    official validation split; this final evaluation is the "testing" step (the official test
    split has no reference summaries). ROUGE is reported on three tiers: full validation,
    unseen-report and clean (see dataset.py).

    Loss: cross-entropy of the decoder logits against the lay summary, positions labelled -100
    (padding) ignored, logits cast to fp32 and NOT passed through softmax (the loss does it).
    Optimiser: AdamW, linear warm-up then linear decay, gradient clipping, bf16 autocast on GPU
    with fp32 weights. Checkpoint choice: the epoch with the highest development ROUGE-Lsum;
    only that one checkpoint is kept (home quota).
    Memory: T5 attention memory grows with the square of the padded length and a few reports
    reach the 512-token cap, so a long batch decides the peak. `--grad_accum N` splits one
    optimiser step into N micro-batches (gradients are summed) to cut the peak at the same
    effective batch; the loss is the mean over tokens of each micro-batch, averaged.

    Distributed: run with `torchrun --standalone --nproc_per_node=N train.py ...`. With one
    process it behaves as a plain script; with N > 1 each process holds a model copy (DDP) and
    only rank 0 evaluates and writes files. Rangpur nodes have one GPU, so N = 1 there.

    With --skip_final_eval the validation split is never touched: only the development set
    is evaluated (used to compare learning rates before the real runs).

Outputs (in `--output_dir`, git-ignored): best/ (checkpoint), curves.png, metrics.json,
    val_predictions.jsonl (every validation prediction with its ROUGE scores, for the autopsy).

Usage
    python train.py --mode zeroshot
    python train.py --mode lora --lora_r 8 --lr 3e-4
    torchrun --standalone --nproc_per_node=1 train.py --mode full --lr 5e-5
    (slurm/train.sh uses explicit loopback flags instead of --standalone: on a laptop whose
    host name does not resolve, --standalone hangs)
    python train.py --mode lora --lr 1e-4 --epochs 1 --max_train_samples 20000 \
        --skip_final_eval --output_dir runs/sweep_lora_lr1e-4     # one learning-rate trial
    python train.py --self_check          # quick checks of loss, schedule and one optimiser step

References
    FLAN-T5: Chung et al. (2024); LoRA: Hu et al. (2022); AdamW: Loshchilov & Hutter (2019).
    The linear warm-up/decay schedule (`lr_factor`) re-implements the common "linear schedule
    with warm-up" (the idea of Hugging Face get_linear_schedule_with_warmup; no code copied).
    AEHRC used AdamW with lr 2e-5 for full fine-tuning of T5-large (BioLaySumm 2025).

AI assistance: drafted with Claude (Anthropic) and reviewed by the author, see the README
section "Artificial Intelligence Usage Disclosure".
"""

import argparse
import json
import os
import random
import socket
import subprocess
import time
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import (IGNORE_INDEX, TARGET_COLUMN, LaySummDataset, evaluation_tiers,
                     load_laysumm, make_collate_fn, make_dev_split)
from modules import MODEL_NAME, build_model, count_parameters, load_checkpoint, load_tokenizer, \
    save_checkpoint
from utils import (ROUGE_TYPES, compute_rouge, generate_summaries, mean_rouge, peak_vram_gib,
                   reset_peak_vram, rouge_per_example)

# Starting learning rates before the small sweep on the development set. LoRA needs a larger
# rate than full fine-tuning because it trains few parameters from a zero-initialised update.
DEFAULT_LR = {"lora": 3e-4, "full": 5e-5}


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line options; fills in the default learning rate and output folder."""
    p = argparse.ArgumentParser(description="Train / evaluate FLAN-T5 lay summarisation")
    p.add_argument("--mode", choices=("zeroshot", "lora", "full"))
    p.add_argument("--model_name", default=MODEL_NAME)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=None, help="default depends on --mode")
    p.add_argument("--lora_r", type=int, default=8, help="LoRA rank (alpha = 2 * r)")
    p.add_argument("--batch_size", type=int, default=16, help="micro-batch per process")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="micro-batches per optimiser step (effective batch = batch_size x "
                        "grad_accum x processes); lowers peak memory at equal effective batch")
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_beams", type=int, default=4, help="decoding, same for every mode")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers; tokenising is cheap, >0 needs fork (Linux)")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_train_samples", type=int, default=None, help="random subset (debug)")
    p.add_argument("--max_eval_samples", type=int, default=None,
                   help="first N dev / validation rows (debug)")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--skip_final_eval", action="store_true",
                   help="train and choose on dev only; never touch the validation split "
                        "(for learning-rate sweeps)")
    p.add_argument("--self_check", action="store_true")
    args = p.parse_args(argv)
    if not args.self_check and args.mode is None:
        p.error("--mode is required")
    if args.skip_final_eval and args.mode == "zeroshot":
        p.error("--skip_final_eval makes no sense for --mode zeroshot (nothing is trained)")
    if args.lr is None and args.mode in DEFAULT_LR:
        args.lr = DEFAULT_LR[args.mode]
    if args.output_dir is None and args.mode:
        tag = {"zeroshot": "zeroshot", "full": "full", "lora": f"lora_r{args.lora_r}"}[args.mode]
        if args.lr is not None:
            tag += f"_lr{args.lr:g}"
        args.output_dir = os.path.join("runs", tag)
    return args


# --------------------------------------------------------------------------------------
# Small pieces: seeds, distributed set-up, loss, schedule
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed Python and PyTorch (CPU and GPU) for repeatable runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed() -> Tuple[int, int, torch.device]:
    """Read the variables torchrun sets (RANK, WORLD_SIZE, LOCAL_RANK); return rank, world, device.
    """
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return rank, world, device


def token_loss(model, batch: Dict[str, torch.Tensor], device: torch.device,
               reduction: str = "mean") -> Tuple[torch.Tensor, int]:
    """Cross-entropy of the decoder logits against `labels`; returns (loss, number of tokens).

    Passing `labels` makes the model build the decoder inputs by shifting the labels right
    (teacher forcing). The loss is computed here from the logits rather than taken from the
    model, so that every step of the maths is visible; the self-check shows it equals the
    model's own loss. -100 positions (padding) are skipped by `ignore_index`.
    """
    labels = batch["labels"].to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device), labels=labels).logits
    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1),
                           ignore_index=IGNORE_INDEX, reduction=reduction)
    return loss, int((labels != IGNORE_INDEX).sum())


@torch.no_grad()
def dev_loss(model, loader: DataLoader, device: torch.device) -> float:
    """Mean per-token cross-entropy over the whole loader."""
    model.eval()
    total, tokens = 0.0, 0
    for batch in loader:
        loss_sum, n = token_loss(model, batch, device, reduction="sum")
        total, tokens = total + float(loss_sum), tokens + n
    return total / tokens


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warm-up to 1, then linear decay to 0 (multiplies the base learning rate)."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def git_info() -> Dict[str, object]:
    """Commit and dirty flag of this script's repository, for reproducibility."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        run = lambda *cmd: subprocess.check_output(cmd, cwd=here, text=True,  # noqa: E731
                                                   stderr=subprocess.DEVNULL).strip()
        return {"commit": run("git", "rev-parse", "HEAD"),
                "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}
    except Exception:  # not a git checkout (for example code copied with rsync)
        return {"commit": "unknown", "dirty": None}


def library_versions() -> Dict[str, object]:
    """Versions that can change a number (kept with the results; the platform may vary)."""
    import platform

    import peft
    import transformers
    return {"python": platform.python_version(), "transformers": transformers.__version__,
            "peft": peft.__version__, "cuda": torch.version.cuda}


def plot_curves(history: Dict[str, list], path: str) -> None:
    """Left: train loss (mean per logging window) and dev loss. Right: dev ROUGE per epoch."""
    import matplotlib
    matplotlib.use("Agg")                       # no display on compute nodes
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(history["train_step"], history["train_loss"], label="train (logging window)")
    left.plot(history["dev_step"], history["dev_loss"], "o-", label="dev")
    left.set(xlabel="optimiser step", ylabel="cross-entropy per token", title="Loss")
    left.legend()
    for name in ROUGE_TYPES:
        right.plot(history["dev_epoch"], [r[name] for r in history["dev_rouge"]], "o-", label=name)
    right.set(xlabel="epoch (0 = before training)", ylabel="ROUGE F1 x100",
              title="Development ROUGE")
    right.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def run_training(args, model, tokenizer, splits, rank: int, world: int, device: torch.device,
                 out_dir: str) -> Dict[str, object]:
    """Fine-tune `model` in place; rank 0 evaluates on the dev set and keeps the best checkpoint."""
    train_idx, dev_idx = make_dev_split(splits["train"])
    if args.max_train_samples:
        train_idx = random.Random(args.seed).sample(train_idx, min(args.max_train_samples,
                                                                   len(train_idx)))
    if args.max_eval_samples:
        dev_idx = dev_idx[:args.max_eval_samples]
    train_set = LaySummDataset(splits["train"].select(train_idx), tokenizer, augment=True)
    dev_part = splits["train"].select(dev_idx)
    dev_set = LaySummDataset(dev_part, tokenizer)
    dev_refs = list(dev_part[TARGET_COLUMN])
    collate = make_collate_fn(tokenizer.pad_token_id)
    sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed) if world > 1 else None
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=sampler is None,
                        sampler=sampler, collate_fn=collate, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_set, batch_size=args.eval_batch_size, collate_fn=collate)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = -(-len(loader) // args.grad_accum)    # optimiser steps, rounded up
    total_steps = args.epochs * steps_per_epoch
    warmup = max(1, int(args.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total_steps, warmup))
    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) \
        if world > 1 else model

    history: Dict[str, list] = {"train_step": [], "train_loss": [], "dev_step": [], "dev_loss": [],
                                "dev_epoch": [], "dev_rouge": []}
    best = {"epoch": 0, "rougeLsum": -1.0}

    def evaluate_dev(epoch: int) -> None:
        """Dev loss and ROUGE after `epoch` (rank 0); keep the checkpoint if it is the best."""
        if rank == 0:
            loss = dev_loss(model, dev_loader, device)
            texts, _ = generate_summaries(model, tokenizer, dev_set, args.eval_batch_size,
                                          args.num_beams, args.max_new_tokens)
            rouge = compute_rouge(texts, dev_refs)
            history["dev_step"].append(epoch * steps_per_epoch)
            history["dev_loss"].append(loss)
            history["dev_epoch"].append(epoch)
            history["dev_rouge"].append(rouge)
            print(f"[dev] epoch {epoch}: loss {loss:.4f} | "
                  + " ".join(f"{k} {rouge[k]:.2f}" for k in ROUGE_TYPES), flush=True)
            if epoch >= 1 and rouge["rougeLsum"] > best["rougeLsum"]:
                best.update(epoch=epoch, rougeLsum=rouge["rougeLsum"])
                save_checkpoint(model, os.path.join(out_dir, "best"))
        if world > 1:
            dist.barrier()

    evaluate_dev(0)                                   # the untrained starting point
    train_seconds, train_peak, step = 0.0, None, 0
    for epoch in range(1, args.epochs + 1):
        train_set.set_epoch(epoch - 1)                # next prompt variant (dataset.py)
        if sampler is not None:
            sampler.set_epoch(epoch)
        ddp.train()
        reset_peak_vram()
        start, window = time.perf_counter(), []
        for i, batch in enumerate(loader):
            if i % args.grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)
            loss, _ = token_loss(ddp, batch, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            (loss / args.grad_accum).backward()      # gradients add up over the micro-batches
            window.append(loss.item())
            if (i + 1) % args.grad_accum and i + 1 < len(loader):
                continue                              # keep accumulating
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            if rank == 0 and step % args.log_every == 0:
                mean = sum(window) / len(window)
                history["train_step"].append(step)
                history["train_loss"].append(mean)
                print(f"epoch {epoch} step {step}/{total_steps} loss {mean:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
                window = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - start
        peak = peak_vram_gib()                   # None without a GPU
        if peak is not None:
            train_peak = max(train_peak or 0.0, peak)
        evaluate_dev(epoch)
    return {"seconds": train_seconds, "peak_vram_gib": train_peak, "epochs": args.epochs,
            "steps": step, "best_epoch": best["epoch"], "best_dev_rougeLsum": best["rougeLsum"],
            "trainable_params": sum(p.numel() for p in trainable), "history": history}


# --------------------------------------------------------------------------------------
# Final evaluation on the official validation split (the "test" of this project)
# --------------------------------------------------------------------------------------
def evaluate_validation(args, model, tokenizer, splits, out_dir: str) -> Dict[str, object]:
    """Generate once for the validation rows, score every pair, average per tier."""
    validation = splits["validation"]
    n = min(args.max_eval_samples or len(validation), len(validation))
    part = validation.select(range(n))
    tiers = {name: [i for i in idx if i < n]
             for name, idx in evaluation_tiers(splits["train"], validation).items()}
    dataset = LaySummDataset(part, tokenizer, with_targets=False)
    texts, stats = generate_summaries(model, tokenizer, dataset, args.eval_batch_size,
                                      args.num_beams, args.max_new_tokens)
    rows = rouge_per_example(texts, list(part[TARGET_COLUMN]))
    with open(os.path.join(out_dir, "val_predictions.jsonl"), "w") as f:
        for i, (text, row) in enumerate(zip(texts, rows)):
            f.write(json.dumps({"index": i, "prediction": text, **row}) + "\n")
    scores = {name: {"n": len(idx), **mean_rouge(rows, idx)} for name, idx in tiers.items() if idx}
    return {"rouge": scores, "inference": stats}


def main(argv=None) -> None:
    """Train (unless zero-shot), then evaluate once on the official validation split
    (skipped with --skip_final_eval)."""
    args = parse_args(argv)
    if args.self_check:
        return _self_check(args.model_name)
    rank, world, device = setup_distributed()
    set_seed(args.seed)
    out_dir = args.output_dir
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    tokenizer = load_tokenizer(args.model_name)
    splits = load_laysumm()
    model = build_model(args.mode, args.model_name, lora_r=args.lora_r).to(device)
    parameters = count_parameters(model)         # before training and before any reload
    print(f"mode {args.mode} | params {parameters} | device {device} | world {world}",
          flush=True)

    training = None
    if args.mode != "zeroshot" and args.epochs > 0:
        training = run_training(args, model, tokenizer, splits, rank, world, device, out_dir)
    if world > 1:                       # only rank 0 continues: no idle rank waits at a barrier
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return

    result = None
    if training is not None:
        plot_curves(training["history"], os.path.join(out_dir, "curves.png"))
    if not args.skip_final_eval:
        if training is not None:        # evaluate the saved best checkpoint, not the last epoch
            best_dir = os.path.join(out_dir, "best")
            model = load_checkpoint(args.mode, best_dir, args.model_name).to(device)
        result = evaluate_validation(args, model, tokenizer, splits, out_dir)
        for name, r in result["rouge"].items():
            print(f"[validation:{name}] n={r['n']} "
                  + " ".join(f"{k} {r[k]:.2f}" for k in ROUGE_TYPES))
        print("inference:", {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in result["inference"].items()})
    metrics = {"args": vars(args), "git": git_info(), "host": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "world_size": world, "torch": torch.__version__, "versions": library_versions(),
               "parameters": parameters, "training": training,
               "validation": result}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("wrote", os.path.join(out_dir, "metrics.json"))


# --------------------------------------------------------------------------------------
# Self-check (no dataset download needed): loss, schedule, and that training reduces the loss
# --------------------------------------------------------------------------------------
def _self_check(model_name: str) -> None:
    """Check the loss, the schedule and that the training steps reduce the loss."""
    tokenizer = load_tokenizer(model_name)
    split = {"radiology_report": ["No pneumothorax.", "Mild bibasilar atelectasis. No effusion.",
                                  "Heart size is normal.", "Right lower lobe opacity."],
             "layman_report": ["There is no collapsed lung.", "Small areas at the bottom of both "
                               "lungs are not fully open. No fluid.", "The heart is a normal size.",
                               "There is a cloudy area in the lower right lung."]}
    dataset = LaySummDataset(split, tokenizer)
    batch = make_collate_fn(tokenizer.pad_token_id)([dataset[i] for i in range(4)])
    cpu = torch.device("cpu")

    # 1. our cross-entropy equals the model's own loss (so ignoring -100 is done right)
    model = build_model("full", model_name)
    model.eval()
    with torch.no_grad():
        ours, tokens = token_loss(model, batch, cpu)
        theirs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["labels"]).loss
    assert abs(float(ours) - float(theirs)) < 1e-4, (float(ours), float(theirs))
    assert tokens == int((batch["labels"] != IGNORE_INDEX).sum())
    assert tokens < batch["labels"].numel(), "the batch must contain padding"

    # 2. schedule: warm-up ramps to 1, decay reaches 0
    factors = [lr_factor(s, 100, 10) for s in range(100)]
    assert factors[0] == 0.1 and factors[9] == 1.0 and factors[10] == 1.0
    assert abs(factors[99] - 1 / 90) < 1e-12
    assert all(a >= b for a, b in zip(factors[10:], factors[11:]))

    # 3. the loop's order of operations drives the loss down on a fixed batch (full and LoRA)
    for mode in ("full", "lora"):
        model = build_model(mode, model_name)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-3 if mode == "full" else 1e-2)
        first = None
        for _ in range(25):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = token_loss(model, batch, cpu)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            first = loss.item() if first is None else first
        last = loss.item()
        assert last < first, (mode, first, last)
        print(f"{mode}: loss {first:.3f} -> {last:.3f} in 25 steps")
    print("train self-check passed")


if __name__ == "__main__":
    main()
