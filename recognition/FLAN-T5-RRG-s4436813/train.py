"""
COMP3710 Pattern Analysis - Final Assessment Project
Title: Clinical Radiology Report Summarisation with FLAN-T5 and Custom LoRA
Author: Aidan Fassy
Student ID: 44368133
File: train.py
Description: Pure PyTorch distributed training and validation loop supporting
             mixed-precision bfloat16, VRAM profiling, and loss logging.
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from dataset import create_dataloaders
from modules import ClinicalFlanT5, TrainingConstants
from typing import Optional

# ----------------- Constants -----------------

class LearningRates():
    LORA: float = 3e-4
    FINETUNE: float = 3e-5

EPOCHS: int = 1
BATCH_SIZE: int = 8
LR: float = LearningRates.LORA
OUT_DIR: str = "checkpoints"

# ----------------- Convenience functions -----------------

def parse_args() -> argparse.Namespace:
    """
    Small convenience function for command line parsing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=TrainingConstants.LORA,
                        choices=[TrainingConstants.LORA, TrainingConstants.FINETUNE])
    parser.add_argument("--max_steps", default=-1, type=int)
    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, torch.device]:
    """
    Returns (distributed, rank, device). Joins the process group under torchrun.
    """
    if "LOCAL_RANK" not in os.environ:
        return False, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, dist.get_rank(), torch.device("cuda", local_rank)

def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
               optimiser: Optional[torch.optim.Optimizer] = None,
               max_steps: int = -1,
               val_loader: Optional[DataLoader] = None,
               eval_interval: int = 1000) -> tuple[float, list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Runs a pass over the DataLoader.
    If optimiser is provided, trains the model and records step-level train/val loss trajectories.
    Otherwise, evaluates and returns mean loss.
    """
    training: bool = optimiser is not None
    model.train() if training else model.eval()

    total_loss: float = 0.0
    steps: int = 0
    t_history: list[tuple[int, float]] = []
    v_history: list[tuple[int, float]] = []

    # Send everything in the batch loader to GPUs
    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items()
        }

        # Forward pass
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(**batch)
            loss: torch.Tensor = outputs.loss

        # Backpropagation and optimisation
        if training:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            steps += 1
            total_loss += loss.item()

            if steps % 100 == 0:
                t_history.append((steps, loss.item()))

            # Periodic validation evaluation during training pass
            if val_loader is not None and steps % eval_interval == 0:
                val_loss, _, _ = evaluate(model=model, loader=val_loader, device=device, max_steps=100)
                v_history.append((steps, val_loss))
                model.train()
        else:
            total_loss += loss.item()
            steps += 1

        if max_steps > 0 and steps >= max_steps:
            break

    mean_loss = total_loss / max(steps, 1)
    return mean_loss, t_history, v_history

def save_checkpoint(model: nn.Module, mode: str, out: str = OUT_DIR) -> None:
    """
    Saves the adapter weights for LoRA, and the full weights for the fine
    tuning.
    """
    state: dict = model.state_dict()
    os.makedirs(out, exist_ok=True)

    if mode == TrainingConstants.LORA:
        state = {
            k: v for k, v in state.items() if "_lora_" in k
        }
    torch.save(state, os.path.join(out, f"{mode}.pt"))

def plot_loss(t_history: list[tuple[int, float]], v_history: list[tuple[int, float]], mode: str) -> None:
    """
    Saves step-level training and validation loss curves.
    """
    plt.figure(figsize=(8, 5))
    if t_history:
        t_steps, t_losses = zip(*t_history)
        plt.plot(t_steps, t_losses, label="Train Loss", alpha=0.6)
    if v_history:
        v_steps, v_losses = zip(*v_history)
        plt.plot(v_steps, v_losses, label="Val Loss", marker="o", linewidth=2.0)
    plt.xlabel("Optimization Steps")
    plt.ylabel("Loss")
    plt.title(f"{mode.capitalize()} Training Dynamics")
    plt.legend()
    plt.savefig(f"{mode}_loss.png")
    plt.close()

# ----------------- Training & Validations-----------------

def train_single_epoch(model: nn.Module, loader: DataLoader,
                       optimiser: torch.optim.Optimizer, device: torch.device,
                       max_steps: int = -1,
                       val_loader: Optional[DataLoader] = None) -> tuple[float, list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Runs a single epoch of training. Returns the mean training loss and step-level history.
    """
    return run_epoch(model=model, loader=loader, device=device,
                     optimiser=optimiser, max_steps=max_steps, val_loader=val_loader)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
              device: torch.device, max_steps: int = -1) -> tuple[float, list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Validation step, returns the mean validation loss.
    """
    return run_epoch(model=model, loader=loader, device=device,
                     optimiser=None, max_steps=max_steps)

# ----------------- Training Loop -----------------

def main() -> None:
    # Parse initial arguments and grab the loaders
    args = parse_args()
    distributed, rank, device = setup_distributed()
    train_loader, val_full_loader, _, _= create_dataloaders(
        B=BATCH_SIZE, distributed=distributed
    )

    lr = (LearningRates.LORA if args.mode == TrainingConstants.LORA 
          else LearningRates.FINETUNE)

    model = ClinicalFlanT5(mode=args.mode).to(device=device)
    train_step_history: list[tuple[int, float]] = list()
    val_step_history: list[tuple[int, float]]   = list()

    # Rank 0 GPU is the 'leader'
    if not rank:
        trainable, total, pct = model.count_parameters()
        print(f"[{args.mode.upper()}] Trainable: {trainable:,}" 
              + f" / Total: {total:,} ({pct:.2f}%)")

    if distributed:
        model = DDP(model, device_ids=[device.index])

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    for epoch in range(EPOCHS):
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        t_loss, t_hist, v_hist = train_single_epoch(
            model=model, loader=train_loader, optimiser=optimiser,
            device=device, max_steps=args.max_steps, val_loader=val_full_loader
        )
        v_loss, _, _ = evaluate(
            model=model, loader=val_full_loader, 
            device=device, max_steps=args.max_steps
        )

        train_step_history.extend(t_hist)
        val_step_history.extend(v_hist)

        if not rank:
            vram: float = (torch.cuda.max_memory_allocated() / (1024 ** 3) 
                           if torch.cuda.is_available() else 0.0)
            print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {t_loss:.4f} " 
                  + f"| Val Loss: {v_loss:.4f} | Peak VRAM: {vram:.2f} GB")

    # Save artifacts on rank 0
    if not rank:
        raw_model = model.module if distributed else model
        save_checkpoint(raw_model, args.mode, out=OUT_DIR)
        plot_loss(train_step_history, val_step_history, args.mode)

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()