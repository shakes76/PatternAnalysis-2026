"""
COMP3710 Pattern Analysis - Final Assessment Project
Title: Clinical Radiology Report Summarisation with FLAN-T5 and Custom LoRA
Author: Aidan Fassy
Student ID: 44368133
File: predict.py
Description: Autoregressive beam search generation, ROUGE metric benchmarking,
             inference throughput telemetry, and clinical safety autopsy extraction.
"""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple
import evaluate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from dataset import (
    IGNORE_INDEX,
    MODEL,
    SOURCE_COL,
    TARGET_COL,
    TASK_PREFIX,
    create_dataloaders,
    load_and_split_dataset,
)
from modules import ClinicalFlanT5, TrainingConstants

# ------------------ Helpers ------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FLAN-T5 RRG Model")
    parser.add_argument(
        "--mode",
        default=TrainingConstants.LORA,
        choices=[
            TrainingConstants.ZERO_SHOT,
            TrainingConstants.LORA,
            TrainingConstants.FINETUNE,
        ],
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100, help="-1 for full split")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def load_eval_model(mode: str, checkpoint_path: Optional[str],
                     device: torch.device) -> ClinicalFlanT5:
    """Initializes model and loads weights if checkpoint is provided."""
    model = ClinicalFlanT5(mode=mode).to(device)

    # Resolve default checkpoint path if not explicitly provided
    if checkpoint_path is None and mode != TrainingConstants.ZERO_SHOT:
        checkpoint_path = os.path.join("checkpoints", f"{mode}.pt")

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        is_lora = mode == TrainingConstants.LORA
        model.load_state_dict(state, strict=not is_lora)
    elif mode != TrainingConstants.ZERO_SHOT:
        print(f"No checkpoint found at '{checkpoint_path}'. Evaluating untuned backbone.")

    model.eval()
    return model


def evaluate_split(model: ClinicalFlanT5, loader: DataLoader,
                    tokeniser: PreTrainedTokenizerBase, device: torch.device,
                      max_samples: int = -1,) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
    """Generates summaries, computes ROUGE metrics, and measures throughput."""
    rouge = evaluate.load("rouge")
    all_preds: List[str] = list()
    all_refs: List[str] = list()
    all_sources: List[str] = list()

    total_tokens: int = 0
    start_time = time.perf_counter()
    sample_count: int = 0

    if torch.cuda.is_available():
      torch.cuda.reset_peak_memory_stats(device)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].clone()

        # Generate summary predictions
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                num_beams=2,
            )

        total_tokens += gen_tokens.numel()

        # Decode predictions
        decoded_preds = tokeniser.batch_decode(gen_tokens, skip_special_tokens=True)

        # Replace -100 in labels so tokenizer can decode references
        labels[labels == IGNORE_INDEX] = tokeniser.pad_token_id
        decoded_refs = tokeniser.batch_decode(labels, skip_special_tokens=True)
        decoded_src = tokeniser.batch_decode(input_ids, skip_special_tokens=True)

        all_preds.extend(decoded_preds)
        all_refs.extend(decoded_refs)
        all_sources.extend(decoded_src)

        sample_count += input_ids.size(0)
        if 0 < max_samples <= sample_count:
            break

    elapsed_time = time.perf_counter() - start_time
    tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0.0
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    # Compute ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum
    results: Dict[str, float] = rouge.compute(
        predictions=all_preds, references=all_refs, use_stemmer=True
    )

    results["throughput_tok_per_sec"] = tokens_per_sec
    results["peak_vram_gb"] = peak_vram

    samples = [
        {"source": src, "reference": ref, "prediction": pred}
        for src, ref, pred in zip(all_sources, all_refs, all_preds)
    ]

    return results, samples

def print_clinical_autopsy(samples: List[Dict[str, str]], max_cases: int = 4) -> None:
    """Prints qualitative clinical safety autopsy cases for error analysis."""
    print("\n" + "=" * 60)
    print("CLINICAL SAFETY AUTOPSY (AUDIT CASES)")
    print("=" * 60)

    keywords = ["pneumothorax", "nodule", "fracture", "effusion", "consolidation", "mass"]
    displayed: int = 0

    # Prioritize cases containing acute pathology keywords
    for item in samples:
        src = item["source"].lower()
        if any(kw in src for kw in keywords) or not displayed:
            displayed += 1
            print(f"\n--- Case {displayed} ---")
            print(f"[RADIOLOGY REPORT]:\n{item['source'].replace(TASK_PREFIX, '').strip()}")
            print(f"\n[GROUND TRUTH LAY SUMMARY]:\n{item['reference'].strip()}")
            print(f"\n[MODEL PREDICTION]:\n{item['prediction'].strip()}")
            print("-" * 50)
            if displayed >= max_cases:
                break

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device} | Mode: {args.mode}")

    model = load_eval_model(args.mode, args.checkpoint, device)
    tokeniser = AutoTokenizer.from_pretrained(MODEL)

    _, val_full_loader, val_unseen_loader, _ = create_dataloaders(
        B=args.batch_size, distributed=False
    )

    print("\nEvaluating on Val_Full...")
    full_metrics, _ = evaluate_split(model, val_full_loader, tokeniser, device, args.num_samples)
    for k, v in full_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nEvaluating on Val_Unseen (deduplicated)...")
    unseen_metrics, samples = evaluate_split(model, val_unseen_loader, tokeniser, device, args.num_samples)
    for k, v in unseen_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print_clinical_autopsy(samples, max_cases=4)

if __name__ == "__main__":
    main()