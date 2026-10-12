"""Pretrained FLAN-T5 baseline and parameter counts."""

from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "google/flan-t5-base"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "flan-t5-base"


def load_baseline(model_name=None, device=None):
    if model_name is None:
        model_name = str(MODEL_DIR) if MODEL_DIR.exists() else MODEL_ID
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    revision_file = Path(model_name) / "download_revision.txt"
    if revision_file.exists():
        model.config._commit_hash = revision_file.read_text(encoding="utf-8-sig").strip()
    model.eval()
    return tokenizer, model


def count_params(model):
    """Evaluation mode does not change requires_grad."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
        "trainable_layers": [name for name, layer in model.named_modules()
                             if any(p.requires_grad for p in
                                    layer.parameters(recurse=False))],
    }


def load_lora(rank=8, alpha=16, dropout=0.05, adapter_path=None, device=None):
    """Train query/value adapters; requires peft==0.15.2."""
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    tokenizer, model = load_baseline(device=device)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
    else:
        config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=rank, lora_alpha=alpha,
            lora_dropout=dropout, target_modules=["q", "v"], bias="none",
        )
        model = get_peft_model(model, config)
        model.train()
    return tokenizer, model
