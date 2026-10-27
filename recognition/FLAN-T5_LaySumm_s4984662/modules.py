"""
modules.py - FLAN-T5 builder for the three compared configurations: zero-shot, LoRA, full FT.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
Course : COMP3710 Pattern Recognition, UQ, Semester 2 2026 - Project 2.6
         (FLAN-T5 lay summarisation of radiology reports, LoRA vs full fine-tuning)

Description
    All three configurations start from the same pre-trained FLAN-T5 weights and differ
    only in which parameters are allowed to change:
        zeroshot : nothing is trained (the baseline, evaluated as it is)
        lora     : base frozen; small low-rank adapters on the attention q and v matrices
        full     : every parameter is trained
    Teaching staff confirmed that `transformers` and `peft` may be used to load the model
    and to add LoRA; the training loop and the evaluation are written by hand elsewhere.

Key components (one function each, spec 4.1)
    load_tokenizer    - the FLAN-T5 tokenizer
    build_model       - the model for one of the three modes
    count_parameters  - total and trainable parameter counts (the spec asks for both)
    save_checkpoint / load_checkpoint - store and restore a trained model
    This module does not import NumPy (spec 4.1).

Self-check (a CPU node on Rangpur is enough; loads ~1 GB of weights):
    python modules.py

References
    FLAN-T5: Chung et al., "Scaling instruction-finetuned language models", JMLR 2024.
    LoRA: Hu et al., "LoRA: Low-rank adaptation of large language models", ICLR 2022.
    peft: https://huggingface.co/docs/peft

AI assistance: drafted with Claude (Anthropic) and reviewed by the author, see the README
section "Artificial Intelligence Usage Disclosure".
"""

import argparse
import os
import tempfile
from typing import Dict, Optional, Sequence

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google/flan-t5-base"
MODES = ("zeroshot", "lora", "full")

# Attention query and value projections: the LoRA paper's default. In FLAN-T5-base this
# adapts 72 matrices (12 encoder self + 12 decoder self + 12 decoder cross attention
# blocks, two matrices each).
LORA_TARGETS = ("q", "v")


def load_tokenizer(model_name: str = MODEL_NAME):
    """Return the tokenizer that matches `model_name`."""
    return AutoTokenizer.from_pretrained(model_name)


def _load_flan_t5(path: str):
    """Load a FLAN-T5 checkpoint with encoder/decoder sharing the `shared` embedding.

    A T5 checkpoint stores one input embedding (`shared.weight`) plus, for FLAN-T5, a
    separate output matrix (`lm_head.weight`). In transformers 5.x each stack has its own
    `embed_tokens` that is tied to `shared` only through the tie flag in the config; when
    that flag is False they are silently re-initialised at random (found by the self-check:
    the parameter count was 2 embedding matrices too high). Tying explicitly is
    independent of the flag, and the output matrix `lm_head` is left as stored.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    model.encoder.embed_tokens.weight = model.shared.weight
    model.decoder.embed_tokens.weight = model.shared.weight
    return model


def build_model(mode: str, model_name: str = MODEL_NAME, lora_r: int = 8,
                lora_alpha: Optional[int] = None, lora_dropout: float = 0.05,
                lora_targets: Sequence[str] = LORA_TARGETS):
    """Build FLAN-T5 for `mode` in {"zeroshot", "lora", "full"}.

    Weights are loaded in fp32. Training runs bf16 autocast on top of them, which is safer
    for AdamW than keeping the weights themselves in bf16.

    LoRA replaces a frozen weight W by W + (alpha / r) * B @ A, with A (r x in) and
    B (out x r) the only trained matrices (B starts at 0, so the model starts unchanged).
    The number of trainable parameters is r * (in + out) per adapted matrix: linear in r
    and in the number of matrices. `lora_alpha` defaults to 2 * r (a common convention).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    model = _load_flan_t5(model_name)

    if mode == "full":
        return model                      # every parameter keeps requires_grad = True
    for param in model.parameters():      # zero-shot and the LoRA base are frozen
        param.requires_grad = False
    if mode == "zeroshot":
        return model.eval()

    config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=lora_r,
                        lora_alpha=lora_alpha if lora_alpha is not None else 2 * lora_r,
                        lora_dropout=lora_dropout, target_modules=list(lora_targets))
    return get_peft_model(model, config)  # only the adapter parameters require grad


def count_parameters(model) -> Dict[str, float]:
    """Total and trainable parameters, as the spec asks, plus the % of the base model.

    `total` includes the LoRA adapters when present; the percentage is relative to the
    base model without adapters (0.357% for r=8 on q,v of FLAN-T5-base).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapters = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    return {"total": total, "trainable": trainable, "adapters": adapters,
            "trainable_pct": 100.0 * trainable / (total - adapters)}


def save_checkpoint(model, directory: str) -> None:
    """Save a trained model. LoRA stores only the adapters (a few MB); full FT stores all."""
    os.makedirs(directory, exist_ok=True)
    model.save_pretrained(directory)


def load_checkpoint(mode: str, directory: str, model_name: str = MODEL_NAME):
    """Restore a model written by `save_checkpoint`, frozen and in eval mode."""
    if mode == "full":
        model = _load_flan_t5(directory)
    elif mode == "lora":
        model = PeftModel.from_pretrained(_load_flan_t5(model_name), directory)
    else:
        raise ValueError("only 'lora' and 'full' checkpoints exist; build 'zeroshot' directly")
    return model.eval()


# --------------------------------------------------------------------------------------
# Self-check. Expected LoRA size is derived from the model config so the same check also
# runs on a tiny model; for FLAN-T5-base it must equal the feasibility review (884,736).
# --------------------------------------------------------------------------------------
def _expected_lora_params(model_name: str, rank: int) -> int:
    """LoRA parameters expected for `rank`: r * (d_model + attention width) per q and v matrix."""
    cfg = AutoConfig.from_pretrained(model_name)
    blocks = cfg.num_layers + 2 * cfg.num_decoder_layers   # enc self, dec self, dec cross
    return blocks * len(LORA_TARGETS) * rank * (cfg.d_model + cfg.num_heads * cfg.d_kv)


def _self_check(model_name: str) -> None:
    """Check parameter counts, embedding sharing, LoRA gradients and checkpoint round trips."""
    tokenizer = load_tokenizer(model_name)
    batch = tokenizer(["Summarize: No pneumothorax."], return_tensors="pt")
    labels = tokenizer(["There is no collapsed lung."], return_tensors="pt").input_ids

    def logits(model):
        """Teacher-forced logits of the fixed example batch."""
        with torch.no_grad():
            return model(**batch, labels=labels).logits

    zeroshot = build_model("zeroshot", model_name)
    base = count_parameters(zeroshot)
    assert base["trainable"] == 0 and base["adapters"] == 0, base
    shared = zeroshot.shared.weight
    assert zeroshot.encoder.embed_tokens.weight is shared, "encoder embedding must be shared"
    assert zeroshot.decoder.embed_tokens.weight is shared, "decoder embedding must be shared"
    if model_name == MODEL_NAME or not AutoConfig.from_pretrained(model_name).tie_word_embeddings:
        assert zeroshot.lm_head.weight.data_ptr() != shared.data_ptr(), \
            "FLAN-T5 output matrix must stay separate from the input embedding"
    again = build_model("zeroshot", model_name)       # two loads must give the same weights
    assert all(torch.equal(a, b) for a, b in zip(zeroshot.state_dict().values(),
                                                  again.state_dict().values())), \
        "loading the same checkpoint twice gave different weights (untied/random embeddings?)"
    del again
    assert torch.isfinite(logits(zeroshot)).all()
    if model_name == MODEL_NAME:
        assert base["total"] == 247_577_856, base
    print("zeroshot:", base)

    full = build_model("full", model_name)
    counts = count_parameters(full)
    assert counts["total"] == counts["trainable"] == base["total"], counts
    print("full    :", counts)
    with tempfile.TemporaryDirectory() as tmp:       # save / load round trip
        save_checkpoint(full, tmp)
        assert torch.equal(logits(load_checkpoint("full", tmp, model_name)), logits(full))
    del full

    for rank in (8, 16):
        lora = build_model("lora", model_name, lora_r=rank)
        counts = count_parameters(lora)
        expected = _expected_lora_params(model_name, rank)
        assert counts["trainable"] == counts["adapters"] == expected, (counts, expected)
        assert counts["total"] == base["total"] + expected
        if model_name == MODEL_NAME:
            assert counts["trainable_pct"] < 1.0, "LoRA must stay under the 1% criterion"
        print(f"lora r={rank:<2d}:", counts)
    if model_name == MODEL_NAME:
        assert _expected_lora_params(model_name, 8) == 884_736

    lora = build_model("lora", model_name, lora_r=8)
    lora(**batch, labels=labels).loss.backward()
    graded = [n for n, p in lora.named_parameters() if p.grad is not None]
    assert graded and all("lora_" in n for n in graded), "only adapters may get gradients"
    for name, param in lora.named_parameters():       # make the adapter non-trivial (B=0 at init)
        if "lora_B" in name:
            param.data.normal_(0, 0.02)
    lora.eval()                                       # adapter dropout must be off to compare
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(lora, tmp)
        assert torch.allclose(logits(load_checkpoint("lora", tmp, model_name)), logits(lora))
        size_mb = sum(os.path.getsize(os.path.join(tmp, f)) for f in os.listdir(tmp)) / 1e6
    print(f"lora adapter checkpoint size: {size_mb:.1f} MB")
    print("modules self-check passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-check for modules.py")
    parser.add_argument("--model_name", default=MODEL_NAME)
    _self_check(parser.parse_args().model_name)
