"""
env_check.py - sanity check of the Rangpur GPU environment (run through slurm/env_check.sh).

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
Course : COMP3710 Pattern Recognition, UQ, Semester 2 2026 - Project 2.6

Prints library versions, the GPU model and VRAM, and loads FLAN-T5-base from the
local Hugging Face cache (offline) to confirm that:
  * CUDA works on the A100 node;
  * the model loads and gives a finite loss in bf16 and fp16 (a one-sentence smoke
    test only: it does NOT prove that fp16 training is stable);
  * the parameter count matches the feasibility review (247,577,856) and that
    shared.weight and lm_head.weight are separate tensors (FLAN-T5 is untied).

AI assistance: drafted with Claude (Anthropic) and reviewed by the author, see the
README section "Artificial Intelligence Usage Disclosure".
"""

import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google/flan-t5-base"
SENTENCE = "Summarize in plain language: No pneumothorax. Mild bibasilar atelectasis."


def main() -> None:
    """Print the environment, then test the model in bf16 and fp16 on the GPU."""
    print("python", sys.version.split()[0], "| torch", torch.__version__,
          "| cuda", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | total VRAM GiB: {props.total_memory / 2**30:.1f}"
          f" | bf16 supported: {torch.cuda.is_bf16_supported()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    batch = tokenizer(SENTENCE, return_tensors="pt").to("cuda")
    for dtype in (torch.bfloat16, torch.float16):
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cuda", dtype=dtype).eval()
        with torch.no_grad():
            loss = model(**batch, labels=batch.input_ids).loss
        total = sum(p.numel() for p in model.parameters())
        untied = model.lm_head.weight.data_ptr() != model.shared.weight.data_ptr()
        print(f"{dtype}: loss finite = {bool(torch.isfinite(loss))} | params = {total:,}"
              f" | lm_head untied = {untied}"
              f" | peak VRAM GiB = {torch.cuda.max_memory_allocated() / 2**30:.2f}")
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
