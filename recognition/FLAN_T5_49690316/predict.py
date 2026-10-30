"""Load a trained full/LoRA checkpoint and explain supplied or held-out reports.

Uses the Transformers pretrained/generate APIs and the PEFT adapter loader in
modules.py. No weights are updated. See README references for those libraries.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import INSTRUCTION
from modules import RadiologySummarizer


PROJECT_DIR = Path(__file__).resolve().parent


def select_device(requested):
    """Select CUDA, Apple MPS or CPU; reject unavailable explicit GPU choices."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use an allocated GPU job or --device cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable")
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="full_all",
                        help="Folder under checkpoints/ or an absolute checkpoint path")
    supplied = parser.add_mutually_exclusive_group()
    supplied.add_argument("--text", help="One report to explain")
    supplied.add_argument("--input-file", type=Path, help="UTF-8 text file containing one report")
    parser.add_argument("--split", choices=["development", "final"], default="final",
                        help="Used only when neither --text nor --input-file is supplied")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, help="Optional new JSON output file")
    args = parser.parse_args()

    if args.samples <= 0 or args.max_new_tokens <= 0:
        parser.error("Sample count and output token limit must be positive")
    if args.output and args.output.exists():
        parser.error("Output file exists; choose a new filename")

    # Checkpoint paths remain relative to this project, not the shell directory.
    checkpoint_dir = Path(args.checkpoint).expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_DIR / "checkpoints" / checkpoint_dir
    if not checkpoint_dir.is_dir():
        parser.error(
            f"Checkpoint directory not found: {checkpoint_dir}. "
            "Train it first or supply the path to downloaded weights."
        )

    if args.text is not None or args.input_file is not None:
        report = args.text if args.text is not None else args.input_file.read_text(encoding="utf-8")
        if not report.strip():
            parser.error("The supplied report is empty")
        examples = [{"radiology_report": report}]
        data_origin = "user-supplied report"
    else:
        from dataset import load_audited_data, split_train_development

        dataset, clean_train = load_audited_data()
        if args.split == "final":
            selected = dataset["validation"]  # Reserved labelled evaluation cases.
        else:
            _, selected = split_train_development(clean_train)
        if args.samples > len(selected):
            parser.error("Requested more examples than this split contains")
        examples = selected.shuffle(seed=42).select(range(args.samples))
        data_origin = args.split

    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = RadiologySummarizer(str(checkpoint_dir)).to(device)
    model.eval() # Disable dropout; generation below also disables gradients.

    print("Checkpoint:", checkpoint_dir)
    print("Device:", device)
    print("Draft outputs require source review; clinical safety is not established.")
    results = []

    for index, example in enumerate(examples, start=1):
        # Only the source report is passed to generation; references are display-only.
        inputs = tokenizer(
            INSTRUCTION + example["radiology_report"],
            return_tensors="pt",
            truncation=False,
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=0,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
            )

        sequence = output_ids[0]
        generated_tokens = sequence.numel() - 1  # Exclude T5 decoder-start token.
        ended_with_eos = sequence[-1].item() == tokenizer.eos_token_id
        result = {
            "example": index,
            "report": example["radiology_report"],
            "summary": tokenizer.decode(sequence, skip_special_tokens=True),
            "generated_tokens": generated_tokens,
            "ended_with_eos": ended_with_eos,
            "reached_cap_without_eos": generated_tokens >= args.max_new_tokens and not ended_with_eos,
        }
        if "layman_report" in example:
            result["reference"] = example["layman_report"]
            result["source"] = example["source"]
        results.append(result)
        print(f"\nExample {index}")
        print("Report:", result["report"])
        if "reference" in result:
            print("Reference:", result["reference"])
        print("Generated:", result["summary"])
        print("Generated tokens:", generated_tokens, "| Ended with EOS:", ended_with_eos)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump({"checkpoint": str(checkpoint_dir), "data_origin": data_origin,
                       "max_new_tokens": args.max_new_tokens, "examples": results},
                      file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
