"""Find review candidates; low ROUGE is not a clinical error label."""

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer


PROJECT_DIR = Path(__file__).resolve().parents[2]


def load_records(folder):
    path = PROJECT_DIR / "results" / folder / "predictions.jsonl"
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "final"], default="development")
    args = parser.parse_args()

    suffix = "all_final_eval" if args.split == "final" else "dev_10k_eval"
    lora_records = load_records(f"lora_{suffix}")
    full_records = load_records(f"full_{suffix}")
    if len(lora_records) != len(full_records):
        raise ValueError("Prediction files have different row counts")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    candidates = []

    for index, (lora, full) in enumerate(zip(lora_records, full_records)):
        # Confirm we compare the same case, not merely the same row position.
        for field in ("source", "images_path", "report", "reference", "baseline"):
            if lora[field] != full[field]:
                raise ValueError(f"Unaligned predictions at row {index}: {field}")

        candidates.append({
            "split": args.split,
            "subset_index": index,
            "development_subset_index": index,
            "source": full["source"],
            "images_path": full["images_path"],
            "report": full["report"],
            "reference": full["reference"],
            "baseline": full["baseline"],
            "lora": lora["trained"],
            "full": full["trained"],
            "lora_diagnostics": lora.get("trained_diagnostics"),
            "full_diagnostics": full.get("trained_diagnostics"),
            "lora_rougeL": scorer.score(
                full["reference"], lora["trained"]
            )["rougeL"].fmeasure,
            "full_rougeL": scorer.score(
                full["reference"], full["trained"]
            )["rougeL"].fmeasure,
        })
        if (index + 1) % 1000 == 0:
            print(f"Scored {index + 1}/{len(full_records)} candidates", flush=True)

    # Preserve the original development record schema without mislabelling final cases.
    for row in candidates:
        if args.split == "final":
            del row["development_subset_index"]

    # First include the lowest full-model ROUGE-L case from each source.
    # Then fill to five using the lowest remaining scores and distinct reports.
    ranked = sorted(
        candidates,
        key=lambda row: (row["full_rougeL"], row["subset_index"]),
    )
    selected = []
    represented_sources = set()
    for row in ranked:
        if row["source"] not in represented_sources:
            selected.append({**row, "selection_reason": "lowest score within source"})
            represented_sources.add(row["source"])

    selected_indices = {row["subset_index"] for row in selected}
    selected_reports = {row["report"].strip() for row in selected}
    for row in ranked:
        if len(selected) >= 5:
            break
        if (
            row["subset_index"] not in selected_indices
            and row["report"].strip() not in selected_reports
        ):
            selected.append({**row, "selection_reason": "lowest remaining score"})
            selected_indices.add(row["subset_index"])
            selected_reports.add(row["report"].strip())

    # Save clinical text only in the ignored local results directory.
    output_path = PROJECT_DIR / "results" / f"{args.split}_failure_candidates.jsonl"
    with output_path.open("w") as file:
        for row in selected:
            file.write(json.dumps(row) + "\n")
            print(
                f"Index {row['subset_index']} | {row['source']} | "
                f"LoRA ROUGE-L {row['lora_rougeL']:.3f} | "
                f"Full ROUGE-L {row['full_rougeL']:.3f}"
            )

    print("\nSaved review candidates to:", output_path)
    print(f"These are {args.split} candidates, not confirmed clinical failures.")
    print("Selection is intentionally biased toward low overlap; do not estimate error rates.")


if __name__ == "__main__":
    main()
