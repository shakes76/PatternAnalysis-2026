"""Audit saved validation predictions for overlap, ROUGE and readability."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import textstat

from dataset import DATASET_NAME, audit_overlap
from train import calculate_rouge


def load_splits():
    """Load only the official training and validation parquet files."""
    files = {
        split: hf_hub_download(
            repo_id=DATASET_NAME, repo_type="dataset",
            filename=f"data/{split}-00000-of-00001.parquet",
        )
        for split in ("train", "validation")
    }
    return load_dataset("parquet", data_files=files)


def readability_rows(texts):
    """Calculate word count and Flesch-Kincaid grade once per report."""
    rows = []
    for text in texts:
        words = textstat.lexicon_count(text, removepunct=True)
        grade = textstat.flesch_kincaid_grade(text) if words else None
        rows.append((words, grade))
    return rows


def summarize_readability(rows, indices):
    selected = [rows[index] for index in indices]
    grades = [grade for _, grade in selected if grade is not None]
    return {
        "reports": len(selected),
        "scored_reports": len(grades),
        "zero_word_reports": len(selected) - len(grades),
        "mean_words": mean(words for words, _ in selected) if selected else None,
        "fk_grade_mean": mean(grades) if grades else None,
        "fk_grade_median": median(grades) if grades else None,
        # The percentage denominator excludes zero-word outputs.
        "fk_grade_le_6_percent": 100 * sum(grade <= 6 for grade in grades) / len(grades)
        if grades else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="New JSON file for the analysis")
    args = parser.parse_args()
    textstat.set_lang("en_US")
    data = load_splits()
    validation = data["validation"]
    overlap, unseen = audit_overlap(data["train"], validation)
    unseen_set = set(unseen)
    subsets = {
        "full": list(range(len(validation))),
        "unseen": unseen,
        "overlap": [index for index in range(len(validation)) if index not in unseen_set],
    }
    readability_subsets = {name: subsets[name] for name in ("full", "unseen")}
    result = {
        "dataset": DATASET_NAME,
        "overlap_matching": "Exact full source and target strings without normalization",
        "overlap": overlap,
        "readability_package": f"textstat=={version('textstat')}",
        "readability_language": "en_US",
        "comparators": {},
        "runs": [],
    }
    for name, field in (("source", "radiology_report"), ("reference", "layman_report")):
        rows = readability_rows(validation[field])
        result["comparators"][name] = {
            subset: summarize_readability(rows, indices)
            for subset, indices in readability_subsets.items()
        }

    expected = [
        (row["radiology_report"], row["layman_report"], row["source"], row["images_path"])
        for row in validation
    ]
    for run_dir in map(Path, args.run_dirs):
        with (run_dir / "predictions.jsonl").open(encoding="utf-8") as file:
            records = [json.loads(line) for line in file]
        actual = [
            (row["radiology_report"], row["reference"], row["source"], row["images_path"])
            for row in records
        ]
        assert actual == expected, f"Validation rows or order differ in {run_dir}"
        predictions = [row["prediction"] for row in records]
        references = [row["reference"] for row in records]
        rows = readability_rows(predictions)
        scores = {}
        for name, indices in subsets.items():
            rouge = calculate_rouge(
                [predictions[index] for index in indices],
                [references[index] for index in indices],
            ) if indices else None
            scores[name] = {"reports": len(indices), "rouge": rouge}
        result["runs"].append({
            "run_dir": str(run_dir.resolve()),
            "rouge": scores,
            "readability": {
                name: summarize_readability(rows, indices)
                for name, indices in readability_subsets.items()
            },
        })
        print(f"Evaluated {run_dir}: {len(records)} aligned validation reports")

    with Path(args.output).open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)


if __name__ == "__main__":
    main()
