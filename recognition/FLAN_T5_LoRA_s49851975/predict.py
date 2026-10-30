"""
Evaluate the models and show example summaries.

Models:
- copy: baseline B1, the "summary" is the original report itself.
- zero_shot: baseline B2, FLAN-T5 without any fine-tuning.
- full: a fully fine-tuned model saved by train.py.
- lora: a LoRA model saved by train.py.

For each model this script:
1. Prints summaries for a few official test reports (demo).
2. Generates summaries for the clean validation set and computes ROUGE-1/2/L/Lsum,
   FKGL and generation speed (tokens per second).
3. Saves every summary with its scores to predictions.csv for the manual safety review,
   and the average scores to metrics.json.
With --plot it draws a bar chart from the metrics.json files of several runs.

Examples:
    python predict.py --model copy --run-dir outputs/B1_copy
    python predict.py --model zero_shot --run-dir outputs/B2_zero_shot
    python predict.py --model full --run-dir outputs/R1_full
    python predict.py --model lora --run-dir outputs/R2_lora
    python predict.py --model lora --run-dir outputs/R3_lora_all
    python predict.py --plot outputs/B1_copy outputs/B2_zero_shot outputs/R1_full outputs/R2_lora \
        outputs/R3_lora_all
"""

import argparse
import csv
import json
import os
import time

import matplotlib.pyplot as plt
import nltk
import textstat
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dataset import MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, add_prompt, prepare_splits
from modules import ALL_LINEAR_TARGETS, MODEL_NAME, QUERY_VALUE_TARGETS, LoRALinear, build_model

# Every model uses the same decoding settings.
NUMBER_OF_BEAMS = 4

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def parse_arguments():
    """Read the evaluation settings from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate a summarisation model.")
    parser.add_argument("--model", choices=["copy", "zero_shot", "full", "lora"])
    parser.add_argument("--run-dir", help="Folder for the results (for full and lora, the folder written by train.py)")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Base model for the zero-shot baseline")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-limit", type=int, default=None, help="Use only the first N validation pairs")
    parser.add_argument("--demo-count", type=int, default=3, help="Number of test reports to print")
    parser.add_argument("--plot", nargs="+", default=None, help="Run folders to compare in a bar chart")
    parser.add_argument("--plot-output", default="outputs/metrics_comparison.png")
    arguments = parser.parse_args()
    if arguments.plot is None and (arguments.model is None or arguments.run_dir is None):
        parser.error("--model and --run-dir are needed unless --plot is used")
    return arguments


def load_model(model_type, run_dir, model_name):
    """Return the model to evaluate and its base model name; the copy baseline has no model."""
    if model_type == "copy":
        return None, model_name
    if model_type == "zero_shot":
        return build_model("full", model_name), model_name
    return load_trained_model(model_type, run_dir)


def load_trained_model(model_type, run_dir):
    """Load a model saved by train.py and return it with the name of its base model."""
    with open(os.path.join(run_dir, "run_summary.json")) as summary_file:
        summary = json.load(summary_file)

    if model_type == "full":
        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(run_dir, "model"))
        return model, summary["model_name"]

    # LoRA: rebuild the same LoRA model, then load only the trained LoRA matrices.
    if summary["lora_targets"] == "query_value":
        targets = QUERY_VALUE_TARGETS
    else:
        targets = ALL_LINEAR_TARGETS
    model = build_model("lora", summary["model_name"],
                        r=summary["lora_rank"], alpha=summary["lora_alpha"], targets=targets)
    lora_weights = torch.load(os.path.join(run_dir, "lora_weights.pt"))
    # strict=False because the file only holds the LoRA matrices, not the frozen weights.
    load_result = model.load_state_dict(lora_weights, strict=False)
    assert len(load_result.unexpected_keys) == 0, "The LoRA file does not match the model"

    # Merge B A into W0 so generation is as fast as the original model.
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    return model, summary["model_name"]


def generate_summaries(model, tokenizer, reports, device, batch_size):
    """Generate a summary for each report with beam search and return them with the tokens per second."""
    use_bfloat16 = device.type == "cuda"
    model.eval()
    summaries = []
    generated_tokens = 0
    generation_seconds = 0.0
    batch_count = 0

    for start in range(0, len(reports), batch_size):
        batch_reports = reports[start:start + batch_size]
        prompts = []
        for report in batch_reports:
            prompts.append(add_prompt(report))
        inputs = tokenizer(prompts, max_length=MAX_INPUT_LENGTH, truncation=True, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        start_time = time.time()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            num_beams=NUMBER_OF_BEAMS, max_new_tokens=MAX_TARGET_LENGTH)
        if device.type == "cuda":
            # Wait for the GPU to finish so the timing is correct.
            torch.cuda.synchronize()
        generation_seconds += time.time() - start_time

        # Padding (which also starts every T5 output) is not counted as a generated token.
        generated_tokens += (output_ids != tokenizer.pad_token_id).sum().item()
        summaries.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

        # Show progress every 10 batches and at the end.
        batch_count += 1
        if batch_count % 10 == 0 or len(summaries) == len(reports):
            print(f"Generated {len(summaries)}/{len(reports)} summaries")

    return summaries, generated_tokens / generation_seconds


def add_sentence_breaks(text):
    """Put each sentence on its own line, which ROUGE-Lsum needs."""
    return "\n".join(nltk.sent_tokenize(text))


def score_summaries(predictions, references):
    """Return the ROUGE F1 scores (0 to 100) and the FKGL of every prediction."""
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
    rows = []
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(add_sentence_breaks(reference), add_sentence_breaks(prediction))
        row = {}
        for rouge_type in ROUGE_TYPES:
            row[rouge_type] = 100 * scores[rouge_type].fmeasure
        row["fkgl"] = textstat.flesch_kincaid_grade(prediction)
        rows.append(row)
    return rows


def average_scores(rows):
    """Average each score over all examples."""
    averages = {}
    for score_name in ROUGE_TYPES + ["fkgl"]:
        total = 0.0
        for row in rows:
            total += row[score_name]
        averages[score_name] = total / len(rows)
    return averages


def save_predictions(path, reports, references, predictions, rows):
    """Write every report, reference, prediction and its scores to a CSV file for manual review."""
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["index", "report", "reference", "prediction"] + ROUGE_TYPES + ["fkgl"])
        for index in range(len(predictions)):
            line = [index, reports[index], references[index], predictions[index]]
            for score_name in ROUGE_TYPES + ["fkgl"]:
                line.append(round(rows[index][score_name], 2))
            writer.writerow(line)


def plot_metrics(run_dirs, output_path):
    """Draw the ROUGE scores and the FKGL of several runs side by side."""
    run_names = []
    all_metrics = []
    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "metrics.json")) as metrics_file:
            all_metrics.append(json.load(metrics_file))
        run_names.append(os.path.basename(os.path.normpath(run_dir)))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    rouge_axis = axes[0]
    fkgl_axis = axes[1]

    # Left: one group of bars per ROUGE type, one bar per run.
    bar_width = 0.8 / len(run_names)
    for run_index in range(len(run_names)):
        positions = []
        heights = []
        for rouge_index in range(len(ROUGE_TYPES)):
            positions.append(rouge_index + run_index * bar_width)
            heights.append(all_metrics[run_index][ROUGE_TYPES[rouge_index]])
        rouge_axis.bar(positions, heights, width=bar_width, label=run_names[run_index])
    group_centres = []
    for rouge_index in range(len(ROUGE_TYPES)):
        group_centres.append(rouge_index + (len(run_names) - 1) * bar_width / 2)
    rouge_axis.set_xticks(group_centres)
    rouge_axis.set_xticklabels(["ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum"])
    rouge_axis.set_ylim(0, 100)
    rouge_axis.set_ylabel("F1 score (0 to 100)")
    rouge_axis.set_title("ROUGE on the clean validation set (higher is better)")
    rouge_axis.legend(loc="upper right")

    # Right: average FKGL per run, with the feasibility target as a dashed line.
    fkgl_values = []
    for metrics in all_metrics:
        fkgl_values.append(metrics["fkgl"])
    fkgl_axis.bar(run_names, fkgl_values, color="gray")
    fkgl_axis.axhline(7.5, color="red", linestyle="--", label="Feasibility target (7.5)")
    fkgl_axis.set_ylabel("Flesch-Kincaid Grade Level")
    fkgl_axis.set_title("Readability (lower is easier)")
    fkgl_axis.tick_params(axis="x", labelrotation=20)
    fkgl_axis.legend()

    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved bar chart to {output_path}")


def main():
    """Evaluate one model, or draw the comparison chart when --plot is given."""
    arguments = parse_arguments()
    if arguments.plot is not None:
        plot_metrics(arguments.plot, arguments.plot_output)
        return

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    os.makedirs(arguments.run_dir, exist_ok=True)

    # Sentence splitter used for ROUGE-Lsum.
    nltk.download("punkt_tab", quiet=True)

    model, model_name = load_model(arguments.model, arguments.run_dir, arguments.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is not None:
        model.to(device)

    train_pairs, validation_pairs, test_reports = prepare_splits()
    if arguments.validation_limit is not None:
        validation_pairs = validation_pairs[:arguments.validation_limit]
    reports = []
    references = []
    for report, summary in validation_pairs:
        reports.append(report)
        references.append(summary)

    # 1. Demo on a few test reports (this also warms up the GPU before timing the validation run).
    demo_reports = test_reports[:arguments.demo_count]
    if model is None:
        demo_summaries = demo_reports
    else:
        # [0] keeps only the summaries; the speed of this small demo is not needed.
        demo_summaries = generate_summaries(model, tokenizer, demo_reports, device, arguments.batch_size)[0]
    for number in range(len(demo_reports)):
        print(f"\nTest report {number + 1}: {demo_reports[number]}")
        print(f"Summary: {demo_summaries[number]}")

    # 2. Generate and score the clean validation set.
    print(f"\nEvaluating {arguments.model} on {len(reports)} validation pairs")
    if model is None:
        # Baseline B1 copies the report, so nothing is generated and there is no speed to measure.
        predictions = reports
        tokens_per_second = None
    else:
        predictions, tokens_per_second = generate_summaries(model, tokenizer, reports, device, arguments.batch_size)
    rows = score_summaries(predictions, references)
    averages = average_scores(rows)

    # 3. Save the scores of every example and the averages.
    metrics = {
        "model": arguments.model,
        "validation_pairs": len(reports),
        "rouge1": averages["rouge1"],
        "rouge2": averages["rouge2"],
        "rougeL": averages["rougeL"],
        "rougeLsum": averages["rougeLsum"],
        "fkgl": averages["fkgl"],
        "tokens_per_second": tokens_per_second,
        "batch_size": arguments.batch_size,
        "number_of_beams": NUMBER_OF_BEAMS,
        "max_new_tokens": MAX_TARGET_LENGTH,
    }
    save_predictions(os.path.join(arguments.run_dir, "predictions.csv"), reports, references, predictions, rows)
    with open(os.path.join(arguments.run_dir, "metrics.json"), "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"ROUGE-1 {averages['rouge1']:.2f}, ROUGE-2 {averages['rouge2']:.2f}, "
          f"ROUGE-L {averages['rougeL']:.2f}, ROUGE-Lsum {averages['rougeLsum']:.2f}, "
          f"FKGL {averages['fkgl']:.2f}")
    if tokens_per_second is not None:
        print(f"Generation speed: {tokens_per_second:.1f} tokens/s")
    print(f"Saved predictions.csv and metrics.json to {arguments.run_dir}")


if __name__ == "__main__":
    main()
