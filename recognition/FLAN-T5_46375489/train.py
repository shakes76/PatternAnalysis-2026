"""Train FLAN-T5 with full fine-tuning or LoRA and save validation results."""

import argparse
import csv
import logging
import math
import platform
import re
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib
import torch
import yaml
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

# Select a non-interactive backend before pyplot for headless training runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from dataset import RadiologyData
from modules import (
    DEFAULT_MODEL_NAME,
    FINE_TUNING_MODES,
    count_parameters,
    load_model,
    trainable_parameters,
)
from predict import GENERATION_METRICS_SCOPE, decode, generate_with_metrics
from utils import (
    DEFAULT_PROMPT_TEMPLATE,
    DeviceClock,
    cuda_memory_gib,
    measure_stage,
    synchronize,
    configure_logger,
    select_device,
    set_seed,
    write_json,
)


# Save definitions beside measurements so results remain interpretable.
METRIC_SCOPES = {
    "timing_scope": (
        "fit_seconds includes data preparation, training, validation, "
        "plots and checkpoints; excludes model/tokenizer loading "
        "before fit()."
    ),
    "validation_scope": "Loss, generation, decoding and ROUGE scoring.",
    "validation_generation_scope": GENERATION_METRICS_SCOPE,
    "validation_throughput_scope": (
        "generated_tokens_per_second uses generation time only; "
        "end_to_end_generated_tokens_per_second uses the complete "
        "validation stage, including loss, decoding and ROUGE. "
        "Aggregate rates use total tokens / total time."
    ),
    "gpu_memory_scope": (
        "PyTorch CUDA allocator peaks, including persistent "
        "model/optimizer memory and cached blocks; "
        "GiB (1024**3 bytes); null on CPU/MPS."
    ),
    "step_history_file": "step_history.csv",
    "step_loss_scope": (
        "Token-weighted mean over the batches contributing to one "
        "optimizer update, measured before that update. grad_norm is "
        "measured after token normalization and before clipping."
    ),
    "step_timing_scope": (
        "Synchronized accumulation-window time includes batch loading, "
        "device transfers, forward/backward, clipping and optimizer "
        "update; excludes CSV writes, validation and checkpoints. "
        "train_elapsed_seconds sums these windows. Synchronization "
        "and per-update writes can add overhead to training."
    ),
    "step_count_scope": (
        "Examples and non-padding target tokens processed per update; "
        "cumulative counts include repeated passes across epochs. "
        "epoch_fraction uses completed batches / batches per epoch."
    ),
    "step_gpu_memory_scope": (
        "Allocated/reserved snapshots after each update, not peaks; "
        "GiB; blank in CSV on CPU/MPS. "
        "Epoch history retains stage peaks."
    ),
}


@dataclass
class _UpdateWindow:
    """Token-weighted loss and counts for one accumulated optimiser update."""

    loss_sum: float = 0.0
    tokens: int = 0
    examples: int = 0

    def add(self, loss_sum, tokens, examples):
        """Adds a batch's summed loss, target-token count and example count."""
        self.loss_sum += loss_sum
        self.tokens += tokens
        self.examples += examples


class RadiologyTrainer:
    """Trains and validates FLAN-T5 while recording metrics and checkpoints.

    Attributes:
        history: List of metric/resource dicts, one per completed epoch.
        step_history: List of training metric dicts, one per optimiser update.
        output_dir: pathlib.Path where logs, plots and checkpoints are saved.
    """

    METRIC_NAMES = ("rouge1", "rouge2", "rougeL", "rougeLsum")
    STEP_FIELDS = (
        "epoch",
        "epoch_fraction",
        "global_step",
        "train_loss",
        "learning_rate",
        "grad_norm",
        "examples",
        "target_tokens",
        "cumulative_examples",
        "cumulative_target_tokens",
        "step_seconds",
        "train_elapsed_seconds",
        "examples_per_second",
        "target_tokens_per_second",
        "allocated_gpu_gib",
        "reserved_gpu_gib",
    )

    def __init__(self, model, data, config):
        """Stores run settings and initialises empty training histories.

        Args:
            model: Hugging Face seq2seq model, optionally with PEFT adapters.
            data: RadiologyData instance; fit() loads its dataset splits.
            config: Nested settings dict returned by parse_config(), containing
                model, data and training sections.
        """
        self.model = model
        self.data = data
        self.config = config
        self.settings = config["training"]
        self.device = torch.device(self.settings["device"])
        self.output_dir = Path(self.settings["output_dir"])

        self.history = []
        self.step_history = []

        self._optimizer = None
        self._best_validation_loss = float("inf")
        self._global_step = 0
        self.logger = logging.getLogger("radiology")

    def fit(self):
        """Runs training epochs and saves validation metrics and checkpoints.

        Creates a fresh optimiser and resets the step CSV. Use a new trainer
        instance for a fresh run.

        Returns:
            The per-epoch metric dicts also stored in history.
        """
        started = perf_counter()
        self._prepare_run()
        resources = self._describe_run()

        for epoch in range(1, self.settings["epochs"] + 1):
            validation = self._run_epoch(epoch)
            self._save_epoch_checkpoints(epoch, validation["loss"])
            resources.update(self._run_totals(started))
            write_json(resources, self.output_dir / "resources.json")

        return self.history

    def _prepare_run(self):
        """Loads data, places the model and creates a fresh optimiser."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "step_history.csv").write_text("", encoding="utf-8")

        self.data.load()
        self.model.to(self.device)
        # Frozen base weights are excluded when the model uses LoRA.
        self._optimizer = AdamW(
            trainable_parameters(self.model),
            lr=self.settings["learning_rate"],
            weight_decay=self.settings["weight_decay"],
        )

    def _describe_run(self):
        """Returns runtime, workload and measurement metadata."""
        trainable, total = count_parameters(self.model)
        return {
            # Runtime details and parameter/data counts identify the workload.
            "device": str(self.device),
            "gpu": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda" else None
            ),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "packages": {
                name: version(name)
                for name in ("transformers", "peft", "datasets")
            },
            "total_parameters": total,
            "trainable_parameters": trainable,
            "train_examples": self.data.num_examples("train"),
            "validation_examples": self.data.num_examples("validation"),

            **METRIC_SCOPES,
        }

    def _run_epoch(self, epoch):
        """Trains, validates and records one epoch.

        Args:
            epoch: One-based epoch number.

        Returns:
            The validation metrics dict, used for checkpoint selection.
        """
        self.logger.info(
            "Epoch %s/%s: training", epoch, self.settings["epochs"]
        )
        step_before = self._global_step
        train_loss, train_resources = self._measure_stage(
            self._train_one_epoch, self.data.dataloader("train")
        )

        self.logger.info("Epoch %s: validation", epoch)
        validation, val_resources = self._measure_stage(self.evaluate)

        self.history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                **{
                    f"val_{name}": value
                    for name, value in validation.items()
                },
                **{
                    f"train_{name}": value
                    for name, value in train_resources.items()
                },
                **{
                    f"val_{name}": value
                    for name, value in val_resources.items()
                },
                "train_examples_per_second": (
                    self.data.num_examples("train") / train_resources["seconds"]
                ),
                "val_examples_per_second": (
                    self.data.num_examples("validation")
                    / val_resources["seconds"]
                ),
                "val_end_to_end_generated_tokens_per_second": (
                    validation["generated_tokens"]
                    / val_resources["seconds"]
                ),
                "optimizer_steps": self._global_step - step_before,
            }
        )

        self.plot_history()
        write_json(self.history, self.output_dir / "history.json")
        self.logger.info("Metrics: %s", self.history[-1])

        return validation

    def _save_epoch_checkpoints(self, epoch, validation_loss):
        """Applies the configured best/latest or per-epoch checkpoint policy.

        Args:
            epoch: One-based epoch number.
            validation_loss: Loss used to select the best checkpoint.
        """
        is_best = validation_loss < self._best_validation_loss
        if is_best:
            self._best_validation_loss = validation_loss

        if self.settings.get("save_every_epoch", False):
            checkpoint = self.save_checkpoint(f"epoch-{epoch:02d}")
            if is_best:
                # Point to the best epoch without saving its weights twice.
                write_json(
                    {
                        "checkpoint": checkpoint.name,
                        "val_loss": validation_loss,
                    },
                    self.output_dir / "best_checkpoint.json",
                )
        else:
            if is_best:
                self.save_checkpoint("best")
            self.save_checkpoint("latest")

    def _run_totals(self, fit_started):
        """Aggregates resource measurements over completed epochs.

        Args:
            fit_started: perf_counter() value recorded when fit() began.

        Returns:
            A dict of cumulative counts, durations and aggregate throughput.
        """
        training_seconds = sum(row["train_seconds"] for row in self.history)
        validation_seconds = sum(row["val_seconds"] for row in self.history)
        generated_tokens = sum(
            row["val_generated_tokens"] for row in self.history
        )
        generation_seconds = sum(
            row["val_generation_seconds"] for row in self.history
        )
        return {
            "completed_epochs": len(self.history),
            "optimizer_steps": self._global_step,
            "total_training_seconds": training_seconds,
            "total_validation_seconds": validation_seconds,
            "total_validation_generated_tokens": generated_tokens,
            "total_validation_generation_seconds": generation_seconds,
            "fit_seconds": perf_counter() - fit_started,
            # Ratios of totals weight each epoch by its actual elapsed time.
            "validation_generated_tokens_per_second": (
                generated_tokens / generation_seconds
            ),
            "validation_end_to_end_generated_tokens_per_second": (
                generated_tokens / validation_seconds
            ),
        }

    def _autocast(self):
        """Selects the configured precision context.

        Returns:
            A torch.autocast context using BF16 when enabled, or nullcontext.
        """
        return (
            torch.autocast(self.device.type, dtype=torch.bfloat16)
            if self.settings.get("bf16", False) else nullcontext()
        )

    def _measure_stage(self, operation, *args):
        """Calls utils.measure_stage on this trainer's device; see its contract."""
        return measure_stage(self.device, operation, *args)

    def _synchronize(self):
        """Waits for queued CUDA or MPS work; does nothing on CPU."""
        synchronize(self.device)

    def _train_one_epoch(self, loader):
        """Trains one epoch with token-weighted gradient accumulation.

        Requires an initialised optimiser. Clips gradients and records each
        update in step_history and its CSV, including a final partial window.

        Args:
            loader: PyTorch DataLoader yielding tokenised batches with target
                labels padded using -100.

        Returns:
            Mean cross-entropy per non-padding target token (float).
        """
        self.model.train()
        self._optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_tokens = 0

        # A window groups the batches contributing to one optimiser update.
        window = _UpdateWindow()
        accumulation_steps = self.settings["gradient_accumulation_steps"]
        # Read current parameters after placement; do not cache before .to().
        parameters = trainable_parameters(self.model)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "step_history.csv").open(
            "a", newline="", encoding="utf-8"
        ) as log:
            writer = csv.DictWriter(log, fieldnames=self.STEP_FIELDS)
            if log.tell() == 0:
                writer.writeheader()

            clock = DeviceClock(self.device)

            for step, batch in enumerate(loader, start=1):
                batch = batch.to(self.device)
                # The collator marks padded labels with -100; ignore them here.
                tokens = batch["labels"].ne(-100).sum().item()
                with self._autocast():
                    loss = self.model(**batch).loss

                # Undo the model's mean reduction before accumulating gradients.
                # Normalising once per window weights unequal target lengths.
                (loss * tokens).backward()
                summed_loss = loss.detach().item() * tokens

                total_loss += summed_loss
                total_tokens += tokens
                window.add(summed_loss, tokens, batch["labels"].shape[0])

                # Flush the final window even when it has fewer batches.
                if step % accumulation_steps == 0 or step == len(loader):
                    grad_norm, learning_rate = self._optimizer_step(
                        parameters, window.tokens
                    )
                    seconds = clock.elapsed()
                    record = self._step_record(
                        window, seconds, grad_norm, learning_rate,
                        step, len(loader),
                    )
                    self.step_history.append(record)
                    writer.writerow(record)
                    log.flush()

                    # Start the next timer after logging I/O has finished.
                    window = _UpdateWindow()
                    clock.restart()

        return total_loss / total_tokens

    def _optimizer_step(self, parameters, tokens):
        """Normalises accumulated gradients, clips them and updates weights.

        Args:
            parameters: Current trainable model parameters.
            tokens: Number of target tokens in the accumulation window.

        Returns:
            The pre-clipping gradient norm tensor and update learning rate.
        """
        for parameter in parameters:
            parameter.grad.div_(tokens)

        # Clip normalised gradients; the return is pre-clip norm.
        grad_norm = clip_grad_norm_(
            parameters, self.settings["max_grad_norm"]
        )
        learning_rate = self._optimizer.param_groups[0]["lr"]
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self._global_step += 1

        return grad_norm, learning_rate

    def _step_record(
        self, window, seconds, grad_norm, learning_rate, step, batches
    ):
        """Builds the metric row for a completed optimiser update.

        Args:
            window: Loss, target-token and example totals for this update.
            seconds: Synchronised window time, excluding CSV writes.
            grad_norm: Gradient norm before clipping.
            learning_rate: Learning rate used for the update.
            step: One-based index of the last batch in this window.
            batches: Total number of batches in this epoch.

        Returns:
            A dict matching STEP_FIELDS, including cumulative counts.
        """
        previous = self.step_history[-1] if self.step_history else {}
        return {
            "epoch": len(self.history) + 1,
            "epoch_fraction": len(self.history) + step / batches,
            "global_step": self._global_step,
            "train_loss": window.loss_sum / window.tokens,
            "learning_rate": learning_rate,
            "grad_norm": float(grad_norm),
            "examples": window.examples,
            "target_tokens": window.tokens,
            "cumulative_examples": (
                previous.get("cumulative_examples", 0)
                + window.examples
            ),
            "cumulative_target_tokens": (
                previous.get("cumulative_target_tokens", 0)
                + window.tokens
            ),
            "step_seconds": seconds,
            "train_elapsed_seconds": (
                previous.get("train_elapsed_seconds", 0.0) + seconds
            ),
            "examples_per_second": window.examples / seconds,
            "target_tokens_per_second": window.tokens / seconds,
            "allocated_gpu_gib": (
                cuda_memory_gib(self.device, torch.cuda.memory_allocated)
            ),
            "reserved_gpu_gib": (
                cuda_memory_gib(self.device, torch.cuda.memory_reserved)
            ),
        }

    @torch.inference_mode()
    def evaluate(self):
        """Computes validation loss, ROUGE and generation throughput.

        Requires loaded data. Loss uses truncated target labels; ROUGE uses
        full reference texts and averages per-report F1 scores. Restores the
        model's previous training mode even if evaluation fails.

        Returns:
            A dict containing token-weighted loss, rouge1, rouge2, rougeL,
            rougeLsum, generated token count, generation seconds and throughput.
        """
        self.model.to(self.device)
        was_training = self.model.training
        self.model.eval()

        try:
            total_loss = 0.0
            total_tokens = 0
            predictions = []
            generated_tokens = 0
            generation_seconds = 0.0

            for batch in self.data.dataloader("validation"):
                batch = batch.to(self.device)
                tokens = batch["labels"].ne(-100).sum().item()
                with self._autocast():
                    total_loss += self.model(**batch).loss.item() * tokens

                    # Labels contribute to loss, not to generation input.
                    generated, generation = generate_with_metrics(
                        self.model,
                        {
                            "input_ids": batch["input_ids"],
                            "attention_mask": batch["attention_mask"],
                        },
                        max_new_tokens=self.settings["max_new_tokens"],
                        num_beams=self.settings["num_beams"],
                        do_sample=False,
                    )

                generated_tokens += generation["generated_tokens"]
                generation_seconds += generation["generation_seconds"]
                total_tokens += tokens
                predictions.extend(decode(self.data.tokenizer, generated))

            return {
                "loss": total_loss / total_tokens,
                "generated_tokens": generated_tokens,
                "generation_seconds": generation_seconds,
                "generated_tokens_per_second": (
                    generated_tokens / generation_seconds
                ),
                **rouge_scores(
                    predictions, self.data.references("validation"),
                    self.METRIC_NAMES,
                ),
            }
        finally:
            self.model.train(was_training)

    def save_checkpoint(self, name="latest"):
        """Saves model weights or adapters, tokenizer, settings and run state.

        Requires an initialised optimiser. Run state includes optimiser state,
        epoch/update counters, history and the CPU PyTorch RNG state. Automatic
        training resumption is not implemented.

        Args:
            name: Checkpoint subdirectory name (str), relative to output_dir.

        Returns:
            The saved checkpoint directory as a pathlib.Path.
        """
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # PEFT saves adapter weights here; a full model saves all model weights.
        self.model.save_pretrained(checkpoint_dir)
        self.data.tokenizer.save_pretrained(checkpoint_dir)
        write_json(self.config, checkpoint_dir / "config_run.json")

        # Keep optimiser/counter state separate from inference model files.
        torch.save(
            {
                "optimizer": self._optimizer.state_dict(),
                "epoch": len(self.history),
                "global_step": self._global_step,
                "best_validation_loss": self._best_validation_loss,
                "history": self.history,
                "torch_rng_state": torch.get_rng_state(),
            },
            checkpoint_dir / "training_state.pt",
        )

        self.logger.info("Saved checkpoint: %s", checkpoint_dir)
        return checkpoint_dir

    def plot_history(self):
        """Saves loss and validation-ROUGE plots as PNG files.

        Uses per-update training loss when available, otherwise epoch averages.

        Returns:
            A (loss_path, metrics_path) tuple of pathlib.Path objects.
        """
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        epochs = [row["epoch"] for row in self.history]

        plots = (
            (
                "losses",
                ("train_loss", "val_loss"),
                "Learning curves",
                "Cross-entropy loss",
            ),
            (
                "metrics",
                tuple(f"val_{key}" for key in self.METRIC_NAMES),
                "Validation metrics",
                "ROUGE score",
            ),
        )

        for filename, columns, title, ylabel in plots:
            fig, ax = plt.subplots(figsize=(7, 4.5))

            for column in columns:
                if column == "train_loss" and self.step_history:
                    # Fractional epochs show every update on the same x-axis.
                    ax.plot(
                        [row["epoch_fraction"] for row in self.step_history],
                        [row["train_loss"] for row in self.step_history],
                        marker=".",
                        label="Training (per update)",
                    )
                else:
                    ax.plot(
                        epochs,
                        [row[column] for row in self.history],
                        marker="o",
                        linestyle="--" if column == "val_loss" else "-",
                        label="Validation" if column == "val_loss" else column,
                    )

            ax.set(xlabel="Epoch", ylabel=ylabel, title=title, xticks=epochs)
            if filename == "losses":
                ax.set_xlim(left=0)
                ax.set_xticks([0, *epochs])
            if filename == "metrics":
                ax.set_ylim(0, 1)

            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"{filename}.png", dpi=160)
            plt.close(fig)

        return plot_dir / "losses.png", plot_dir / "metrics.png"


def _sentence_lines(text):
    """Separates sentences with newlines for ROUGE-Lsum scoring."""
    return "\n".join(re.split(r"(?<=[.!?])\s+|\n+", text.strip()))


def rouge_scores(predictions, references, metric_names):
    """Returns mean per-report ROUGE F1 scores against full references.

    Args:
        predictions: Generated summary strings in dataset order.
        references: Untruncated reference summaries in the same order.
        metric_names: ROUGE variants to compute.

    Returns:
        A dict mapping each metric name to its mean F1 score.
    """
    scorer = rouge_scorer.RougeScorer(metric_names, use_stemmer=True)
    totals = dict.fromkeys(metric_names, 0.0)
    for prediction, reference in zip(predictions, references, strict=True):
        scores = scorer.score(
            _sentence_lines(reference), _sentence_lines(prediction)
        )
        for name in metric_names:
            totals[name] += scores[name].fmeasure
    return {name: value / len(predictions) for name, value in totals.items()}


def _build_parser():
    """Builds and returns the command-line parser with its default settings.

    Ordinary runs default to four beams; the final experiment used one.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune FLAN-T5 on official LaymanRRG splits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "YAML defaults; CLI options override them. "
            "YAML output_dir is relative to its file."
        ),
    )
    parser.add_argument(
        "--smoke",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use small train/validation prefixes, one epoch and "
            "short sequences; "
            "append /smoke to output_dir."
        ),
    )

    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--fine-tuning-mode", choices=FINE_TUNING_MODES, default="lora"
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-validation-examples", type=int, default=None)

    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument(
        "--output-dir", type=Path, default=Path("training_outputs")
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser


def _read_yaml_defaults(path, parser, defaults):
    """Reads flat YAML defaults and rejects malformed or unknown settings.

    Args:
        path: YAML file path.
        parser: ArgumentParser used to report configuration errors.
        defaults: Original CLI defaults, whose keys define allowed settings.

    Returns:
        A dict of defaults with relative output paths resolved beside the YAML.
    """
    try:
        settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        parser.error(f"Invalid YAML in {path}: {error}")
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        parser.error(f"{path}: settings must be a YAML mapping")
    if any(not isinstance(key, str) for key in settings):
        parser.error(f"{path}: setting names must be strings")

    unknown = sorted(settings.keys() - defaults.keys())
    if unknown:
        parser.error(f"Unknown settings in {path}: {', '.join(unknown)}")
    if isinstance(settings.get("output_dir"), str):
        settings["output_dir"] = (
            path.resolve().parent / settings["output_dir"]
        ).resolve()
    return settings


def _validate_arguments(parser, args, defaults):
    """Validates final values after CLI options override YAML defaults.

    argparse validates explicit CLI values, but YAML defaults can bypass its
    type and choice checks. Use the same option definitions for both sources.

    Args:
        parser: ArgumentParser defining the supported options.
        args: Parsed namespace after combining CLI and YAML values.
        defaults: Original defaults, used to identify nullable/boolean options.
    """
    for action in parser._actions:
        name = action.dest
        if name not in defaults:
            continue
        value = getattr(args, name)
        if value is None:
            if defaults[name] is not None:
                parser.error(f"{name} cannot be null")
            continue

        if isinstance(defaults[name], bool):
            valid = type(value) is bool
            expected = "a boolean (true or false without quotes)"
        elif action.type is int:
            valid = type(value) is int
            expected = "an integer"
        elif action.type is float:
            valid = type(value) in (int, float) and math.isfinite(value)
            expected = "a finite number"
        elif action.type is Path:
            valid = isinstance(value, Path)
            expected = "a path string"
        else:
            valid = isinstance(value, str)
            expected = "a string"
        if not valid:
            parser.error(f"{name} must be {expected}")
        if action.choices is not None and value not in action.choices:
            parser.error(f"{name} must be one of {tuple(action.choices)}")

    positive_counts = (
        "epochs", "batch_size", "eval_batch_size", "max_train_examples",
        "max_validation_examples", "gradient_accumulation_steps",
        "max_input_length", "max_target_length", "max_new_tokens",
        "num_beams", "lora_r",
    )
    for name in positive_counts:
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"{name} must be greater than zero")
    for name in (
        "learning_rate", "max_grad_norm", "weight_decay", "lora_alpha"
    ):
        if getattr(args, name) < 0:
            parser.error(f"{name} must be nonnegative")
    if not 0 <= args.lora_dropout <= 1:
        parser.error("lora_dropout must be between 0 and 1")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be an integer between 0 and 2**32 - 1")


def parse_config(argv=None):
    """Combines command-line arguments and optional YAML into run settings.

    Command-line options override YAML defaults. A YAML output_dir is relative
    to its configuration file. Smoke mode then applies small-run limits and
    appends /smoke to the output directory.

    YAML values are checked after CLI overrides. The returned configuration
    records the seed, but does not guarantee bit-for-bit reproducibility.
    Neither CLI nor YAML exposes model or dataset revision selection;
    programmatic callers must supply revisions to the relevant loaders.

    Args:
        argv: Sequence of argument strings without the program name, or None
            to read the command line.

    Returns:
        A (config, seed) tuple: a dict with model, data and training sections,
        and the integer random seed.

    Raises:
        SystemExit: If arguments are invalid or help is requested.
    """
    parser = _build_parser()
    defaults = vars(parser.parse_args([]))
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config is not None:
        parser.set_defaults(
            **_read_yaml_defaults(preliminary.config, parser, defaults)
        )

    args = parser.parse_args(argv)
    _validate_arguments(parser, args, defaults)

    # Apply smoke limits last so they also constrain YAML/CLI values.
    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 2)
        args.gradient_accumulation_steps = 1
        args.max_input_length = min(args.max_input_length, 128)
        args.max_target_length = min(args.max_target_length, 64)
        args.max_new_tokens = min(args.max_new_tokens, 32)
        args.num_beams = 1
        args.output_dir = args.output_dir / "smoke"

    config = {
        "seed": args.seed,
        "model": {
            "model_name": args.model_name,
            "fine_tuning_mode": args.fine_tuning_mode,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": ["q", "v"],
        },
        "data": {
            "dataset_id": RadiologyData.DATASET_ID,
            "smoke": args.smoke,
            "batch_size": args.batch_size,
            "max_input_length": args.max_input_length,
            "max_target_length": args.max_target_length,
            "prompt_template": DEFAULT_PROMPT_TEMPLATE,
            "eval_batch_size": args.eval_batch_size,
            "max_train_examples": args.max_train_examples,
            "max_validation_examples": args.max_validation_examples,
        },
        "training": {
            "device": str(select_device(args.device)),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "output_dir": str(args.output_dir),
            "bf16": args.bf16,
            "save_every_epoch": args.save_every_epoch,
        },
    }
    return config, args.seed


def main():
    """Runs training from the command line with seeded randomness and logging.

    Loads the tokenizer and model, then writes training artifacts to output_dir.
    """
    config, seed = parse_config()
    set_seed(seed)
    logger = configure_logger(output_dir=config["training"]["output_dir"])
    if config["data"]["smoke"]:
        logger.info(
            "Smoke test: up to %s rows, one epoch; "
            "uncached downloads still required.",
            RadiologyData.SMOKE_ROWS,
        )

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    data = RadiologyData(tokenizer, config["data"])
    model = load_model(**config["model"])
    trainer = RadiologyTrainer(model, data, config)

    logger.info(
        "Trainable parameters: %s / %s",
        *count_parameters(model),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
