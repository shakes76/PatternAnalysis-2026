"""Load FLAN-T5 for full fine-tuning or LoRA."""

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM

DEFAULT_MODEL_NAME = "google/flan-t5-base"
FINE_TUNING_MODES = ("full", "lora")


def load_model(
    model_name=DEFAULT_MODEL_NAME,
    fine_tuning_mode="lora",
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=("q", "v"),
    model_revision=None,
):
    """Loads a pretrained model, optionally adding new LoRA adapters.

    Full checkpoints use "full" mode. For saved LoRA adapters, use
    RadiologyPredictor.from_checkpoint(). Only the model weights are loaded.

    Args:
        model_name: Hub ID (str) or full checkpoint directory (str or PathLike).
        fine_tuning_mode: Mode string: "full" or "lora".
        lora_r: Integer adapter rank.
        lora_alpha: Integer alpha; update scaling is lora_alpha / lora_r.
        lora_dropout: Float adapter dropout probability.
        lora_target_modules: Iterable of module-name strings, e.g. ("q", "v").
        model_revision: Hub branch, tag or commit (str), or None for default.

    Returns:
        A transformers.PreTrainedModel or, in LoRA mode, a peft.PeftModel.

    Raises:
        ValueError: If fine_tuning_mode is unsupported.
    """
    if fine_tuning_mode not in FINE_TUNING_MODES:
        raise ValueError("fine_tuning_mode must be 'full' or 'lora'.")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, revision=model_revision
    )

    if fine_tuning_mode == "lora":
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                bias="none",
            ),
        )

    return model


def trainable_parameters(model):
    """Returns the model parameters that receive gradients.

    Args:
        model: PyTorch model, optionally with frozen base weights.

    Returns:
        A list of parameters whose requires_grad flag is True.
    """
    return [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]


def count_parameters(model):
    """Counts trainable and total model parameters.

    Args:
        model: PyTorch model whose parameters are counted.

    Returns:
        A (trainable, total) tuple of integer parameter counts.
    """
    trainable = sum(
        parameter.numel() for parameter in trainable_parameters(model)
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total
