"""
Model initialisation for radiology report summarisation.

Supports two fine-tuning approaches:
1. Full fine-tuning: update all FLAN-T5 parameters.
2. LoRA fine-tuning: freeze the pretrained model and train
   low-rank adapters attached to attention projections.
"""
import logging

from utils import setup_logging
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, TaskType, get_peft_model

from dataset import MODEL_NAME

logger = logging.getLogger(__name__)

def load_full_model():
    """
    Load FLAN-T5 Base for full fine-tuning.

    All model parameters remain trainable.

    Returns:
        PreTrainedModel: FLAN-T5 sequence-to-sequence model.
    """

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Enable gradient updates for all model parameters.
    model.requires_grad_(True)

    return model

def load_lora_model(rank=8, alpha=16, dropout=0.05):
    """
    Load FLAN-T5 Base with LoRA adapters.

    LoRA introduces trainable low-rank matrices into selected
    attention projections while freezing the pretrained weights.

    Args:
        rank (int): Rank of the low-rank adaptation matrices.
        alpha (int): LoRA scaling factor.
        dropout (float): Dropout probability applied to LoRA inputs.

    Returns:
        PeftModel: FLAN-T5 model with LoRA adapters.
    """

    # Load the same pretrained checkpoint used for full fine-tuning.
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Configure LoRA adapters for T5 attention query/value projections.
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q", "v"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    # Attach LoRA adapters and freeze the original model parameters.
    model = get_peft_model(base_model, lora_config)

    return model

def count_parameters(model):
    """
    Count total and trainable model parameters.

    Args:
        model: PyTorch model.

    Returns:
        tuple[int, int]: Total and trainable parameter counts.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_parameters, trainable_parameters

def inspect_model(model, method):
    """Log model parameter statistics."""

    total, trainable = count_parameters(model)
    percentage = 100 * trainable / total

    logger.info("%s Model", method)
    logger.info("Total parameters: %s", f"{total:,}")
    logger.info("Trainable parameters: %s", f"{trainable:,}")
    logger.debug("Frozen parameters: %s", f"{total - trainable:,}")
    logger.info("Trainable percentage: %.4f%%", percentage)

    return total, trainable

if __name__ == "__main__":
    setup_logging()

    full_model = load_full_model()
    full_total, full_trainable = inspect_model(
        full_model,
        "Full Fine-Tuning",
    )

    del full_model

    lora_model = load_lora_model()
    lora_total, lora_trainable = inspect_model(
        lora_model,
        "LoRA Fine-Tuning",
    )

    assert full_total == full_trainable
    assert lora_trainable < lora_total
    assert lora_trainable > 0

    logger.info("Model initialisation completed successfully")