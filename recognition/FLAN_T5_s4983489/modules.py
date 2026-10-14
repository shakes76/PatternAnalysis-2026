"""Load FLAN-T5 for full fine-tuning or LoRA.

LoRA integration reference: https://huggingface.co/docs/peft/quicktour
"""

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM


def load_model(model_name="google/flan-t5-base", use_lora=False):
    """Load a fresh model instance from the specified checkpoint."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if use_lora:
        # Add low-rank adapters to T5's q and v attention projections.
        # r sets the rank, alpha scales the update, and dropout regularizes training.
        config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["q", "v"],
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
        )
        # PEFT freezes the base model and trains only the LoRA parameters.
        model = get_peft_model(model, config)

    # Without LoRA, all base model parameters remain trainable.
    return model


def count_parameters(model):
    """Return total and trainable parameter counts for the results table."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


if __name__ == "__main__":
    for use_lora in (False, True):
        model = load_model(use_lora=use_lora)
        total, trainable = count_parameters(model)
        method = "LoRA" if use_lora else "Full fine-tuning"
        print(f"{method}: {total:,} total, {trainable:,} trainable parameters")
        del model
