'''
Teaching staff confirmed that `transformers` and `peft` may be used to load the model
and to add LoRA; the training loop and the evaluation are written by hand elsewhere.
'''


"""PyTorch FLAN-T5 model construction and LoRA adapter configuration."""
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

MODEL_ID = 'google/flan-t5-base'


def create_model(mode='full', model_id=MODEL_ID, lora_r=8, lora_alpha=16):
    """Return a pretrained model; mode='lora' adds trainable low-rank adapters."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if mode == 'lora':
        config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=lora_r,
                            lora_alpha=lora_alpha, lora_dropout=0.05,
                            bias='none', target_modules=['q', 'v'])
        model = get_peft_model(model, config)
    elif mode == 'zero':
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    elif mode != 'full':
        raise ValueError('mode must be full, lora, or zero')
    return model


def load_checkpoint(mode, checkpoint, model_id=MODEL_ID):
    """Load a full fine-tuned model or LoRA adapter; zero uses pretrained weights."""
    if mode == 'zero':
        return create_model('zero', model_id)
    if mode == 'full':
        return AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    if mode == 'lora':
        return PeftModel.from_pretrained(create_model('zero', model_id), checkpoint)
    raise ValueError(mode)


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total, 'trainable_params': trainable}
