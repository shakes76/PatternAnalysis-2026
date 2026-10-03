"""加载 FLAN-T5，并选择全量微调或 LoRA。

LoRA 接入参考：https://huggingface.co/docs/peft/quicktour
"""

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM


def load_model(model_name="google/flan-t5-base", use_lora=False):
    """每次从原始预训练权重加载模型，避免不同实验之间共享修改。"""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if use_lora:
        # T5 的注意力投影层名为 q 和 v；在这些层上加入低秩适配器。
        # r 是低秩矩阵的秩，alpha 控制更新的缩放，dropout 用于训练时正则化。
        config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["q", "v"],
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
        )
        # PEFT 冻结原模型参数，只保留 LoRA 参数参与训练。
        model = get_peft_model(model, config)

    # 不使用 LoRA 时，原模型的全部参数均可训练。
    return model


def count_parameters(model):
    """返回总参数量和可训练参数量，供实验结果表使用。"""
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
