"""读取并编码 LaymanRRG 数据，供 FLAN-T5 训练使用。

分词与批次填充参考：https://huggingface.co/docs/transformers/tasks/summarization
"""

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq


DATASET_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
TASK_PREFIX = (
    "Explain this radiology report in plain English for a patient. "
    "Preserve the findings, negation, and uncertainty:\n"
)


def load_data():
    """返回官方数据集，保留原有的 train、validation 和 test 划分。"""
    return load_dataset(DATASET_NAME)


def tokenize_data(data, tokenizer, max_input_length=512, max_target_length=512):
    """编码一个数据划分，返回 input_ids、attention_mask 和 labels。"""

    def tokenize_batch(batch):
        # 编码器只接收任务提示和原报告，不接收参考答案。
        inputs = [TASK_PREFIX + report for report in batch["radiology_report"]]
        encoded = tokenizer(
            inputs, max_length=max_input_length, truncation=True
        )

        # 参考解释单独编码，作为解码器学习的目标。
        targets = tokenizer(
            text_target=batch["layman_report"],
            max_length=max_target_length,
            truncation=True,
        )
        encoded["labels"] = targets["input_ids"]
        return encoded

    # 此处不填充；组批时再补齐，避免每条短文本都占用 512 个位置。
    # map 返回新数据，原始 data 中的完整文本仍保留，供后续评估使用。
    return data.map(
        tokenize_batch, batched=True, remove_columns=data.column_names
    )


if __name__ == "__main__":
    data = load_data()

    # 显示官方划分的样本数，方便确认数据已读入。
    for split_name, split in data.items():
        print(f"{split_name}: {len(split)} samples")

    # 输入是专业放射报告，训练目标是同一条记录中的通俗解释。
    example = data["train"][0]
    print("\nRadiology report (input):")
    print(example["radiology_report"])
    print("\nLayperson report (target):")
    print(example["layman_report"])

    # 本步只加载分词器，用两条真实样本展示一个批次，不加载模型权重。
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    examples = tokenize_data(data["train"].select(range(2)), tokenizer)

    # 输入补齐为 pad token；目标补齐为 -100，使填充位置不参与损失计算。
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)
    batch = collator([examples[0], examples[1]])
    print("\nTokenized batch:")
    for name, tensor in batch.items():
        print(f"{name}: {tuple(tensor.shape)}")
    print(f"Ignored target padding positions: {(batch['labels'] == -100).sum().item()}")
