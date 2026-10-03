"""读取作业指定的 LaymanRRG 数据集，并展示一组报告与通俗解释。"""

from datasets import load_dataset


DATASET_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"


def load_data():
    """返回官方数据集，保留原有的 train、validation 和 test 划分。"""
    return load_dataset(DATASET_NAME)


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
