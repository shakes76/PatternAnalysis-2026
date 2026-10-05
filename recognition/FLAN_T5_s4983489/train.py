"""运行基线评估、全量微调或 LoRA，并保存结果。

训练接口：https://huggingface.co/docs/transformers/main_classes/trainer
ROUGE 实现：https://github.com/google-research/google-research/tree/master/rouge
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from dataset import load_data, tokenize_data
from modules import count_parameters, load_model

matplotlib.use("Agg")  # 直接保存图片，服务器运行时不需要图形界面。
import matplotlib.pyplot as plt


ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")


def calculate_rouge(predictions, references):
    """返回逐条 F1 的平均值，采用 0–100 分制；Lsum 使用英文分句。"""
    scorer = rouge_scorer.RougeScorer(
        ROUGE_TYPES, use_stemmer=True, split_summaries=True
    )
    scores = [
        scorer.score(reference, prediction)
        for reference, prediction in zip(references, predictions)
    ]
    return {
        name: 100 * sum(score[name].fmeasure for score in scores) / len(scores)
        for name in ROUGE_TYPES
    }


def plot_history(history, output_path):
    """按 epoch 绘制训练/验证损失和四种验证 ROUGE。"""
    training = [entry for entry in history if "loss" in entry]
    validation = [entry for entry in history if "eval_loss" in entry]
    epochs = [entry["epoch"] for entry in validation]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        [entry["epoch"] for entry in training],
        [entry["loss"] for entry in training],
        marker="o", label="Training",
    )
    axes[0].plot(
        epochs, [entry["eval_loss"] for entry in validation],
        marker="o", label="Validation",
    )
    axes[0].set_ylabel("Loss")
    for name in ROUGE_TYPES:
        axes[1].plot(
            epochs, [entry[f"eval_{name}"] for entry in validation],
            marker="o", label=name,
        )
    axes[1].set_ylabel("ROUGE F1 (0–100)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "full", "lora"], required=True)
    parser.add_argument("--output-dir", required=True, help="使用新的目录保存本次实验")
    parser.add_argument("--model-name", default="google/flan-t5-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=512)
    parser.add_argument("--train-samples", type=int, help="仅用于小规模试跑")
    parser.add_argument("--eval-samples", type=int, help="仅用于小规模试跑")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        logging_strategy="epoch",
        # 每轮验证后保存权重、优化器和随机状态，保留最新检查点用于续训。
        save_strategy="epoch",
        save_total_limit=1,
        predict_with_generate=True,
        optim="adamw_torch",
        report_to=[],
        seed=args.seed,
    )
    # 由主进程创建新目录，保留已有实验；其余进程等待创建完成。
    with training_args.main_process_first(local=False):
        if training_args.process_index == 0:
            output_dir.mkdir(parents=True)
            # 开始时就保存实验配置，作业中断后仍能查到原来的训练参数。
            (output_dir / "run_config.json").write_text(
                json.dumps(vars(args), indent=2), encoding="utf-8"
            )

    data = load_data()
    raw_train, raw_eval = data["train"], data["validation"]
    # 试跑从各自官方划分内随机抽样，不把验证数据混入训练集。
    if args.train_samples is not None:
        raw_train = raw_train.shuffle(seed=args.seed).select(range(args.train_samples))
    if args.eval_samples is not None:
        raw_eval = raw_eval.shuffle(seed=args.seed).select(range(args.eval_samples))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    eval_data = tokenize_data(
        raw_eval, tokenizer, args.max_input_length, args.max_target_length
    )
    train_data = None
    if args.mode != "baseline":
        train_data = tokenize_data(
            raw_train, tokenizer, args.max_input_length, args.max_target_length
        )
    model = load_model(args.model_name, use_lora=args.mode == "lora")
    if args.mode == "baseline":
        model.requires_grad_(False)
    # max_length 包含解码起始 token，因此比目标 token 上限多一个。
    model.generation_config.max_length = args.max_target_length + 1
    model.generation_config.num_beams = 1
    model.generation_config.do_sample = False
    total, trainable = count_parameters(model)
    predictions = []

    def compute_metrics(result):
        nonlocal predictions
        # Trainer 汇总不同批次时可能用 -100 补齐；解码前改回 pad token。
        ids = np.where(result.predictions == -100, tokenizer.pad_token_id, result.predictions)
        predictions = tokenizer.batch_decode(ids, skip_special_tokens=True)
        # 使用完整参考文本，而不是训练时可能被截断的 labels。
        return calculate_rouge(predictions, raw_eval["layman_report"])

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics,
    )
    device = training_args.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.mode == "baseline":
        metrics = trainer.evaluate()
    else:
        result = trainer.train()
        # 复用最后一轮的验证结果，避免重复生成整套验证数据。
        evaluations = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
        metrics = {**result.metrics, **evaluations[-1]}
        trainer.save_model(str(output_dir / "model"))
        trainer.save_state()

    if trainer.is_world_process_zero():
        metrics.update(
            total_parameters=total,
            trainable_parameters=trainable,
            device=str(device),
            peak_gpu_memory_mb=(
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda" else None
            ),
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        # 保留原文、完整参考和生成结果，供后续临床错误案例分析。
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
            for row, prediction in zip(raw_eval, predictions):
                record = {
                    "source": row["source"],
                    "images_path": row["images_path"],
                    "radiology_report": row["radiology_report"],
                    "reference": row["layman_report"],
                    "prediction": prediction,
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if args.mode != "baseline":
            # LoRA 的 adapter 文件不包含生成设置，因此单独保存。
            model.generation_config.save_pretrained(output_dir / "model")
            plot_history(trainer.state.log_history, output_dir / "training_curves.png")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
