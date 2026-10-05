"""加载已保存的实验，为新报告或官方数据生成通俗解释。

LoRA 重载：https://huggingface.co/docs/peft/developer_guides/checkpoint
生成设置：https://huggingface.co/docs/transformers/main_classes/text_generation
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from peft import PeftModel
from transformers import AutoTokenizer, GenerationConfig

from dataset import TASK_PREFIX, load_data
from modules import load_model
from train import calculate_rouge


def load_saved_model(run_dir, device):
    """根据实验记录加载基线、完整模型或原始模型加 LoRA 适配器。"""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    model_dir = run_dir / "model"
    if config["mode"] == "baseline":
        model = load_model(config["model_name"])
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        # 基线没有训练权重目录，按实验记录恢复与训练脚本相同的生成设置。
        model.generation_config.max_length = config["max_target_length"] + 1
        model.generation_config.num_beams = 1
        model.generation_config.do_sample = False
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if config["mode"] == "lora":
            # 适配器只包含 LoRA 权重，必须接回原始预训练模型。
            base_model = load_model(config["model_name"])
            model = PeftModel.from_pretrained(base_model, model_dir)
        else:
            model = load_model(model_dir)
        model.generation_config = GenerationConfig.from_pretrained(model_dir)
    model.to(device).eval()
    return model, tokenizer, config


@torch.inference_mode()
def generate_report(model, tokenizer, report, max_input_length):
    """生成一条解释，并测量生成耗时和输出 token 数。"""
    inputs = tokenizer(
        TASK_PREFIX + report,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    # CUDA 操作异步执行；计时前后同步，才能测得实际生成耗时。
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    start = perf_counter()
    # 显式传入设置，保证 LoRA 底层模型也使用保存的生成长度等参数。
    output = model.generate(**inputs, generation_config=model.generation_config)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = perf_counter() - start

    # 第一个 token 是解码起始符；统计其后的非 padding token，包含 EOS。
    generated_tokens = (output[0, 1:] != tokenizer.pad_token_id).sum().item()
    return {
        "prediction": tokenizer.decode(output[0], skip_special_tokens=True),
        "generated_tokens": generated_tokens,
        "generation_seconds": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="train.py 保存的实验目录")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="直接输入一条英文放射报告")
    inputs.add_argument("--split", choices=["validation", "test"])
    parser.add_argument("--samples", type=int, help="从官方划分中按实验种子抽样；省略时使用全部")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", help="结果 JSON 的新文件路径；省略时直接显示结果")
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer, config = load_saved_model(args.run_dir, args.device)
    if args.text is not None:
        records = [{"radiology_report": args.text}]
    else:
        data = load_data()[args.split]
        if args.samples is not None:
            data = data.shuffle(seed=config["seed"]).select(range(args.samples))
        records = []
        for row in data:
            record = {
                "source": row["source"],
                "images_path": row["images_path"],
                "radiology_report": row["radiology_report"],
            }
            # 官方测试集不公开参考答案，仅验证集可计算 ROUGE。
            if args.split == "validation":
                record["reference"] = row["layman_report"]
            records.append(record)

    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    for record in records:
        record.update(generate_report(
            model, tokenizer, record["radiology_report"], config["max_input_length"]
        ))

    seconds = sum(record["generation_seconds"] for record in records)
    tokens = sum(record["generated_tokens"] for record in records)
    metrics = {
        "reports": len(records),
        "generated_tokens": tokens,
        "generation_seconds": seconds,
        "seconds_per_report": seconds / len(records),
        "tokens_per_second": tokens / seconds,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(model.device) / 1024**2
            if model.device.type == "cuda" else None
        ),
    }
    if args.split == "validation":
        metrics.update(calculate_rouge(
            [record["prediction"] for record in records],
            [record["reference"] for record in records],
        ))
    result = {
        "run_dir": str(Path(args.run_dir).resolve()),
        "mode": config["mode"],
        "split": args.split,
        "device": str(model.device),
        "seed": config["seed"],
        "metrics": metrics,
        "predictions": records,
    }
    if args.output is not None:
        # x 模式创建新文件，保留之前已经保存的预测结果。
        with Path(args.output).open("x", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        print(json.dumps(metrics, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
