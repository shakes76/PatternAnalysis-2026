# Python 代码蓝图

更新：2026-10-29 对应 `.py` 已实现并通过合成 CPU 集成检查。本文保留为最初接口草图；实际接口与用法以 `.py` 和 `RUN_GUIDE.md` 为准。

这是接口设计，代码块中的 `NotImplementedError` 表示后续需要实现。不是可训练的完整代码，不会替换现有数据脚本。

## 目录

```text
YOLO_SkinLesion_s4849402/
├── modules.py          # 模型配置、构建、训练后权重加载
├── dataset.py          # 已有数据检查/导出；下面展示计划统一的接口
├── train.py            # 训练和验证，保存权重与实验信息
├── predict.py          # 新图预测、保存检测框和可视化
├── evaluate.py         # 评估、验证集阈值选择、资源统计
├── README.md           # 报告与使用说明
├── reports/            # 可提交的小型报告图表
├── data/               # 原始/处理数据，Git 忽略
└── runs/               # 训练日志和权重，Git 忽略
```

## 1. modules.py：模型层

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLASS_NAMES = ("melanoma", "seborrheic_keratosis", "nevus")

@dataclass(frozen=True)
class DetectorConfig:
    weights: str = "yolov8n.pt"
    class_names: tuple[str, ...] = CLASS_NAMES
    seed: int = 42

def build_detector(config: DetectorConfig) -> Any:
    """构建训练起点，返回 Ultralytics YOLO 对象。

    1. 固定初始化随机种子。
    2. 加载 COCO 预训练网络。
    3. 将检测头真正适配成三类并迁移兼容权重。
    4. 检查检测任务、类别数和名称顺序。
    注意：只修改 names 不等于修改检测头。
    """
    raise NotImplementedError

def load_trained_detector(weights: Path) -> Any:
    """预测时加载已训练的 best.pt；检查三类映射，不重新初始化检测头。"""
    raise NotImplementedError

def model_summary(detector: Any) -> dict:
    """返回参数量、可训练参数量、类别、stride 和库版本等信息。"""
    raise NotImplementedError
```

Backbone、Neck、Detect Head 和默认检测损失复用 Ultralytics/PyTorch；不在这里凭空重复实现 YOLO。保留第三方来源声明，模型组件适配深度需满足 tutor 的要求。

## 2. dataset.py：数据层

```python
from pathlib import Path

def mask_to_box(mask) -> tuple[float, float, float, float]:
    """mask 前景 -> 归一化的中心 x、中心 y、宽、高（已有实现）。"""
    raise NotImplementedError  # 蓝图占位；实际文件已有函数体

def audit_manifest(manifest: Path, class_count: int):
    """校验患者隔离、重复图像、图片/mask 尺寸与类别（已有实现）。"""
    raise NotImplementedError

def export(manifest: Path, output: Path, names: list[str]):
    """导出 images/、labels/、data.yaml 和审查报告（已有实现）。"""
    raise NotImplementedError
```

正式导出的输入必须符合已确定的患者/数据划分协议。患者编号缺失目前未解决，不能用 image_id 伪造。训练增强交给框架的数据加载器，对图像和框同步变换；验证与测试不随机增强。

## 3. train.py：训练入口

```python
from dataclasses import dataclass
from pathlib import Path
from modules import DetectorConfig, build_detector, model_summary

@dataclass(frozen=True)
class TrainConfig:
    data_yaml: Path
    output_dir: Path
    image_size: int = 640
    epochs: int = 50
    batch_size: int = 8
    device: str = "0"
    learning_rate: float = 0.001
    seed: int = 42

def validate_training_data(config: TrainConfig) -> dict:
    """检查路径、三类顺序、train/val 分离及数据划分审查记录。"""
    raise NotImplementedError

def run_training(model_config: DetectorConfig, train_config: TrainConfig) -> Path:
    """返回 best.pt 路径；只使用 train/val，不执行 test。

    validate_training_data(train_config)
    detector = build_detector(model_config)
    保存配置、split 哈希、版本和 model_summary(detector)
    detector.train(...): AdamW、固定设置、默认 CIoU+BCE+DFL
    保存 best.pt、last.pt、曲线、时间和显存记录
    """
    raise NotImplementedError

def main():
    """解析命令行参数，构建配置，调用 run_training。"""
    raise NotImplementedError

if __name__ == "__main__":
    main()
```

Baseline 用 640，对比实验用 960；其余设置一致，每次重新从同一预训练起点训练。正式训练预算暂定 50 epochs，需短训测量后固定；不是已经训练过的参数。

## 4. predict.py：推理入口

```python
from pathlib import Path

def predict_images(weights: Path, source: Path, output_dir: Path,
                   image_size: int, confidence: float, device: str) -> list[dict]:
    """加载训练权重，预测图片；不接收真实框或诊断作为模型输入。

    load_trained_detector(weights)
    detector.predict(source=..., imgsz=..., conf=..., device=...)
    保存原图坐标框、类别、分数及叠加图
    返回可 JSON 序列化结果；零个检测也保留空列表
    """
    raise NotImplementedError

def main():
    """命令行传入权重、图片、输出目录及推理参数。"""
    raise NotImplementedError

if __name__ == "__main__":
    main()
```

## 5. evaluate.py：实验评估

```python
from pathlib import Path

def evaluate_detector(weights: Path, data_yaml: Path, split: str,
                      image_size: int, output_dir: Path) -> dict:
    """显式选择 val/test，计算总体及逐类 mAP、定位和检出指标。

    AP 使用低分数截断保留 PR 曲线，不能套用较高的部署阈值。
    匹配规则明确类别和一对一框匹配；漏检计入召回分母。
    """
    raise NotImplementedError

def choose_threshold(validation_predictions: list[dict],
                     validation_ground_truth: list[dict], policy: dict) -> dict:
    """只在验证集比较 Precision/Recall/误报，按预定规则选择阈值。"""
    raise NotImplementedError

def profile_inference(detector, example_tensor, warmup: int, repeats: int) -> dict:
    """GPU 热身/同步后记录速度和显存，声明计时范围。"""
    raise NotImplementedError
```

## 实现顺序

1. 实现并验证 modules.py 的模型构建、三类检测头和前向输出。
2. 确认划分协议，补齐正式数据导入，保留现有校验。
3. 实现 train.py，学校 GPU 上短训验证保存/加载。
4. 实现 predict.py，检查真实预测与空预测。
5. 实现评估和对比实验，将真实证据写入报告。

Blueprint drafted with Codex assistance. No training or model inference has been run by this document.
