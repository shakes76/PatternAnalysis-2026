# 运行顺序

在 `recognition/YOLO_SkinLesion_s4849402/` 目录执行。以下真实训练命令尚需完整数据；不要把合成测试数据结果当成模型性能。

## 1. 环境

若 CUDA 张量运算通过但 cuDNN 卷积报动态库加载失败，在当前 Linux 终端补充安装库目录，再测试真实卷积；不要立即重装学校 PyTorch：

```bash
export LD_LIBRARY_PATH="$(python - <<'PY'
import sys
from pathlib import Path
folders = [str(p.resolve()) for root in sys.path for p in Path(root).glob('nvidia/*/lib') if p.is_dir()]
print(':'.join(dict.fromkeys(folders)))
PY
)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c "import torch; m=torch.nn.Conv2d(3,16,3,padding=1).cuda(); m(torch.randn(2,3,64,64,device='cuda')).square().mean().backward(); torch.cuda.synchronize(); print('Convolution forward/backward passed')"
```

该设置仅作用于当前终端。仍报错时检查缺失依赖，不盲目关闭 cuDNN 或改变实验参数。

学校已有可用 PyTorch/CUDA 时保留它，在项目环境安装 `python -m pip install -r requirements.txt`，然后 `python check_environment.py --training`。本地 `.venv` 只用于 CPU 软件验证，不上传。

## 2. 准备数据

将官方训练、验证图像包与 Part1 masks 解压到 `/path/to/isic2017_raw`，同时放入对应 `Part3_GroundTruth.csv`。脚本递归查找文件名；重复文件名会报错，不随意选一个。

用户于 2026-10-29 反馈已确认可沿用 ISIC 官方划分。项目使用该协议并保留确认说明，程序本身没有患者映射，也不声称核验了患者身份。

```bash
python prepare_isic.py --source /path/to/isic2017_raw --output /path/to/isic2017_yolo --split-evidence "User reports tutor confirmation on 2026-10-29: use official ISIC 2017 splits"
```

默认只准备 train/val；准备最终测试数据时提供所有对应文件并显式加 `--include-test`，输出到一个新目录。不要修改 train/val 成员。测试集标签不参与参数或阈值选择。

这会保存 `manifest_official.csv`、YOLO 标签和 `audit.json`；检查官方类别与划分、空 mask、尺寸和重复解码图像，不伪造患者 ID。若真实数据发现重复，先调查并记录统一处理决定，不直接关闭检查。

### 已发现的训练/验证重复

学校运行发现 `ISIC_0012970`（train）与 `ISIC_0012127`（val）解码像素一致，诊断均为 nevus。用户全量扫描反馈仅一组重复。保留验证集原成员，从训练清单排除重复训练图；原图不删除。预期处理后 train=1999、val=150。

在导入命令中增加 `--duplicate-audit ~/comp3710-data/isic2017/duplicate_audit.json`。导入器重新核对报告成员、划分、像素和类别，只自动处理训练与单个 held-out 成员的同类重复组。其他重复和类别冲突仍报错；排除记录、来源报告及哈希写入输出目录，后续全部模型使用同一处理后的划分。

## 3. 检查模型

```bash
python modules.py --weights yolov8n.pt
```

输出三类名称、检测头类别数、参数量及软件版本。第一次运行可能下载预训练权重；它不是已经训练好的病灶检测器。

## 4. 短训与正式对比

```bash
python train.py --data /path/to/isic2017_yolo/data.yaml --output runs/smoke640 --epochs 2 --imgsz 640 --batch 8 --device 0
python train.py --data /path/to/isic2017_yolo/data.yaml --output runs/baseline640 --epochs 50 --imgsz 640 --batch 8 --device 0
python train.py --data /path/to/isic2017_yolo/data.yaml --output runs/compare960 --epochs 50 --imgsz 960 --batch 8 --device 0
```

每次从相同预训练起点重新训练。50 epochs/batch 8 是暂定预算，按短训测量后统一确定。输出目录不能预先存在，避免混淆实验。训练没有自动测试步骤。正式对比不启用早停；保存默认验证 fitness 选择的 best.pt，并保留 results.csv、args.yaml 和自定义实验记录以核对该版本选择规则。

## 5. 预测与评估

```bash
python predict.py --weights runs/baseline640/weights/best.pt --source /path/to/images --output runs/predictions640 --imgsz 640 --conf 0.25 --device 0
python evaluate.py --weights runs/baseline640/weights/best.pt --data /path/to/isic2017_yolo/data.yaml --output runs/validation640 --split val --imgsz 640 --threshold-sweep --device 0
```

0.25 仅为演示分数阈值。验证集扫描默认输出描述性曲线（status=descriptive），比较逐类 Precision/Recall 和每图误报，不附加作业未规定的召回率合格门槛，也不自动选阈值。文件名仍为 threshold_selection.json。定位按作业明确的 IoU >= 0.8 目标评估，同时保留标准 mAP50、mAP50:95 和分类指标。

```bash
python evaluate.py --weights runs/baseline640/weights/best.pt --data /path/to/isic2017_yolo/data.yaml --output runs/test640 --split test --imgsz 640 --device 0
```

测试数据需先准备好；锁定实验方案之后才运行。测试集禁止 `--threshold-sweep`。mAP 使用低置信度截断形成 PR 曲线，不使用高部署阈值。网络速度辅助函数只测前向计算，不包括 NMS、读盘和可视化。

要报告固定工作点的 Precision/Recall 和 IoU >= 0.8 检出比例，额外传入 `--operating-conf`，其数值必须来自已锁定的验证集选择。例如已选定 0.35 时加 `--operating-conf 0.35`；这个数字仅为用法示例，不是已选结果。

## 6. 本地逻辑检查

### 补充平均定位 IoU（不重跑推理）

```bash
python summarize_iou.py --data ~/comp3710-data/isic2017/test_yolo/data.yaml --predictions runs/test640_final/test_predictions.json --output runs/test640_final/localization_iou.json --conf 0.15
python summarize_iou.py --data ~/comp3710-data/isic2017/test_yolo/data.yaml --predictions runs/test960_final/test_predictions.json --output runs/test960_final/localization_iou.json --conf 0.15
```

读取现有预测和标签，输出 IoU >= 0.5 正确类别匹配框的条件均值，以及把漏检记为 0 的全 GT 均值。二者都明确给出分母，不能混作 IoU >= 0.8 检出率。不会使用测试集改阈值或模型。

### 准备最终测试数据（不复制训练图片）

在测试协议锁定后运行；测试原文件名含 `Test_v2`，不需改名。

```bash
python prepare_test.py --archives ~/comp3710-data/isic2017/archives --raw ~/comp3710-data/isic2017/test_raw --training-data ~/comp3710-data/isic2017/yolo --output ~/comp3710-data/isic2017/test_yolo
```

脚本读取 600 条官方测试标签，仅校验/解压对应图像与病灶 mask。对照实际使用的训练/验证图片计算解码哈希，将重叠测试样本及测试集内部重复隔离并记录；标签冲突则停止。原始数据和已训练模型不修改。若排除了测试样本，最终成绩必须标注为排重后的测试子集。

通过同文件系统硬链接将测试图像导出，输出目录不能预先存在；不支持硬链接时会报错，不偷偷复制大图。解压中断后可复用 CRC 校验一致的原始文件。查看输出 `audit.json` 后再运行测试；评估 `--data` 使用新的 `test_yolo/data.yaml`，两组使用同一测试清单和已固定阈值。详见 `reports/TEST_PROTOCOL.md`。

### 使用已有预测做失败案例与固定阈值分析

先检查 `runs/val640_analysis/`、`runs/val960_analysis/` 中是否有 `val_predictions.json`（旧版本可能名为 `validation_predictions.json`，按实际文件传参）。不需要重新加载模型或消耗 GPU 推理。

```bash
python analyze_failures.py --data ~/comp3710-data/isic2017/yolo/data.yaml --predictions runs/val640_analysis/val_predictions.json --output runs/failures640_conf015 --conf 0.15
python analyze_failures.py --data ~/comp3710-data/isic2017/yolo/data.yaml --predictions runs/val960_analysis/val_predictions.json --output runs/failures960_conf015 --conf 0.15
```

输出 `failure_summary.json`（包括 IoU 0.5/0.8 工作点）、`all_cases.json`、最多五张候选叠加图和待人工填写的 `case_review.md`。该工具明确限定为验证集、一图一 GT 病灶。症状分类不能自动证明毛发/低对比度等物理原因。

0.15 是查看验证结果后提出、在测试前固定的探索性共同工作点，不是作业规定或经临床验证的阈值。历史 JSON 中的旧约束及 met/unmet 状态保留作为实验记录，不作为作业是否达标的判断。完整进度见 `reports/PROJECT_PLAN.md`。

```bash
python -m unittest discover -s tests -v
```

这些测试使用合成图片/框，不是患者数据或真实 benchmark。完整训练、595 张排重测试子集评估和失败案例复核已完成，真实结果与计时范围见 README。
