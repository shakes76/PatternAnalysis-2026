# R3d ConvNeXt-F 本地入口

本阶段只完成本地实现和合成测试，不提交集群。旧 R2/N5 及 CNN 入口保持不变。

## 固定协议

`run_r3_convnext.py` 每次从原预训练重新初始化并完整训练30轮，支持训练seed42/43/44，患者划分仍为seed42。沿用R2-F：全层训练、官方236短边缩放/224裁剪/RGB归一化、batch4、FP32、AdamW lr3e-5/wd0.01、训练患者平衡权重、无增强/早停/调度器。保持train模式及随机深度0.1。

主权重 `selection/best_patient_nll.pt` 按患者平均概率后的NLL选择；辅助权重 `selection/best_slice_ce.pt` 按不加权切片CE选择。均严格小于才更新，同分取早轮。只保留两份best，不保存30份逐轮权重；逐轮切片/患者预测全部保留。配置、预训练/输入/代码哈希、实际优化器、初始化、环境和可训练参数均写入摘要。

## 历史核对

所有seed必须传入各自N5的 **training目录**；seed42还必须传入R2的 **run/F目录**。开始前核对已通过状态、配置、连续历史、输入/源码哈希；初始化后核对模型初态、环境/GPU、预训练与优化器设置。

每轮对照相应N5的训练/验证全部已有指标、数量与顺序；seed42前5轮另核对R2-F的训练/验证及患者指标。整数/顺序必须完全一致；浮点采用事先固定的 `abs(a-b) <= 1e-6 + 1e-5*abs(b)`。漂移立即停止并保留已完成轮次、错误和摘要，不继续凑满30轮。结束后确认历史文件未改动。

本地合成数据运行使用小模型验证流程；真实MRI、CUDA轨迹与显存尚未验收。缺少历史参考仅允许显式传入 `expected_lengths=None` 的合成API调用；CLI真实队列不提供此绕过选项。

## 后续计算节点运行命令（本轮未执行）

先同步新入口、此说明与完整tests，核对既有 run_n5.py/run_r2.py/run_n4.py/run_n3.py/train.py/predict.py/modules.py/dataset.py/check_dataloader.py/check_convnext.py/patient_evaluation.py/dual_checkpoint.py 与本地一致。不要复制Mac的.venv。

下面命令只能在已申请的计算节点运行。先完成seed42验收再安排43/44。输出目录必须全新，不接受续训或覆盖。

```bash
cd /home/Student/s4982104/comp3710-code/ConvNeXt_ADNI_s4982104
PYTHON_BIN=/home/Student/s4982104/comp3710-env/bin/python
DATA_ROOT="$HOME/comp3710-data/adni"
export OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PYTHON_BIN" run_r3_convnext.py \
  --manifest "$DATA_ROOT/split_seed42_60_20_20/split_samples.csv" \
  --summary "$DATA_ROOT/split_seed42_60_20_20/split_summary.json" \
  --weights "$HOME/comp3710-data/pretrained/convnext_tiny-983f1562.pth" \
  --weights-summary "$HOME/comp3710-data/pretrained/convnext_tiny_weights_summary.json" \
  --n5-reference "$DATA_ROOT/convnext_formal_seed42_702352/run/training" \
  --r2-reference "$DATA_ROOT/r2_diagnostic_704838/run/F" \
  --seed 42 --device cuda \
  --output-dir "$DATA_ROOT/r3_convnext_seed42_${SLURM_JOB_ID:?}/run"
```

seed43/44的N5参考分别为 `convnext_formal_seed43_702999/run/training`、`convnext_formal_seed44_703004/run/training`，不传R2参考。Slurm提交包装和实际提交留到下一步。

## 读取预算及回传

每次30轮：训练244800、验证81600、测试0。两个best各独立重载两次：额外验证10880。总计训练244800、验证92480、测试0；每轮诊断复用验证前向，额外读图0。两份best相同也执行规定的四次评价，不改变验收计数。

回传 `config.json`、`training_summary.json`、`epochs.jsonl`、`training.log`、`selection/selection_summary.json`、全部 `diagnostics/epoch_*/{summary.json,slice_predictions.csv,patient_predictions.csv}` 和 `reload/{patient_nll,slice_ce}/` 同名三文件。失败时保留 `error.log` 及所有已有文件。后续集群还需作业号、退出码、测试日志、Slurm状态与输出；权重留集群，患者CSV只放仓库外私人证据目录。

本地验证：`OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v`。
