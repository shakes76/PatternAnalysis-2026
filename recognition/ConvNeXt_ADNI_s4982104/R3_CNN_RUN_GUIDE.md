# R3a：CNN患者级选模入口

本页是新入口的使用说明，不是实验进度记录。R3a只做本地验收；R3b历史轨迹核对及GPU提交材料需另行准备。不要由本页直接启动六次30轮训练。

## 固定设置

`run_r3_cnn.py`复用原`SimpleCNN`、`train.run_epoch`和灰度预处理：1×240×256、像素除以255，无增强。Adam lr=0.001、weight_decay=0、batch=16、FP32、workers=0、不丢末批，无早停或调度。

- A：普通训练CE；B：按训练患者数计算类别权重，顺序NC/AD。真实清单对应[408/550,408/266]。
- 两组验证CE均不加权。患者划分始终用既有清单，不随训练seed改变。
- 训练seed支持42/43/44，初始化、PyTorch/CUDA和训练DataLoader均实际使用该seed。
- CUDA确定性设置沿用原CNN入口；TF32开关不另改，实际后端状态写入摘要，R3b应与历史环境核对。
- 主best：每位患者20张切片概率先取平均，再计算患者NLL，取最低轮。
- 辅best：未加权切片CE最低轮。两条规则均严格改善才替换，同分取较早轮。
- 阈值固定0.5；本入口不进行阈值搜索、校准或测试集评价。

## 本地验收

在项目目录运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_r3_cnn.py --help
```

新增测试使用合成图片，包含真实SimpleCNN与原入口的逐轮等值比较，也包含小模型的快速失败/重载测试。不能把这些结果称为真实MRI/GPU验收。测试集图片不创建；任何误读会失败。

## 入口参数（供下一步准备使用）

```text
run_r3_cnn.py --manifest SPLIT_SAMPLES_CSV --summary SPLIT_SUMMARY_JSON
             --output-dir NEW_PRIVATE_DIRECTORY --arm A|B
             --epochs 3|30 --seed 42|43|44 --device cpu|cuda
```

默认epochs=3、seed=42。每次只执行一个arm/seed，输出目录必须不存在。真实图片需要在计算节点读取。本轮没有Slurm脚本、没有提交作业。

R3b须先增加/准备E3-A42和E4-B42参考文件的来源与轨迹核验流程：初始化、配置、数据、环境、逐轮顺序与原切片指标对应，异常停止；不能仅凭本地通过跳到30轮。保留V1全部结果，不复用其权重作为新初始化。

## 每次输出

```text
config.json
training.log
training_summary.json
epochs.jsonl
selection/
  selection_summary.json
  best_patient_nll.pt
  best_slice_ce.pt
diagnostics/epoch_N/
  slice_predictions.csv
  patient_predictions.csv
  summary.json
reload/patient_nll/（slice_ce结构相同）
  slice_predictions.csv
  patient_predictions.csv
  summary.json
error.log（失败时）
```

逐轮CSV复用原验证前向，无额外读图；含两级指标和错误病例置信度汇总。每个best训练结束后重新构建模型、严格核验配置/来源/权重SHA256，独立预测两遍，并与选中轮的逐切片概率核对（rtol=1e-5、atol=1e-6）。所有患者均与固定验证清单核对。摘要保留源码/输入哈希、实际优化器参数、环境、初始状态哈希、读图数、时间和峰值显存。

| 每次运行 | 训练阶段train | 训练阶段validation | 两个best双遍validation | test |
|---|---:|---:|---:|---:|
| 3轮 | 24,480 | 8,160 | 10,880 | 0 |
| 30轮 | 244,800 | 81,600 | 10,880 | 0 |

两个best即使来自同一轮仍分别完成两遍评价，不省略计数。所有耗时包含加载/处理与核验，不代表纯模型推理延迟。

失败保留已完成轮次、权重和错误摘要，不自动重试或覆盖。R3不要求保存每轮权重，只保存全部逐轮预测与双best。CSV、日志和模型放仓库外；后续权重留集群，私有验收材料不进Git。

## 历史源码核对注意

E3之后，`modules.py`和`dataset.py`增加过ConvNeXt代码，因此当前整文件哈希与E3记录不同；不能把它们宣称为整文件相同。本地已从Git版本`031011a`取出与E3哈希完全匹配的历史文件，确认`SimpleCNN`和`ADNISliceDataset`类的AST与当前一致；`train.py`和`check_dataloader.py`仍与E3整文件哈希一致。R3b应保留这条源码演变证据，并在真实数据/GPU上继续核对轨迹。
