# R4 V2 两阶段统一效率测量：集群运行与验收

本步先严格复现九个患者NLL主best的历史预测，再统一计算设置测效率。不训练、不改阈值、不读取测试图像。706801失败与706846诊断保留原样；新版使用全新 `r4_efficiency_v2_JOBID` 目录。

## 1. 两阶段通过条件

**阶段1：历史复现。**CNN恢复训练摘要记录的backend开关和batch16；ConvNeXt使用原R3代码明确设置的TF32关、batch4，并核对该历史入口源码SHA。均使用eval/no_grad，完整验证1遍。保存 `historical_replay_*` 预测/差异；与原R3患者、切片预测按原R4 `math.isclose(rel_tol=1e-5,abs_tol=1e-6)` 检查，0.5类别相同；失败即停，保留完整证据。

**阶段2：统一测量。**所有模型batch4、TF32关、FP32、inference_mode。统一预热遍生成单独的 `unified_reference_*`。相对原R3的漂移、超差人数、分类变化、准确率/AUROC/F1/NLL/Brier均记录，不把它当成历史复现通过。随后3遍正式计时分别与该统一参考按原容差严格比较，保存 `unified_repeat_1/2/3_*`；超差停止。不覆盖历史参考，不修改容差或best。

总摘要只有两阶段都通过才passed。全九权重严格锁定原清单SHA；新输出 `nine_run_prediction_drift.csv` 独立汇总漂移，效率表保持单独保存。阶段1和审计/保存开销均不计入推理延迟，计入作业资源消耗。

## 2. 固定测量口径

- 同一作业的一张 A100、FP32、关闭 AMP/TF32/compile，batch=4，2 CPU threads，workers=0，不打乱、不丢尾批；每个权重独立进程。
- CNN-A/B 保留灰度 1×240×256；ConvNeXt-F 保留官方预处理的 3×224×224。比较的是已训练的实际流程，不为测速度更改预处理。
- 九个固定主 best：A42/43/44=28/24/26轮；B42/43/44=27/20/25轮；F42/43/44=4/1/30轮。`r4_frozen_bests.json` 锁定权重、训练摘要、选模摘要、原患者预测和划分文件 SHA-256。
- **纯前向：**第一个验证 batch 预先放到 GPU；20次预热；每组连续100次前向，共5组。计时边界均 CUDA 同步。报告每组原始秒数、平均/中位耗时、每batch/每切片耗时、切片吞吐量及显存。排除读图、预处理、传输、softmax、患者汇总。
- **端到端：**先完整预热2720张验证切片，再测完整验证集3遍。包括读JPEG、预处理、CPU→GPU、前向、概率回传、逐切片记录、136位患者平均概率与0.5判断。排除权重加载、审计比较、指标评分和文件写盘。计时覆盖整个流程，前后同步。
- 文件系统/OS缓存预热后测量；没有应用层图像缓存。这不是冷盘性能。每患者耗时为全队列耗时/136，是摊销值，不是单独一个患者请求的延迟。
- 显存分别记录 allocated/reserved 峰值，包含该阶段常驻模型。每阶段预热后重置峰值；前向结束释放输入并清空空闲缓存后进入端到端。reserved 不等于 nvidia-smi 的整进程占用。
- 每权重分别报告重复间波动，再汇总同模型三个种子的均值/样本标准差；不将重复次数当作新增独立患者或模型样本。
- 顺序轮换：A42→B42→F42→B43→F43→A43→F44→A44→B44。记录硬件、软件、作业、源码哈希；不进行速度优劣的预设验收。

## 3. 先同步文件

将最新项目代码同步到：
`/home/Student/s4982104/comp3710-code/ConvNeXt_ADNI_s4982104`

本轮需要同步的代码和测试：
- `run_r4_efficiency.py`
- `run_r4_efficiency.slurm`
- `r4_prediction_audit.py`（新增）
- `tests/test_r4_two_stage.py`（新增）
- `tests/test_r4_efficiency.py`

原 `r4_frozen_bests.json` 保留原样，不编辑；同时同步本指南和 README。共享依赖沿用现有项目，不需重新下载预训练权重。若集群已有全部 R3 代码，只需覆盖上述新增文件及说明；全量同步时排除本机 `.venv`、数据、权重。

输入沿用 `$HOME/comp3710-data/adni`：
- `split_seed42_60_20_20/split_samples.csv` 与 `split_summary.json`
- CNN作业文件夹：`r3_cnn_formal_seed42_705863/run/{A,B}`、`r3_cnn_formal_seed43_705982/run/{A,B}`、`r3_cnn_formal_seed44_705983/run/{A,B}`
- ConvNeXt作业文件夹：`r3_convnext_seed42_706130/run`、`r3_convnext_seed43_706348/run`、`r3_convnext_seed44_706349/run`
- 每个run须保留 `training_summary.json`、`selection/selection_summary.json`、`selection/best_patient_nll.pt`、`reload/patient_nll/patient_predictions.csv`、`reload/patient_nll/slice_predictions.csv`。

路径或哈希不符时停止并保留错误，不重新训练、不改锁定哈希来绕过检查。

## 4. 提交作业

在登录节点：
```bash
cd /home/Student/s4982104/comp3710-code/ConvNeXt_ADNI_s4982104
sbatch run_r4_efficiency.slurm /home/Student/s4982104/comp3710-env/bin/python
```

记录返回的 job ID。脚本先跑完整测试，再在GPU上串行测九个权重。2小时是单次作业保护上限，非预计必用时间。当前不预估精确实际用时，完成后按单GPU作业 `ElapsedRaw/3600` 入账。不要在同一GPU并行启动其他训练或benchmark。

## 5. 回传结果

作业结束后，在登录节点运行（将 `JOBID` 换为数字）：
```bash
sacct -j JOBID --format=JobID,JobName,Partition,AllocTRES,State,ExitCode,Elapsed,ElapsedRaw -P > "$HOME/comp3710-data/adni/r4_efficiency_v2_JOBID_sacct.txt"
```

回传以下两个文件，并告知job ID：
1. `$HOME/comp3710-data/adni/r4_efficiency_v2_JOBID_evidence.tar.gz`
2. `$HOME/comp3710-data/adni/r4_efficiency_v2_JOBID_sacct.txt`

放到本机 `final_report/adni_manifest_local/r4_efficiency_v2/`；解压后保留原作业目录。无需回传 `.pt/.pth` 或 MRI。

压缩包应包括：
- `run/r4_summary.json`、`run/nine_run_efficiency.csv`、`run/model_efficiency_mean_sd.csv`、`run/nine_run_prediction_drift.csv`；
- 九个 `run/模型_种子/efficiency_summary.json`、`historical_replay_*`、`unified_reference_*`、`unified_repeat_*`预测/差异CSV，包含原始计时、显存、读图计数、固定权重哈希和预测一致性；
- 完整测试日志、源码快照、pip freeze、运行及Slurm stdout/stderr、exit_code/job_id、测量前后 nvidia-smi。

如果超时/取消导致未生成压缩包，保留整个证据目录和Slurm日志，回传失败信息，不覆盖原目录重跑。

## 6. 验收标准

1. Slurm COMPLETED、退出0:0；完整测试通过；总摘要与九份摘要均 passed。
2. 九个权重身份/epoch/SHA-256完全对应锁定主best；manifest及预处理源码不变。
3. 同GPU/软件/FP32/batch/协议，九次的参数量和输入尺寸合理；CNN-A/B参数量一致。
4. 每模型前向5组×100次（20次预热），端到端3个2720切片/136患者整遍（1遍预热）；时间正且有限。没有记录为0、NaN、漏测或混入训练。
5. 每模型读取 train=0、validation=13604（历史1遍+统一预热1遍+计时3遍，共5遍×2720，另加探针4张）、test=0；参数不变、无梯度；历史复现与原R3一致，统一计时遍与统一参考一致（均使用原math.isclose、atol=1e-6、rtol=1e-5及0.5类别检查）；统一参考相对历史的差异独立报告，不要求为0、不自动覆盖历史。
6. 根据原始时长独立复算吞吐量、摊销患者耗时、三seed均值/样本标准差；检查重复波动和 nvidia-smi 干扰后再判断比较是否可靠。
7. 资源按作业实测核算；不能将训练峰值显存或历史训练时间混入推理效率表。测试集继续封存。

本地单元测试只证明实现逻辑；真实CUDA测量须待这次集群回传后验收。下一步再确定校准、人工复核及最终测试协议。

## AI 使用记录

本轮由AI协助提出上述具体计时默认值、实现脚本/测试、核对已有checkpoint身份并编写操作说明；用户授权准备R4。没有生成或填充真实GPU测量成绩。实际运行及结果确认由集群证据支撑。
