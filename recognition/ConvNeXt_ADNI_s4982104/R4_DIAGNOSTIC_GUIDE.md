# 作业706801后：CNN-A42单因素数值诊断

## 目的与固定项

定位原R3与R4的患者概率差异。只加载CNN-A42第28轮患者NLL主best。锁定文件原SHA-256为 `d781fb6bec3f9e378aaf6ddfa08229aadb2fc3c7361159f01e87b3be0106ebb4`；权重为 `2bc73f2cd6e0dd19dd303e67bd12754769141c6da0e42bd199e076351dbef106`。不重新训练、不修改权重、不修改原R4脚本或其容差、不使用测试图像。

R4原判定仍为 `math.isclose(p,q,rel_tol=1e-5,abs_tol=1e-6)`。同时独立记录原R3判定 `abs(p-q)<=1e-6+1e-5*abs(q)`，用于解释判定差异，不用它替代R4的通过标准。

## 对照条件

| 条件 | batch | 卷积允许TF32 | 执行模式 | 用途 |
|---|---:|---|---|---|
| A_original |16|开|no_grad|复现原R3 |
| B_tf32_off |16|关|no_grad|相对A只改TF32 |
| C_batch4 |4|开|no_grad|相对A只改batch |
| D_both |4|关|no_grad|组合R4的batch与TF32 |
| E_r4_mode |4|关|inference_mode|相对D只改执行模式，衔接R4 |

五组各用新进程、同一A100、同一数据和权重；每组完整验证两遍。每组应读取 train=0、validation=5440、test=0。2 CPU线程、workers0、矩阵乘TF32关闭、cudnn benchmark关闭、deterministic开启。逐组保存实际backend设置、环境、源码哈希。没有复现R4的计时预热序列；若E仍无法重现706801，下一步检查该执行序列，不能直接认为问题消失。

## 在集群运行

把以下4个新增文件同步到 `/home/Student/s4982104/comp3710-code/ConvNeXt_ADNI_s4982104`（测试文件保留tests子目录）：
- `run_r4_diagnostic.py`
- `run_r4_diagnostic.slurm`
- `tests/test_r4_diagnostic.py`
- `R4_DIAGNOSTIC_GUIDE.md`

沿用已有R4及R3依赖、原 `r4_frozen_bests.json` 和 `$HOME/comp3710-data/adni` 中的原始文件。额外需要原CNN-A42的 `reload/patient_nll/slice_predictions.csv`；脚本会对照已锁定训练摘要中的哈希。无需重新下载预训练权重。

登录节点执行：
```bash
cd /home/Student/s4982104/comp3710-code/ConvNeXt_ADNI_s4982104
sbatch run_r4_diagnostic.slurm /home/Student/s4982104/comp3710-env/bin/python
```

作业上限30分钟，仅为保护上限。先跑完整测试，再运行五组诊断。记录job ID。概率超差不会让任务报错退出；结构/来源/数值无效等错误会失败并保留日志。

## 回传

作业完成后（把JOBID替换为数字）：
```bash
sacct -j JOBID --format=JobID,JobName,Partition,AllocTRES,State,ExitCode,Elapsed,ElapsedRaw -P > "$HOME/comp3710-data/adni/r4_diagnostic_JOBID_sacct.txt"
```

将以下文件回传到本机 `final_report/adni_manifest_local/r4_diagnostic/`，并告知job ID：
1. `$HOME/comp3710-data/adni/r4_diagnostic_JOBID_evidence.tar.gz`
2. `$HOME/comp3710-data/adni/r4_diagnostic_JOBID_sacct.txt`

压缩包包括：总摘要 `run/diagnosis_summary.json`、五组摘要、两遍完整切片/患者预测、每遍对R3的差异、组内重复差异、组间单因素差异CSV、环境/源码/测试日志、Slurm日志和退出码。不回传MRI/权重；预测含患者身份，只保留在私人证据目录，不进入Git。超时未打包时保留证据目录和日志。

## 验收与判读

- 首先验收结构：COMPLETED/0:0、测试通过、五组各两遍预测齐全（每遍2720切片/136患者）、身份/哈希未变、参数不变、无梯度、读图计数正确。
- `status=completed` 仅表示证据收集完整。需要分别查看 `r4_exceed_count`、`r3_exceed_count`、`class_change_count`、最大/平均绝对差；所有逐样本差值均保存，不能只看一个总通过标记。
- 先看A能否复现原R3及各组内重复是否稳定。若A已不一致，不能把原因归结为TF32或batch，应继续核查环境/底层执行。
- 再比较A→B（TF32）、A→C（batch）、C→D和B→D（组合条件）、D→E（执行模式）。同一条件两遍都要检查；组间CSV以第一遍比较，第二遍完整数据可复核。
- 数值超差与分类翻转分开报告；不得据此自动放宽容差、更换best或改写历史R3/R3f成绩。暂不恢复九模型效率作业。

AI协助：实现诊断、差异记录、测试和运行说明。真实A100结果待运行与证据验收；本地测试不代替GPU原因确认。
