# R2：集群运行与回传

R2c完成运行准备后使用本页。R2d真实训练尚未验收。
固定seed42，F全层微调、H只训练最后Linear；各5轮，其余设置沿用N4：
学习率3e-5、AdamW/wd0.01、batch4、FP32、类别加权和官方预处理。
H保留训练模式和随机深度，classifier归一化层冻结。

## 1. 在Mac同步代码

沿用已有集群环境，不上传Mac的`.venv`。将下面的`R2_REMOTE`替换为你平时使用的SSH主机或别名（包含用户名，如有需要）。

```bash
R2_REMOTE='替换为你的集群SSH目标'
R2_LOCAL='/Users/emily/Desktop/COMP3710/PRA/final_report/PatternAnalysis-2026/recognition/ConvNeXt_ADNI_s4982104'
rsync -av --exclude='__pycache__/' --exclude='*.pyc' \
 "$R2_LOCAL/run_r2.py" "$R2_LOCAL/run_r2.slurm" \
 "$R2_LOCAL/patient_evaluation.py" "$R2_LOCAL/dual_checkpoint.py" \
 "$R2_LOCAL/tests" \
 "$R2_REMOTE:/home/Student/s4982104/comp3710-code/ConvNeXt_ADNI_s4982104/"
```

其余模块沿用已验收N4版本。新入口会核对N4记录的源码哈希，不要随意更新旧训练文件或集群依赖。也可用原来的文件传输工具同步这些文件；tests是整个目录。

## 2. 在集群登录节点提交

必须先进入项目目录。脚本使用Slurm提交目录定位代码，不要在其他目录提交。

```bash
cd /home/Student/s4982104/comp3710-code/ConvNeXt_ADNI_s4982104
sbatch run_r2.slurm \
 /home/Student/s4982104/comp3710-env/bin/python \
 /home/Student/s4982104/comp3710-data/adni/convnext_stability_lr3e5_702225/run/training/training_summary.json
```

记下返回的作业ID。参考文件是N4的training_summary.json，不是N3或N5摘要；若此前移动过N4目录，仅替换第二个参数为其当前真实路径。

脚本申请comp3710账户/分区、1张A100、2CPU，最多2小时。F/H合计0.6–1GPU-hours是估算，实际消耗以作业记录为准。它会运行完整测试，再顺序运行F和H。F的N4前缀不匹配时，H不会启动。

每组保留5份逐轮模型和2份best，合计14份权重，模型张量约1.6GB；另留日志和预测空间。权重保留集群，不打包回Mac。

## 3. 查看完成状态

在集群上将下面数字替换成此次作业ID：

```bash
R2_JOB_ID=替换为作业ID
squeue -j "$R2_JOB_ID"
sacct -j "$R2_JOB_ID" --format=JobID,State,ExitCode,Elapsed,AllocTRES -P
```

结果目录为：
`/home/Student/s4982104/comp3710-data/adni/r2_diagnostic_<作业ID>/`

- `test_results.txt`：完整测试日志。
- `run_stdout.txt`、`run_stderr.txt`：训练入口输出。
- `run/F/training.log`、`run/H/training.log`：逐轮进展。
- `run/r2_summary.json`：整组结果，成功必须为`status: passed`。
- `exit_code.txt`：成功应为0；同时检查Slurm为COMPLETED、ExitCode为0:0。

结束时自动生成同级`r2_diagnostic_<作业ID>_evidence.tar.gz`，包含源码快照、测试日志、配置、逐轮预测/指标、双best重载结果和退出码，排除`.pt/.pth`权重。失败时也尽量打包；强制终止可能来不及打包，此时保留原目录和Slurm日志用于排查。

作业结束后另存最终调度状态（结束前无法在自动包中得到最终sacct状态）：

```bash
R2_JOB_ID=替换为作业ID
sacct -j "$R2_JOB_ID" --format=JobID,State,ExitCode,Elapsed,AllocTRES -P \
 > "$HOME/comp3710-data/adni/r2_diagnostic_${R2_JOB_ID}/scheduler_status.txt"
```

## 4. 回传到Mac

在Mac终端执行；R2_REMOTE为实际SSH目标，R2_JOB_ID为实际数字：

```bash
R2_REMOTE='替换为你的集群SSH目标'
R2_JOB_ID=替换为作业ID
R2_DEST='/Users/emily/Desktop/COMP3710/PRA/final_report/adni_manifest_local/r2_diagnostic'
mkdir -p "$R2_DEST"
scp "$R2_REMOTE:/home/Student/s4982104/comp3710-data/adni/r2_diagnostic_${R2_JOB_ID}_evidence.tar.gz" "$R2_DEST/"
scp "$R2_REMOTE:/home/Student/s4982104/comp3710-data/adni/r2_diagnostic_${R2_JOB_ID}/scheduler_status.txt" "$R2_DEST/scheduler_status_${R2_JOB_ID}.txt"
tar -xzf "$R2_DEST/r2_diagnostic_${R2_JOB_ID}_evidence.tar.gz" -C "$R2_DEST"
```

下载完成后告诉AI目录和作业ID即可。若失败也回传日志，先查原因，不直接重复提交。

## 5. R2d验收清单

- 两组各5轮成功，初始状态、输入、环境、配置和样本顺序匹配。
- F前3轮loss/未加权loss与N4一致（rtol1e-5、atol1e-6），顺序哈希及轮次一致；同时独立复核分类指标和患者预测。
- H每轮骨干无梯度、参数不变，分类头更新；训练模式/随机深度未另改。
- 患者NLL与切片CE分别选最小值，同分取较早轮；两个best均重载两次并对上所选轮预测。
- 每组训练读取40800、训练期验证13600、重载验证10880、测试0；两组合计训练81600、验证48960、测试0。
- 比较完整5轮患者NLL、两类召回、balanced accuracy、AUROC、Brier和错误置信度；记录时间、显存、总GPU-hours。
- 结果只用于决定下一步V2实验方案，不直接替换V1正式成绩，不启用封存测试集。

AI使用记录：AI协助准备R2入口、测试、批处理及本运行指南；本地验证使用合成数据和模拟进程，未代替真实ConvNeXt/GPU验收。所有真实效果以集群证据为准。
