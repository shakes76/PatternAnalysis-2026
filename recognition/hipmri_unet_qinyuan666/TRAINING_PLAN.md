# 正式训练预算与验证规则

本计划在单轮 U-Net pilot 和按 Case 验证后确定。**两模型的单轮试跑和各 20 epoch 正式实验均已完成并核验；测试集尚未评估。**
正式基线按最低验证 loss 选中第 5 轮，验证集前列腺像素汇总 Dice 为 0.8291、Case 等权平均为 0.8090，
实际训练计时为 51.70 分钟。完整结果见 [正式基线报告](FORMAL_UNET.md)。下文保留实验开始前确定的设置与 pilot 依据。

[Improved U-Net 试跑](IMPROVED_UNET_PILOT.md) 已确认本机能训练六类配置：单轮 451.92 秒，
验证前列腺像素汇总 Dice 0.7919、Case 等权平均 0.7625。原估计正式 20 轮约 151 分钟，
实际正式训练计时为 66.70 分钟；单次 pilot 线性估计不能保证耗时。
[正式 Improved U-Net 报告](FORMAL_IMPROVED_UNET.md) 记录第 11 轮权重及验证比较：
像素汇总前列腺 Dice 0.8320、Case 等权平均 0.7934。

## 当前后台任务

由 [run_improved_experiment.py](run_improved_experiment.py) 启动的任务现已完成，状态为 `completed`，不要重复运行训练命令。
脚本按下方固定设置从头训练 20 epochs，成功后自动运行验证集 Case 评估；不评估测试集。
三个编排测试通过：拒绝覆盖已有目录、训练失败后不继续评估、验证结束后才标记完成。
训练及验证结果已核对并整理。以下保留状态查看与复现方法；当前不再需要等待训练。

本机 PowerShell 查看状态与最新训练日志：

```powershell
cd C:\Users\ROG\Desktop\Final\recognition\hipmri_unet_qinyuan666
Get-Content .\runs\real_improved_unet_20ep_s3710\status.json
Get-Content .\runs\real_improved_unet_20ep_s3710\training.log -Tail 5
```

运行期间日志每完成一轮才增加指标；若另行使用 `-Wait` 跟踪日志，`Ctrl+C` 只退出查看，不停止训练。
状态依次为 `training`、`evaluating`、`completed`；异常时为 `failed`。
状态只在阶段切换时更新，不是逐秒心跳。训练错误/警告在 `training_stderr.log`，
验证日志在 `evaluating.log` / `evaluating_stderr.log`；源码和清单哈希在 `launch_provenance.json`。
这些文件均在被 Git 忽略的实验目录内。

如怀疑进程已停止，结合状态中的 `wrapper_pid` 检查：

```powershell
$runStatus = Get-Content .\runs\real_improved_unet_20ep_s3710\status.json | ConvertFrom-Json
Get-Process -Id $runStatus.wrapper_pid -ErrorAction SilentlyContinue
```

未来复现训练期间保持电脑开机、接电且不睡眠，尽量不同时打游戏。强制结束或关机可能来不及更新状态；
脚本不会自动重启或恢复训练，重新运行会拒绝已有目录。
另行复现时指定新目录：`python run_improved_experiment.py --output runs/improved_reproduce`。

## 为什么继续训练

单轮模型在 4 个验证 Case 上的前列腺结果如下。别名仅用于本次报告，
原始病例编号与混淆矩阵保存在本机 `runs/real_unet_pilot/case_validation/case_details_private.json`。

| 验证 Case | 切片数 | 前列腺 Dice | 前列腺 IoU |
|---|---:|---:|---:|
| Case 01 | 60 | 0.7938 | 0.6581 |
| Case 02 | 60 | 0.6366 | 0.4669 |
| Case 03 | 60 | 0.6474 | 0.4786 |
| Case 04 | 480 | 0.7397 | 0.5869 |

按 Case 等权平均 Dice 为 **0.7044**，总体标准差 **0.0654**，没有 undefined 病例。
先将全部切片像素汇总的 Dice 为 **0.7244**。前者给每个 Case 相同权重；
后者由累计像素交并数量决定，含更多切片或目标像素的 Case 影响更大。
每个 Case 内的所有周次一起累计，不是先按扫描或切片求平均，也没有重建 3D 物理体积。
数据分组依据与限制见 [数据记录](DATA_ACCESS.md)。

制定计划时只有一轮训练，无法判断是否收敛；目前已完成两组 20 轮，但仍不能据验证集结果声称达到最终测试要求。
原任务 PDF 第 6 页的目标是测试集前列腺 Dice ≥ 0.75，未明确指定平均方式。
逐 Case 分数及等权平均是本项目用于呈现病例差异的补充，同时保留像素汇总指标。
完整数值及来源哈希见 [病例评估汇总](results/pilot_unet_cases.json)。

## 固定实验设置

| 项目 | 下一阶段设置 |
|---|---|
| 比较 | 标准 U-Net 与 Improved U-Net，各从头训练 20 epochs |
| 数据 | 同一 `course_v1.csv`；保留课程 31/4/3 Case 分组，不改验证/测试分配 |
| 输入 | 六类、前列腺 raw label 5；256×256 pad，现有逐切片归一化 |
| 优化 | Adam，固定学习率 0.001，batch size 4，base channels 16，float32 |
| 随机性 | seed 3710；均不启用数据翻转增强 |
| 验证 | 每个 epoch 遍历全部 660 张验证切片，记录 loss 和像素汇总 Dice |
| 权重选择 | 按最低验证 CE + foreground soft Dice loss 保存 best.pt；相同 loss 保留较早 epoch |
| 病例评估 | 对所选 best.pt 运行 Case evaluator，报告逐 Case Dice/IoU、等权均值和分布 |
| 测试集 | 当前阶段不评估；模型与设置确定后再进行最终测试 |

选择 20 epochs 是有上限的工程预算，并非已经证明最优的轮数。
它让训练曲线有足够观察点，又将第一组正式比较控制在笔记本可承担的范围。
同一 epoch 数意味着两模型看到相同数量的训练样本，不意味着耗时或计算量相同。
不加载 pilot 权重继续训练，以免基线实际比 Improved U-Net 多训练一轮。
如后续调整预算、学习率或增强，应作为另一个明确记录的实验，不能覆盖这组结果。

基线 pilot 每轮训练、验证及保存约 129.56 秒，线性估计 20 轮约 **43 分钟**；
这只是原规划值，实际受设备负载、散热和 I/O 影响；基线正式运行实际为 51.70 分钟。
Improved U-Net 的真实六类单轮耗时为 451.92 秒；从头开始的正式 20 轮实验现已完成。
不能直接把此前两类合成资源测试当作真实六类实验耗时。

## 复现与下一阶段命令

在本项目目录、已激活项目 Python 环境的终端逐条执行；每次使用全新的输出目录。
本机已完成基线 GPU 运行，当前不需要 Rangpur。

基线正式训练已完成。以下是原实验命令；复现时将输出改为新目录，例如 `runs/real_unet_20ep_s3710_reproduce`：

```bash
python train.py --model unet --manifest private_data/manifests/course_v1.csv --label-values 0 1 2 3 4 5 --prostate-label 5 --image-size 256 --resize-mode pad --base-channels 16 --batch-size 4 --epochs 20 --lr 0.001 --seed 3710 --device cuda --output runs/real_unet_20ep_s3710
```

训练结束后评估其最佳权重：

```bash
python evaluate_cases.py --checkpoint runs/real_unet_20ep_s3710/best.pt --manifest private_data/manifests/course_v1.csv --output runs/real_unet_20ep_s3710/case_validation --device cuda --batch-size 4
```

Improved U-Net 独立的一轮试跑已完成，输出位于 `runs/real_improved_unet_pilot`。
以下是已完成的正式实验命令，从头训练且未加载 pilot 权重；复现时同时修改训练和评估的输出路径：

```bash
python train.py --model improved_unet --manifest private_data/manifests/course_v1.csv --label-values 0 1 2 3 4 5 --prostate-label 5 --image-size 256 --resize-mode pad --base-channels 16 --batch-size 4 --epochs 20 --lr 0.001 --seed 3710 --device cuda --output runs/real_improved_unet_20ep_s3710
```

```bash
python evaluate_cases.py --checkpoint runs/real_improved_unet_20ep_s3710/best.pt --manifest private_data/manifests/course_v1.csv --output runs/real_improved_unet_20ep_s3710/case_validation --device cuda --batch-size 4
```
同 seed 不保证两架构初始化相同；PyTorch 已提示部分 CUDA 运算非确定性。

## 本阶段验证范围

新增 5 项针对性测试和现有 9 项指标测试通过。真实评估遍历全部 660 张验证切片，
按 Case 汇总后再合并的像素指标与训练时的验证指标完全一致。
人工 NIfTI 测试包含不存在的 train/test 文件，验证评估工具没有读取这些图像。
新工具只提供验证集模式；病例、周次元数据仍来自固定清单，未据指标重新划分数据。

两模型正式长训练、单轮试跑及病例验证已完成；下一步准备固定测试评估。
最终还需两模型资源比较、边界分析与 3–5 个真实失败案例；未知物理单位不能当成毫米报告。
