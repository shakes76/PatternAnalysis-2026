# 标准 3D U-Net：方法、实验流程与结果记录

更新：2026-10-30（Australia/Brisbane）。本记录由已归档的代码、配置、训练 CSV 和独立测试 JSON/CSV 核对整理；没有重新训练或修改指标。

## 1. 实验目标与当前结论

本项目当前采用 **Normal 难度的标准 3D U-Net**，目标是完成 HipMRI 六类分割，重点评价前列腺。模型不包含残差模块或深监督。最终模型在验证集选定后冻结，在独立测试集的 27 个扫描、6 位患者上得到前列腺逐扫描平均 Dice **0.7732**。这个结果达到本次采用的 0.70 平均 Dice 目标，但不表示每个扫描都超过 0.70。

2D 对照已完成：原生二维测试 Dice 0.825359；共同网格的 2D/3D Dice 为 0.823198/0.773216。2D 的平均重叠更好，但直肠侵入更多；不是 3D 普遍优于 2D 的证据。

## 2. 文件与实验步骤对应

| 实验步骤 | 对应文件 | 实际作用 |
| --- | --- | --- |
| 数据配对、患者划分、预处理 | `dataset.py` | MRI/标签配对；按患者划分；归一化、转轴、降采样 |
| 标准网络结构 | `modules.py` | 三层编码器、瓶颈、三层解码器及跳跃连接；六类 logits |
| 最终模型构造 | `support/baseline_model.py` | 将默认 BatchNorm3d 替换为带 affine 参数的 InstanceNorm3d |
| 初始训练 | `support/train_phase1/scratch.py` | 从头训练，实际跑 15 轮，保留最后权重 |
| 继续训练 | `train.py` | 加载初始阶段最后权重，重新创建优化器，调整损失权重 |
| 完整验证与独立测试 | `evaluate.py`、`support/evaluation_utils.py` | 严格加载冻结模型，计算逐扫描及汇总指标、HD95、侵入计数 |
| 预测图 | `predict.py` | 使用评价缓存绘制 MRI、GT、预测及 TP/FP/FN，无需第二次推理 |
| 原始实验记录 | `support/saved_results/test_results/` | 配置、划分、训练历史、测试指标及五张预测图 |

**入口区别：** 整理后的根目录 `train.py` 就是最终继续训练阶段；初始阶段为 `python -m support.train_phase1.scratch`。早期实现移到 `support/old_code_and_notes/early_train.py`。两阶段实际训练逻辑没有修改，原始提交快照通过 `support/old_code_and_notes/source_snapshots.json` 记录的 Git 版本恢复。

## 3. 数据与预处理：`dataset.py`

- 数据为 211 个扫描、38 位患者，影像位于 `semantic_MRs/`，标签位于 `semantic_labels_only/`，按文件名前缀配对。
- 原始字母前缀患者 ID 排序后用 `random.Random(42)` 打乱；26/6/6 位患者分别用于训练/验证/测试，对应 145/39/27 个扫描。同一患者不同周次不跨组。实际训练与验证清单保存在 `split.json`，测试分组由同一固定规则推导。
- MRI 和标签首先检查尺寸及 affine 是否一致。既有审计记录中 210 个扫描为 `(256,256,128)`，一个为 `(256,256,144)`；统一处理不依赖所有体积原始尺寸相同。
- 只对 MRI 非零体素做逐体积 Z-score，零值背景保留为零；标签不做强度归一化。
- 将 NIfTI 的 `(X,Y,Z)` 转为 PyTorch 的 `(D,H,W)=(Z,X,Y)`。
- MRI 使用 trilinear（`align_corners=False`），标签使用 nearest，统一到 `(64,128,128)`；网络输入为 `[B,1,64,128,128]`。
- 标签为 0..5，项目现有名称依次为 Background、Body、Bones、Bladder、Rectum、Prostate。名称沿用项目映射；计算依赖数字标签。

## 4. 模型：`modules.py` 与 `support/baseline_model.py`

| 阶段 | 输出通道 | 完整体积空间尺寸 D×H×W |
| --- | ---: | --- |
| 编码器 1 | 8 | 64×128×128 |
| 编码器 2 | 16 | 32×64×64 |
| 编码器 3 | 32 | 16×32×32 |
| 瓶颈 | 64 | 8×16×16 |
| 解码器 3 | 32 | 16×32×32 |
| 解码器 2 | 16 | 32×64×64 |
| 解码器 1 | 8 | 64×128×128 |
| 1×1×1 输出卷积 | 6 | 64×128×128 |

每个双卷积块使用两次 3×3×3 卷积、归一化和 ReLU；下采样为 MaxPool3d，上采样为转置卷积，跳跃连接沿通道拼接。`baseline_model.make_model` 将 BatchNorm3d 换成 `InstanceNorm3d(affine=True, track_running_stats=False)`。最终参数量 **351,206**。输出为 logits，预测时取 argmax；无额外后处理。

## 5. 实际训练方案

| 设置 | 实际配置 |
| --- | --- |
| 训练阶段 | 初始 15 轮＋继续训练 11 轮，共 26 轮 |
| 优化器 | AdamW；学习率 0.0005；weight decay 1e-5 |
| 每次更新 | 一个完整体积或一个裁剪块，batch size 1 |
| 采样 | 50% 完整体积、25% 直肠中心块、25% 前列腺中心块；概率为随机选择，非每轮精确配额 |
| 裁剪块 | 32×64×64，中心带随机偏移；指定标签不存在时回退到完整体积 |
| 精度与数据读取 | CUDA BF16 autocast；预处理后缓存到 RAM；1 线程、0 个额外加载 worker |
| 总损失 | 0.5×加权 CE＋0.5×加权前景 soft Dice loss |
| 前景 Dice 权重 | 类别 1..5 为 `[1,1,1,2,2]` |
| 最优权重选择 | 验证集逐扫描平均前列腺 Dice；相同分数时以前景平均 Dice 打破平局 |
| 实际停止原因 | 连续三轮前列腺验证 Dice ≥0.70，且直肠 Dice 非零 |

初始阶段的 CE 权重为 `[1,1,1,2,3,3]`；前列腺验证 Dice 一直为零，但其他结构已经开始学习。继续训练加载的是初始阶段 **第 15 轮 `last_model.pt`**，不是在前列腺分数均为零时停留于早期的 `best_model.pt`。继续训练重建了优化器，不是完整恢复优化器/RNG 的断点续训。

继续训练的 CE 权重只用训练集体素数计算：`sqrt(body_count/class_count)`，限制在 `[0.25,25]`。实际 0..5 类权重为：

`0.7792, 1.0000, 3.3000, 8.1902, 16.0901, 18.8062`。

继续训练最多配置 100 轮；调度器要到新阶段第 30 轮且前列腺 Dice >0.05 才开始，平台早停最早第 60 轮。本次第 11 轮已按连续达标规则结束，所以这些后期机制未触发。不能把本次提升归因于单一权重调整：优化状态、损失和追加训练共同改变，没有对应消融实验。

### 继续训练的验证轨迹（`history.csv`）

| 新阶段轮次 | 累计轮次 | 前列腺验证 Dice | 训练损失 |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 0.0000 | 0.5580 |
| 2 | 17 | 0.0000 | 0.5089 |
| 3 | 18 | 0.0000 | 0.4706 |
| 4 | 19 | 0.0000 | 0.4208 |
| 5 | 20 | 0.0000 | 0.3985 |
| 6 | 21 | 0.4395 | 0.3895 |
| 7 | 22 | 0.6710 | 0.3428 |
| 8 | 23 | 0.3671 | 0.3125 |
| 9 | 24 | 0.7415 | 0.2797 |
| 10 | 25 | 0.7029 | 0.2371 |
| 11 | 26 | 0.7560 | 0.2183 |

第 9/10/11 轮连续达标，最终选第 11 轮，验证 Dice **0.755972**。训练日志的每轮时间包含该轮训练和验证，但不包含开头全部缓存和完整作业开销，不应当作整个训练任务总耗时。

## 6. 独立测试流程与指标定义

评价任务 715670 使用 `evaluation_refine_v2_final/`，读取 `results/refine_v2/best_model.pt`。它核对模型配置并严格加载权重，检查训练/验证清单与患者互斥，采用相同预处理和 BF16 精度。测试期间没有更新权重、调整阈值或应用后处理。

- 每个扫描分别计算各类别 Dice = `2TP/(2TP+FP+FN)`、IoU = `TP/(TP+FP+FN)`，再对有效扫描取平均。
- 单独报告总体 TP/FP/FN 汇总后的 pooled Dice，以及先对同一患者扫描平均、再对患者平均的 patient Dice。这三种平均不能混用。
- 预测与真实都为空时，该类别 Dice/IoU 未定义，不算满分；预测或真实单边为空时，重叠分数为零。
- 前列腺 HD95 是两方向六连通边界距离合并后的第 95 百分位；任一掩码为空时记为未定义并注明状态。本次 27 个扫描都有有效值。
- HD95 使用原始 affine 体素大小按尺寸比例调整到降采样网格后的间距，单位近似为 mm；不是原分辨率临床边界精度。
- 体积比例为预测前列腺体素数／真实前列腺体素数。每扫描可单独统计，总体比例为总预测数／总真实数，不是各扫描比例的算术平均。
- 错误侵入定义为预测标签为前列腺、GT 标签为膀胱或直肠的体素数；它测量与这两个 GT 区域的交叠，不涵盖其他所有假阳性。

### 六类独立测试结果

| 类别 | 名称 | 逐扫描平均 Dice ± SD | 逐扫描平均 IoU | 有效扫描数 |
| ---: | --- | ---: | ---: | ---: |
| 0 | 背景 | 0.9939 ± 0.0025 | 0.9878 | 27 |
| 1 | 身体区域 | 0.9708 ± 0.0093 | 0.9434 | 27 |
| 2 | 骨骼 | 0.8748 ± 0.0200 | 0.7781 | 27 |
| 3 | 膀胱 | 0.9146 ± 0.0190 | 0.8432 | 27 |
| 4 | 直肠 | 0.8197 ± 0.0494 | 0.6972 | 27 |
| 5 | 前列腺 | 0.7732 ± 0.0557 | 0.6336 | 27 |

### 前列腺重点指标

| 指标 | 结果 |
| --- | ---: |
| 逐扫描平均 Dice | 0.773216 |
| 逐患者平均 Dice | 0.767818 |
| 总体汇总 Dice | 0.772990 |
| 平均 HD95（近似降采样 mm） | 6.4995 |
| 总体精确率／召回率 | 0.7147 / 0.8416 |
| 总体预测／真实体积比 | 1.1775 |
| 总体 TP / FP / FN | 19690 / 7859 / 3706 |
| 预测／真实前列腺体素总数 | 27549 / 23396 |
| 误入膀胱／直肠体素数 | 89 / 21 |
| Dice <0.70 的扫描 | 3 / 27 |
| 单扫描 Dice 范围 | 0.6413–0.8850 |

SD 为扫描间标准差，不是置信区间；同一患者多周扫描并非独立样本，不能把 27 个扫描当成 27 位独立患者做统计推断。

## 7. 结果解释与尚未完成的图像分析

预测总体积较 GT 大 **17.75%**，召回率高于精确率，符合仍有过分割倾向。总 FP 为 7859，其中 110 个落在 GT 膀胱或直肠内，占全部前列腺 FP 的 1.40%；因此不能把主要假阳性都解释成侵入这两个器官。

以上数值已结合五张 MRI/GT/预测/TP-FP-FN 图逐张检查，详见 [CASE_ANALYSIS.md](case_analysis.md) 和本项目 [README.md](../../README.md) 的案例分析。它们覆盖低 Dice、边界误差、器官混淆、局部漏分和中位表现，优先不同患者，并非随机样本；选例策略名称不等于全组最差排序。已完成对展示图和原始指标的算法错误分析，未完成独立临床标注质检或全部切片审查。

图里的 inferior/superior extent 来自 GT 首末两层，是端部代理区域，不是独立专家标注的 apex/base 地标。详见 [图与原始结果目录](../saved_results/test_results)。

## 8. 资源记录与可重复性

评价 GPU：NVIDIA A100-PCIE-40GB；参数量：351,206；峰值推理张量分配：0.1364 GiB。前向延迟均值 17.34 ms，中位数 17.55 ms；对第一个样本预热三次，每扫描测三次取中位数。计时只包含 BF16 forward，不含加载、传输、argmax、指标或绘图，也不是训练显存。

Slurm 资源保持原模板：`partition=comp3710`、`account=comp3710`、`gres=gpu`、`time=01:30:00`，没有额外 CPU 或内存申请。

| 软件 | 评价环境版本 |
| --- | --- |
| python | 3.11.15 |
| torch | 2.13.0+cu130 |
| numpy | 2.4.6 |
| nibabel | 5.4.2 |
| scipy | 1.17.1 |
| matplotlib | 3.11.2 |

冻结权重 SHA256：

`77ecc9f051d5d1213c2e62474d582050fe5b425a08182e3d2becc73540737156`

权重保留在集群，GitHub 不包含权重、原始 MRI 或预测缓存。原始 CSV/JSON 可离线检查，但重新推理需要对应私有权重和课程数据。随机种子与配置有记录，不保证不同硬件环境得到逐位相同结果。

## 9. 与 2D 的对比安排

当前完整对比表统一在项目 [README.md](../../README.md)。详细填写及空间对齐规则见
[COMPARISON_PROTOCOL.md](comparison_protocol.md)。完整体积、原生二维和共同网格三种
评价范围分开保存，未完成的 2D／可选 Improved 2D 分数保留空值。

2D 继续使用同一批患者划分，但课程 2D 数据仅覆盖原始 axis 1 的 98..157 切片，并逐切片归一化；3D 使用非零体素逐体积归一化，且已降采样。最终比较应明确这些预处理差异，将两边预测映射到共同空间覆盖和分辨率，采用相同的逐扫描指标定义。

本次 `shared_2d_region_metrics.csv` 提供了 3D 降采样网格的共同切片带统计。该网格上前列腺的 TP/FP/FN 恰好与完整体积统计一致，因此前列腺 Dice 数值相同；不意味着所有类别或其他数据也等价，也不自动使它与原始二维分辨率分数公平可比。

原始共同区域 CSV 未计算 HD95 与器官交叉混淆。本次任务 716656 通过冻结模型独立补算后才填入共同区域表；没有直接把完整体积数值搬过去。评价标签 nearest 映射到 W=49..78，对应原始
Y=98,100,...,156，仅 30 个目标平面；二维的 60 张原生切片得分仍需单独报告。

| 对比项目 | 标准 3D U-Net | 标准 2D U-Net |
| --- | --- | --- |
| 患者划分 | 26/6/6 | 同一患者分组 |
| 训练状态 | 完成，冻结 | 完成，冻结第 5 轮 |
| 原生任务独立测试 Dice | 0.773216（完整降采样体积） | 0.825359（60 张原生切片带，不能直接与左列判胜负） |
| 共同网格逐扫描 Dice | 0.773216 | 0.823198 |
| 结果图 | 原始 5 图及错误分析保留 | 同五例配对图已生成并分析 |

## 10. 报告可用的英文结果段落

A standard 3D U-Net with a base width of 8 and affine instance normalization was trained on 145 scans from 26 patients. Validation used 39 scans from six disjoint patients. After 15 initial epochs and 11 refinement epochs, the checkpoint selected by mean validation prostate Dice (0.7560) was frozen. On 27 held-out scans from six patients, it achieved mean scan-level prostate Dice of 0.7732 (SD 0.0557), mean patient-level Dice of 0.7678, and mean HD95 of 6.50 mm using approximate spacing on the resized grid. Pooled precision and recall were 0.7147 and 0.8416, respectively; the pooled predicted-to-reference volume ratio was 1.1775, indicating an oversegmentation tendency. Three scans had Dice below 0.70. In the aligned shared-grid evaluation, the implemented 2D comparator achieved higher mean scan Dice (0.8232 versus 0.7732), but more rectal invasion (400 versus 21 voxels). This compares two training pipelines with differing capacity and preprocessing and does not isolate convolution dimensionality.

## 11. 提交前剩余事项

- 五张展示图的成功与失败分析已完成，后续与 2D 的相同病例结果联动比较。
- 2D 模型冻结、独立测试及共同空间对比已完成；保留这次结果，不针对测试集调参。
- 将本记录内容整合进最终课程报告，保留完整 AI 使用说明；本文件不是已经排版完成的最终报告。
