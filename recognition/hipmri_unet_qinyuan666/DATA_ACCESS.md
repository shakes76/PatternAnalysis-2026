# 第 1 步：获取 HipMRI 原始数据与说明

本文件记录真实数据来源、原始标签说明、下载检查和后续输入准备。
数据准备后的单轮真实 U-Net 试跑与 20 epoch 正式基线已完成，见 [试跑记录](REAL_DATA_PILOT.md)
和 [正式基线报告](FORMAL_UNET.md)；两模型比较仍待进行。

## 官方来源与课程数据的关系

原任务文件 `COMP3710_Report_2026_draft_v2.2.pdf` 第 17 页给出的
HipMRI 链接是 [DOI 10.25919/45t8-p065](https://doi.org/10.25919/45t8-p065)。
它对应 CSIRO 的 [Labelled weekly MR images of the male pelvis](https://data.csiro.au/collection/csiro:51392)。
已通过官方接口核对 Version 2：

- [集合元数据](https://data.csiro.au/dap/ws/v2/collections/52549)。
- [文件清单](https://data.csiro.au/dap/ws/v2/collections/52549/data)。
- [原始 ZIP 下载](https://data.csiro.au/dap/ws/v2/collections/52549/data/58895460)。
- [BriefDataDescription.txt](https://data.csiro.au/dap/ws/v2/collections/52549/support/5034)。

集合的元数据版本是 2，但其数据文件名仍是
`HipMRI_study_complete_release_v1.zip`；文件名本身不能用来判断集合版本。
官方页面标明文件公开，许可为 CC BY-NC-ND 4.0。保留原始文件、许可和来源引用，
数据及标注不上传项目 GitHub 仓库。

**官方原始包包含 3D 体积，并不包含课程的 `keras_slices_data` 目录。**
当前模型仍然是 2D；得到这个原始包不等于已经得到课程预处理的 2D 切片，
也不代表现在可以直接把 3D 文件交给现有 2D 数据加载器。

## 本地文件与获取状态

以下路径均相对于本项目目录，位于 Git 忽略的 `private_data/` 中：

```text
private_data/
  downloads/HipMRI_study_complete_release_v1.zip
  hipmri_csiro_v2/
    semantic_MRs_anon/
    semantic_labels_anon/
  keras_slices_data/  # 六个课程切片目录，已完成本地检查
  source_metadata/
    BriefDataDescription_v2.txt
    download_receipt.json
```

目录索引已核对：211 个 MRI 体积、211 个对应标注，按文件名中的 Case/Week
键可以配成 211 对，包含 38 个 Case 编号组和 Week0–Week7。
这些是原始 3D 体积的数量，不能当作 2D 切片数量。

下载和解压已完成：压缩包为 1,373,932,069 字节，与官方清单一致；ZIP CRC
校验通过，解压得到 422 个 NIfTI 文件。保存的 SHA-256 是本地计算的记录，
没有声称与官方公布的独立哈希比对。

另外按文件名排序选取首、中、末三对体积作获取后的基本检查：三对均为
`256×256×128`，图像与 mask 的形状和 affine 一致，图像及 mask 均为有限值，
mask 原始值均为整数 `0–5`。这不是所有体积的完整审计，也没有计算模型准确率。

完整汇总见 [results/data_source.json](results/data_source.json)。
本地原始报告保留在 `private_data/source_metadata/`。

## 原始数据的标签字典

以下映射直接来自上述 CSIRO `BriefDataDescription.txt`：

| 原始值 | 官方名称 | 含义 |
|---|---|---|
| 0 | Background | 背景 |
| 1 | Body | 身体 |
| 2 | Bones | 骨骼 |
| 3 | Bladder | 膀胱 |
| 4 | Rectum | 直肠 |
| 5 | Prostate | 前列腺 |

这是官方原始数据的定义。项目现采用 [labels.json](labels.json) 中的同一套映射，
依据是官方说明及三个不同训练病例的精确对应检查，结果见
[标签核对记录](results/label_verification.json)。三个样本均包含全部 `0–5`，课程图像
和完整标注数组都与同病例/周次原始体积的某个平面逐元素一致；没有做标签重编码或插值。
因此标签 4、5 的核对不再只依赖之前缺少这两类的样本。验证/测试集没有用于选择这些例子。
标签 4/5 的叠加对照图保存在私有目录 `private_data/source_metadata/label_4_5_overlays.png`。
此结论支持项目标签字典，不等于逐张重做医学标注或验证全数据所有平面的来源。
不能使用论文列举类别时的顺序编号来替代 NIfTI 原始值。

官方说明指出，Week0 在治疗前采集，Week1–Week7 是治疗期间的每周扫描。
原始文件采用下面的配对形式：

```text
semantic_MRs_anon/Case_<编号>_Week<周数>_LFOV.nii.gz
semantic_labels_anon/Case_<编号>_Week<周数>_SEMANTIC_LFOV.nii.gz
```

这些模式来自实际压缩包目录。病例编号与患者身份的映射依据应在正式划分前记录清楚；
同一病例的不同 Week 及其切片不能分到不同数据集。当前没有生成训练/验证/测试划分。

## Rangpur 课程目录：已确认与待核对内容

用户提供的终端截图显示主机为 `login0.compute.eait.uq.edu.au`，
`python3 --version` 输出 `Python 3.14.7`，且课程目录确实存在。
该 Python 版本仅作为截图观测记录，不表示已经验证项目依赖或训练环境。
本次依据用户终端输出记录；助手此前的非交互 SSH 连接在 22 端口超时。

`/home/groups/comp3710/HipMRI_Study_open` 下观察到：

```text
data_use_agreement.txt
keras_slices_data/
semantic_labels_only/
semantic_MRs/
```

`keras_slices_data` 下的以下六个目录现已全部下载并检查内容。
按病例编号的跨目录检查结果见后文；目录名称本身不作为患者隔离的证据。

| 分组名称 | 图像目录 | 分割标注目录 |
|---|---|---|
| train | `keras_slices_train` | `keras_slices_seg_train` |
| validate | `keras_slices_validate` | `keras_slices_seg_validate` |
| test | `keras_slices_test` | `keras_slices_seg_test` |

### 早期训练文件的配对线索

用户提供了两个训练目录各自 `ls -1 ... | head -5` 的输出。观察到的通用格式为：

```text
keras_slices_train/case_<病例编号>_week_<周次>_slice_<切片编号>.nii.gz
keras_slices_seg_train/seg_<病例编号>_week_<周次>_slice_<切片编号>.nii.gz
```

这五组名称在病例、周次和切片编号上对应，只有 `case_` / `seg_` 前缀不同。
后续清单应按这三个字段配对，不能仅把目录排序后按行配对。
文件名记录覆盖五组示例；其中一对已下载并成功读取为 NIfTI，检查结果见下文。
病例编号是患者分组的线索；后续全量检查验证了这些编号无跨分组重叠，
但没有独立核实病例编号与实际患者身份的映射。

### 课程数据使用说明

已收到 `head -80 /home/groups/comp3710/HipMRI_Study_open/data_use_agreement.txt`
的终端截图。以下记录限于截图可见内容，尚未取得说明文件本身：

- 不尝试识别或联系数据参与者；不得用于商业用途。
- 对使用 HipMRI 或其派生数据的公开成果，需要致谢数据来源，并引用所列相关论文中的至少一篇。
- 数据不提供正确性保证；违反使用条款会终止访问和使用权利。
- 截图列出了 [MRI-alone treatment planning 论文](https://doi.org/10.1016/j.ijrobp.2015.08.045)
  和 [image synthesis and translation 论文](https://arxiv.org/abs/2011.13615)。
  分割条目后未显示链接，因此不补造该条目的引用。

[项目 README](README.md) 已加入 HipMRI 数据致谢、数据集 DOI 和上面的 2015 年论文引用。
此处记录课程说明，不将它替代 CSIRO 原始数据包的许可记录。

### 早期一对课程训练切片：本地检查结果

本地已准备 `private_data/course_sample/` 接收目录，并确认受 Git 忽略规则保护。
在本机 **Windows PowerShell** 中使用 `scp` 下载已观察到的一对训练图像和标注；
不要在远程 `s4906920@login0` 终端中使用 Windows 本地路径。传输命令必须使用实际文件名，
上面的尖括号格式只是命名示意，不能原样执行。用户已完成一对样本传输，助手已读取本地文件。

| 检查 | 本次一个样本的结果 |
|---|---|
| 图像 / 标注形状 | 均为 `256×128`，符合 2D 数组要求 |
| 文件数据类型 | 图像 `float32`，标注 `uint8` |
| 图像 / 标注 affine | 相同，均为单位矩阵 |
| 有限值 / 整数标注 | 检查通过 |
| 观察到的标注值 | `0、1、2、3`；不能据此认定全数据只有四类 |
| 现有 `load_nifti_pair` 读取 | 通过；没有建立训练 manifest 或运行模型 |
| 与本地原始体积的对应 | 在同病例/周次体积中找到图像和标注均逐元素完全一致的平面，需选择平面并翻转一个方向 |

文件头的空间单位为 `unknown`；不能把单位矩阵中的 `1` 直接解释为真实的 `1 mm`。
因此这个样本通过了图像/标注文件一致性检查，但尚不能据此做毫米单位的边界测量。
在此样本上，匹配平面沿原始数组第 2 维切出，再将结果第 1 维反向；
课程文件名中的切片编号不等于找到的原始数组索引。因此不能只按同名切片编号截取原始体积。
此结果只验证了一个样本的来源及其已有标签值，没有建立全数据转换规则。
单个样本不能确定全局标签字典，也不能证明课程切片保留了原始体积的空间坐标或标签语义。
带文件名、文件哈希和详细检查的私有记录位于 `private_data/course_sample/sample_validation_private.json`；
公开汇总见 [results/data_source.json](results/data_source.json)。这是一对样本的历史检查记录；
整个课程切片集的后续本地检查见下一节。

原始切片转换流程与物理空间信息仍未完全恢复。后续标签字典和病例分组采用的依据见
本文件的标签核对与训练清单章节。可用 [只读预检脚本](inspect_rangpur.py) 做远程目录预检，
操作见 [RANGPUR.md](RANGPUR.md)。仅汇总通用格式进入仓库，具体患者文件名保持私有。

### 完整课程目录：下载后本地检查

用户通过 `scp` 下载，并在连接中断后用 SFTP `reget -R` 续传，已报告传输结束。
本地路径为 `private_data/keras_slices_data/`。目录包含 25,320 个文件，
总文件字节数为 629,732,130；远端 `du -sh` 截图的 `795M` 是磁盘占用，
不作为逐文件字节数或完整性的校验依据。

| 课程分组 | 图像/标注对数 | 文件名中的病例编号数 | 病例/周次组合数 |
|---|---:|---:|---:|
| train | 11,460 | 31 | 191 |
| validate | 660 | 4 | 11 |
| test | 540 | 3 | 9 |
| 合计 | 12,660 | 38 | 211 |

每个病例/周次组合有 60 对切片。三组病例编号的两两交集均为零；全部 211 个
病例/周次键也与本地官方原始体积的文件名键集合一致。这是命名和分组证据，
不等于已核实每张切片的原始平面、真实患者身份或源文件哈希。

[完整检查结果](results/course_download_audit.json) 记录：

- 25,320 个文件全部通过完整 gzip 流读取（包括 CRC/长度尾部校验）与 NIfTI 读取。
- 12,660 对均按病例/周次/切片键配对；没有缺失标注、重复规范化键或异常目录项。
- 每对均为二维数组，图像/标注形状和 affine 一致，数据为有限值，标注为整数。
- 12,600 对形状为 `256×128`，另 60 对为 `256×144`；后续已实现统一补边，见下一节。
- 图像均为 `float32`，标注均为 `uint8`；全数据观察到的标注值为 `0–5`，后续标签字典采用依据见前述三个训练病例的核对记录。
- 所有 affine 均为单位矩阵，空间单位均为 `unknown`，不能据此采用毫米单位指标。

没有远端逐文件清单或独立哈希可用于比对，因此结论是**本地现有下载的完整性检查通过**，
不声称完成了远端逐文件一致性校验。下载检查本身不生成训练 manifest、不改变课程分组，也不运行模型。

新增 [audit_course_download.py](audit_course_download.py) 可复现该检查；六项人工数据测试
覆盖正常配对、缺失标注、截断压缩文件、跨周病例重叠、affine 不一致和空目录。
在项目 Python 环境中，从本项目目录执行以下单行命令：

```bash
python audit_course_download.py --root private_data/keras_slices_data --output results/course_download_audit.json --private-output private_data/source_metadata/course_download_audit_private.json
```

公开报告仅含汇总；实际文件名、病例编号列表、SHA-256 和错误详情保留在受 Git 忽略的私有报告中。

## 已准备的训练清单与输入处理

[prepare_course_manifest.py](prepare_course_manifest.py) 已对全部 25,320 个文件重新计算 SHA-256，
确认与下载检查时一致，生成本地 `private_data/manifests/course_v1.csv`。
[公开汇总](results/data_preparation.json) 记录清单哈希、标签字典哈希和各组数量。
CSV 内采用相对路径，仍为训练 11,460、验证 660、测试 540 对；课程 `validate` 只更名为程序接受的 `val`。

`patient_id` 采用数据集原始 Case 编号作为去标识化分组单位，保持同病例的所有周次与切片在同一组。
采用依据是课程三组病例编号无重叠、211 个病例/周次键与官方原始体积键集合一致，
以及训练样本内容的来源对应。这是项目记录的病例分组约定，不代表独立识别了真实患者身份。
没有重新随机划分、剔除困难切片或依据测试集表现选择数据。

新训练的预处理模式默认是 `pad`，本项目真实数据使用 `--image-size 256 --resize-mode pad`：

- 每张图只用自身非零像素计算 z-score，并截断到 `[-5,5]`；不跨分组拟合统计量。
- `256×128` 左右各补 64 列，`256×144` 左右各补 56 列，均得到 `256×256`。
- 这批数据在该设置下不发生空间缩放，所有原始标注像素保留；补边的图像值与背景类别均为 0。
- 通用尺寸输入采用同一比例缩放，图像双线性、标注最近邻；模式写入 checkpoint，预测时复用。
- 不含该模式字段的旧 checkpoint 按原来的 `stretch` 方式读取，避免改变旧实验的预处理。

[实际输入检查](results/preprocessing_check.json) 覆盖训练/验证例子及两种原始尺寸；
前景标签像素数完全保留，真实训练输入可组成 `[4,1,256,256]` 批次。
上述输入检查阶段没有对真实 MRI 执行模型前向或优化器步骤；后续单轮训练见 [试跑记录](REAL_DATA_PILOT.md)。没有访问测试集进行模型选择。
软件验证为 52 项回归测试及随后新增的 4 项清单测试，均通过。

以后重建清单可在项目目录运行 `python prepare_course_manifest.py`。
同一清单路径拒绝覆盖不同内容；若确需改划分或源数据，应另起版本并保留实验对应关系。

## 以往实验文件的用途

提供的 `COMP3710_Lab_2_2026_v2.01.pdf` 包含 Fourier/PCA/CNN、CIFAR10
及 OASIS 脑图像任务；`COMP3710_Lab_1.3.py` 是 Gaussian、sine 和 Gabor
示例。它们可参考课程概念和 GPU 使用方式，但没有提供 HipMRI 标签或患者映射。
Lab 2 的 OASIS 下载地址及现场演示要求不替代本项目任务文件中的 HipMRI 要求。

## 后续步骤的边界

数据获取与本地完整性检查、项目标签字典、保留课程病例分组的训练清单和统一输入预处理已完成。
随后已在本机完成一个 epoch 的真实 U-Net 试跑及 20 epoch 正式基线，
结果分别见 [试跑记录](REAL_DATA_PILOT.md) 和 [正式报告](FORMAL_UNET.md)。
所选权重的四个验证 Case 已评估；下一步是 Improved U-Net 真实训练及匹配的模型比较。
仍未完成全量切片来源对应、物理空间恢复、测试集评估或失败案例分析。
原有中期高 Dice 图来自合成数据；真实试跑和正式基线分别使用 `pilot_unet` 和 `formal_unet` 结果名，曲线使用 `real_` 前缀。
