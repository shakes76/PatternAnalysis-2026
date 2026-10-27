# 依赖、路径与复现边界

本页说明依赖、路径与复现边界；正式报告和项目入口见<a href="README.md">README</a>。

本页记录已有运行与文档环境及复现边界。依赖版本来自已安装环境的元数据；下列安装步骤是尚未执行的模板，不代表新环境安装成功。

## 直接依赖与本机记录

<table>
<tr><th>直接依赖</th><th>本机已安装版本</th><th>安装元数据要求的 Python</th><th>代码用途</th></tr>
<tr><td>NumPy</td><td>2.5.4</td><td>≥3.12</td><td>数组、归一化检查与记录</td></tr>
<tr><td>NiBabel</td><td>5.4.2</td><td>≥3.10</td><td>NIfTI 文件与头信息</td></tr>
<tr><td>PyTorch</td><td>2.11.0+cu128</td><td>≥3.10</td><td>张量、数据批次、模型及训练／推理</td></tr>
<tr><td>Matplotlib</td><td>3.11.2</td><td>≥3.11</td><td>曲线、对照图和聚合图</td></tr>
</table>

实测解释器为 Windows 64 位 CPython 3.13.2。主文件 <a href="requirements.txt">requirements.txt</a> 固定这四个直接依赖的公共版本；其中 torch==2.11.0 不能指定或证明 CPU／CUDA 构建。<a href="requirements-cuda-observed.txt">requirements-cuda-observed.txt</a> 继承主文件，再要求本机观察到的精确版本 2.11.0+cu128。两者版本约束相容，不表示任何安装源已经提供了适用轮子。

这两份文件不是所有间接依赖的完整锁文件。pip 仍需解析并获取它们的间接依赖；本机各包的 Requires-Dist 及本轮核查边界保存在 <a href="evidence/submission_environment_20261026.json">环境证据 JSON</a>。没有验证 CPU 构建、其他操作系统、其他 Python 版本或跨环境数值一致性。+cu128 是已安装发行包的版本标记；该标记本身不证明另一环境的 CUDA 运行时或 GPU 可用。

## 完整已安装依赖快照与文档环境

补充：<a href="requirements-observed-lock.txt">requirements-observed-lock.txt</a>记录原CPython3.13.2运行环境的23包active依赖闭包，含精确间接版本；它来自安装元数据与markers检查，不是新机器安装成功的记录。前述两份文件仍是较小的direct／CUDA约束，三者用途分别保留。

PDF与新架构图使用独立的bundled CPython3.12.14文档环境，<a href="requirements-docs.txt">requirements-docs.txt</a>固定其完整3包闭包：ReportLab4.4.9、Pillow12.3.0、charset-normalizer3.5.2。原运行环境的Pillow为13.0.0，不与文档环境混用作同一份实测记录。两环境的Requires-Dist、marker决策和精确版本已保存在<a href="evidence/submission_observed_lock_20261026.json">依赖闭包证据</a>。

只审阅保存证据无需安装上述科学包；Git和Python标准库即可运行verify_submission_archive.py。保留历史的只读核验和PDF构建示例已合并至<a href="README.md">README的复现说明</a>。换机器安装与跨平台数值复现仍未验证。

## 离线安装准备示例

以下PowerShell模板从仓库根目录使用。需要预先取得适用于Windows 64位CPython 3.13的完整本地wheel目录，并替换示例路径；本仓库不提供轮子，也没有核验其下载来源、可获得性或兼容性。

```powershell
py -3.13 -m venv .venv-submission
$python = (Resolve-Path '.\.venv-submission\Scripts\python.exe').Path
$wheelhouse = 'C:\wheelhouse\hipmri-py313-win_amd64'
& $python --version
& $python -m pip install --no-index --find-links $wheelhouse -r recognition/hipmri_2d_improved_unet/requirements-cuda-observed.txt
& $python -m pip check
```

py -3.13 可能选中其他 3.13 补丁版本；若要匹配本机记录，应事先提供并确认 Python 3.13.2。没有完整轮子目录时，这组离线命令会失败。若另选后端，可将依赖文件换为 requirements.txt，但必须自行确认实际安装的 torch 构建并记录差异；主文件不保证自动选择 CPU，也不保证得到本实验的 CUDA 构建。依赖检查通过仍不能证明模型训练、推理或指标跨环境一致。

## 独立文档环境安装模板

以下仍从仓库根目录使用，与上面的CPython3.13模型环境分开。需要适用于CPython3.12的完整文档wheel目录；本机实际构建解释器是3.12.14，py -3.12可能选其他补丁版本，需自行确认--version。本组使用既有bundled环境，没有运行下面的新环境安装。

```powershell
py -3.12 -m venv .venv-docs
$docPython = (Resolve-Path '.\.venv-docs\Scripts\python.exe').Path
$docWheelhouse = 'C:\wheelhouse\hipmri-docs-py312-win_amd64'
& $docPython --version
& $docPython -m pip install --no-index --find-links $docWheelhouse -r recognition/hipmri_2d_improved_unet/requirements-docs.txt
& $docPython -m pip check
```

没有适用的完整wheel目录时命令会失败；本仓库未提供轮子，也未验证新机器安装。文档环境只用于本地文本／图片构建，模型环境和模型运行授权不因此改变。

## 数据、权重与入口

提交整个 recognition/hipmri_2d_improved_unet 目录，保留 dataset.py、modules.py、train.py、predict.py、README.md，以及 losses.py、metrics.py、checkpoint_selection.py、resource_monitor.py 等辅助模块。只复制任务书列出的五个文件会缺少导入依赖。用脚本文件路径调用入口；没有验证以 python -m 包名方式启动。

数据、.venv、outputs 和 .pt／.pth／.ckpt 权重由 .gitignore 排除；普通 clone 不包含它们。必须通过课程许可渠道另行准备数据，需要权重的运行还要提供对应的外部 checkpoint。当前脚本默认数据位置依赖原机器的 comp3710 目录结构，换位置后应显式传入 --data-root。它指向包含以下目录的数据根目录，而不是某个单独的 MRI 文件夹。

```text
keras_slices_data/
  keras_slices_train/
  keras_slices_seg_train/
  keras_slices_validate/
  keras_slices_seg_validate/
  keras_slices_test/
  keras_slices_seg_test/
```

train.py读取训练／验证两组；predict.py保留training／validation，并增加默认关闭、需显式--allow-test-example的test样例入口。该test分支本轮仅构造检查，真实test样例没有执行。两者均有 --data-root、--device 和 --output-dir；predict.py 还需要 --checkpoint。输出目录必须是新目录。单样本 predict 的批次组成不同于冻结的全验证／测试批次，不能用它替代正式分数。入口参数示例模板如下；尖括号为需替换的参数，省略号表示其余明确选择的参数，因此不是直接执行命令。

```text
python recognition/hipmri_2d_improved_unet/train.py --data-root <data-root> --output-dir <new-output-dir> ...
python recognition/hipmri_2d_improved_unet/predict.py --checkpoint <checkpoint.pt> --data-root <data-root> --split validation --filename <validation-MRI-basename> --output-dir <new-output-dir> ...
```

## 现有实验的执行限制

普通入口接受 --device cpu，但现有正式训练计划明确要求 CUDA，冻结测试与获准恢复也校验其原运行协议。CPU 接口不代表获准把正式计划改成 CPU，或声称获得同一次实验的结果。最终正式选模是验证 ID5 Dice，并列时依次选更低 CE、更早轮次；train.py 的通用默认参数仍为历史 CE 规则，不能只用默认值声称复现最终方案。

测试展示分支仅通过11项构造CPU检查；最终模型展示实际涉及5对验证输入和两份冻结权重，没有产生额外真实测试分数。构造检查与真实验证展示分开记录。

原 outputs/frozen_test_once_20261026.guard.json 为 incomplete：baseline 已完整通过，Improved 在原尝试中中断。outputs/frozen_test_recovery_once_20261026.guard.json 为 passed：用户仅授权 Improved 完整补跑一次，原记录全部保留。不得删除、重置或更名 guard 来重新评估。guard 位于忽略目录，普通 clone 没有它们；缺少 guard 也不产生新的测试授权。

历史计划与部分证据的 SHA256 针对当时的文件字节，换行转换可能改变摘要。不得通过修改旧摘要绕过冻结校验；核验工具分别检查冻结源文件与保留记录。已有实验结果和本机版本记录是可审查证据，不能替代尚未执行的新环境安装验证。

当前展示入口及原实验的源码版本分开：<a href="evidence/submission_source_evolution_20261026.json">演进记录</a>绑定历史predict.py与当前predict.py／prediction_panels.py；原训练、预处理、loss、metrics和测试执行源码仍按原记录核对。<a href="evidence/final_validation_display_20261026.json">真实验证展示</a>只涉及5对指定验证输入和两份冻结权重，未产生新的测试结果。具体用法见README。
