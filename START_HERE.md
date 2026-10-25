# COMP3710 项目交付入口

选题：HipMRI 2D 分割，标准 2D U-Net 基线与 2D Improved U-Net 主模型（Normal），学号 s4906920。

代码、两模型各 20 epoch 正式训练、固定测试评估、资源测量、边界分析、五个失败案例和最终报告均已完成并上传。课程 PR 已创建，等待教学团队评审。

## 最终提交材料

- [课程 PR #23](https://github.com/shakes76/PatternAnalysis-2026/pull/23)：从 `qinyuan666/PatternAnalysis-2026:topic-recognition` 提交到 `shakes76/PatternAnalysis-2026:topic-recognition`。
- [最终技术报告 README](recognition/hipmri_unet_qinyuan666/README.md)：完整方法、实验、局限、复现命令、引用及 AI 使用披露。
- [最终报告 PDF](recognition/hipmri_unet_qinyuan666/final_report.pdf)：14 页，包含一页历史可行性评估、训练曲线及五张失败案例图；这是需要交到 Turnitin 的文件。
- [最终核验记录](recognition/hipmri_unet_qinyuan666/results/final_verification.json)：73 项自动化测试通过，合成训练与预测命令实际运行通过，固定权重评估结果复核一致，PDF 排版与文件链接已检查。

## 结果如何解释

| 测试指标 | U-Net | Improved U-Net |
| --- | --- | --- |
| 前列腺像素汇总 Dice | 0.8460 | 0.8495 |
| 前列腺等权病例平均 Dice | 0.8416 | 0.8406 |
| 前列腺阳性切片平均边界 F1（2 像素容差） | 0.8675 | 0.8659 |

两模型的前列腺 Dice 均超过 0.75；Improved U-Net 没有达到原先提出的边界 F1 提升 0.02 的目标，也没有在所有指标上优于基线。报告保留这一结果，并根据计算成本及失败案例建议以 U-Net 作为当前研究基线。一次随机种子、三个测试病例不足以证明统计优势或临床适用性。

整病例测速已在关闭游戏后复测，首次受其他 GPU 任务影响的结果保留为历史记录。无需为本次交付重新训练或再次连接 Rangpur。

## 还需要本人完成的提交操作

1. 阅读最终报告，确认理解实现、结果及 AI 使用披露；在课程指定入口上传 `final_report.pdf` 到 Turnitin，并检查提交回执。创建 GitHub PR 不等于完成 Turnitin 提交。
2. 教学团队发布 PR 评审后，在课程指定窗口内回复并处理反馈。PR 当前处于等待评审阶段，不代表已经合并或评分。

截止时间、提交入口和评审回复窗口以 Blackboard / course profile 的最新通知为准。

## 本地数据与复现

项目目录为 `recognition/hipmri_unet_qinyuan666/`。原始数据、私有病例清单和模型权重均留在本地，没有上传到公开仓库；五张经本人明确授权的派生失败案例图及含图报告已上传。

- [数据访问与目录说明](recognition/hipmri_unet_qinyuan666/DATA_ACCESS.md)
- [正式 U-Net 训练记录](recognition/hipmri_unet_qinyuan666/FORMAL_UNET.md)
- [正式 Improved U-Net 训练记录](recognition/hipmri_unet_qinyuan666/FORMAL_IMPROVED_UNET.md)
- [固定测试集评估](recognition/hipmri_unet_qinyuan666/TEST_RESULTS.md)
- [整病例测速及复测记录](recognition/hipmri_unet_qinyuan666/CASE_BENCHMARK.md)

中期材料保留在 [可行性评估 PDF](recognition/hipmri_unet_qinyuan666/feasibility_review.pdf) 和 [中文讨论笔记](recognition/hipmri_unet_qinyuan666/MIDPOINT_NOTES_ZH.md)，反映当时进展；最终结论以最终技术报告为准。

开发仓库为 [qinyuan666/PatternAnalysis-2026](https://github.com/qinyuan666/PatternAnalysis-2026/tree/topic-recognition)，是课程仓库的真实 Fork。原 `comp3710-Final` 空仓库保留，后续开发与提交均在课程 Fork 的 `topic-recognition` 分支进行。各项工作的提交记录可通过 `git log` 查看。
