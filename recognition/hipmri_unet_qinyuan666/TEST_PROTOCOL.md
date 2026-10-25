# 固定测试集评估方案

本方案在首次测试集模型评分前固定。两模型各完成 20 epochs，权重均按最低验证
CE + foreground soft Dice loss 选定；不根据测试结果更换轮次、调参或追加训练。
本阶段只完成 Dice/IoU 测试评分，整病例测速与边界/失败分析另行记录。

## 固定输入

- 清单：`private_data/manifests/course_v1.csv`。
- 清单 SHA256：`5ad6cc420967afc51ec60d0772cfa8c3ff12332db3efe25437693777494c04d3`。
- 沿用课程 test 分组：540 张切片，3 个 Case，9 个 Case/week；不重新分组。
- 六类固定为 0–5，前列腺为 5；逐切片非零像素归一化，256×256 pad，不增强。
- 预测采用六类 logits 的 argmax，不进行阈值搜索或后处理。
- CUDA float32、batch size 4、CPU threads 4；模型 eval 模式、无梯度、无优化器。

| 模型 | 权重路径 | 验证选中轮次 | 权重 SHA256 |
|---|---|---:|---|
| U-Net | `runs/real_unet_20ep_s3710/best.pt` | 5 | `d4dc369bcfdd279e5337c12c4e2d454e88a9a957771bb17b14e84b8813ee7fd7` |
| Improved U-Net | `runs/real_improved_unet_20ep_s3710/best.pt` | 11 | `7ed4f8516e1c058a5dd2b3ca1f0c6dc9b1e90ff78aa1b8f1bde20e4d15bcecae` |

两权重及训练源码已在各自正式训练记录中追溯。评估前后核对权重哈希，
逐病例混淆矩阵留在本机，公开汇总使用 Case 别名；测试别名与验证别名不是同一病例。
不复制原始 MRI 或病例编号到 GitHub。

## 指标与解释规则

全部切片累加混淆矩阵，报告各类 Dice、IoU 和前景宏平均 Dice。
另在每个 Case 内合并全部周次与切片，报告各类病例等权平均及前列腺病例分布。
真实和预测都为空时记 null，不纳入均值并报告数量；仅一方为空时为 0。
补边计入指标，与训练/验证规则保持一致；没有重建 3D 体积。

原任务要求测试集前列腺 Dice ≥ 0.75，未明确指定聚合方式。
因此同时给出像素汇总与病例等权平均是否达到该值，不能事后挑选较高的方式。
只有 3 个测试 Case、单个 seed，不声称统计显著性或临床有效性。
不能仅凭 Dice 宣称边界改善；物理单位未知，后续边界测量须明确单位。

## 执行和复现

在项目目录、项目 Python 环境中依次执行。首次结果保存在以下新目录；
复现时替换为其他新目录，工具拒绝覆盖已有评估。

```bash
python evaluate_cases.py --checkpoint runs/real_unet_20ep_s3710/best.pt --manifest private_data/manifests/course_v1.csv --output runs/real_unet_20ep_s3710/test_evaluation --device cuda --batch-size 4 --split test
python evaluate_cases.py --checkpoint runs/real_improved_unet_20ep_s3710/best.pt --manifest private_data/manifests/course_v1.csv --output runs/real_improved_unet_20ep_s3710/test_evaluation --device cuda --batch-size 4 --split test
```

`evaluate_cases.py` 默认仍只评估 val，只有显式 `--split test` 才评估 test。
它检查清单哈希、模型/标签配置，并记录权重和评估代码哈希。
若执行发生技术错误，记录错误并修复评估工具；不得据测试分数修改模型。
