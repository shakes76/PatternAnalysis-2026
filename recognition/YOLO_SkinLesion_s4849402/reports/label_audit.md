# ISIC 2017 标签检查

检查日期：2026-10-28。通过 `python audit_labels.py` 获取官方训练集与验证集诊断 CSV；总下载量 47,380 字节。没有下载图片、mask 或测试集标签。

| 类别 | class_id（项目约定） | 训练集 | 验证集 |
|---|---:|---:|---:|
| melanoma（黑色素瘤） | 0 | 374 | 30 |
| seborrheic keratosis（脂溢性角化病） | 1 | 254 | 42 |
| nevus（痣） | 2 | 1372 | 78 |
| 合计 | | 2000 | 150 |

原始字段为 `image_id,melanoma,seborrheic_keratosis`。项目按该三类别数据定义将 melanoma=1 映射为 0、seborrheic_keratosis=1 映射为 1、两者均为 0 映射为 nevus（2）。脚本拒绝两个标志同时为 1、非二值标签和重复 image_id。该映射不应泛化到其他年份或其他标签表。

## 已检查与未检查

- 每个 CSV 内图片 ID 唯一，训练与验证图片 ID 无重叠。
- 类别不均衡，训练集中 nevus 占 68.6%；后续不能只看总体准确率，需报告逐类指标。
- 两个 CSV 均无患者编号或受试者编号。图片 ID 不重叠不等于患者独立，患者隔离尚未验证。
- 下一步核对官方图像包是否含可用元数据或能否取得可靠的 image-to-patient 映射；如果不能，需要向 tutor 说明并确认合适方案。
- 现有 `dataset.py` 会拒绝缺失 patient_id 的导出。目前不能直接把诊断 CSV 当作已经合格的训练清单，更不能填入伪造的患者编号。
- 图片、mask、诊断表之间的对应关系，以及框的位置、图像重复/近重复尚未检查。
- 尚未运行真实模型实验；本报告不包含模型性能结果。

## 复现和来源

在项目目录运行 `python audit_labels.py`。下载文件位于被 Git 忽略的 `data/metadata/`，可提交的汇总与 SHA-256 位于 `reports/label_audit.json`。

- https://challenge.isic-archive.com/data/#2017
- https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Part3_GroundTruth.csv
- https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Validation_Part3_GroundTruth.csv
