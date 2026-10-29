"""Write concise current status and a reproducible selected-model command."""
import hashlib,json
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[1]
read=lambda p:json.loads(p.read_text())
d=read(S/'decision.json');x=read(S/'results.json');assert d['status']=='completed'
names={'reference_geometry':'上一轮 n + 轻度几何增强','n_resolution896':'n + 轻度几何增强 + 896 输入',
       'n_positive2':'n + 轻度几何增强 + 恶性正项权重 2','s_geometry':'s + 轻度几何增强'}
lines=['本轮 validation 优化已完成（2026-10-22）',
    '范围：1,098 张有 lesion_id 的数据；训练 768，validation 165；补充评估集未参与。',
    '主比较：恶性检测 IoU>=0.5；良性图片误警报最多 12/124；统一的 14 个阈值。','',
    '方案 | confidence | 恶性 TP/41 | 良性误警报/124 | 恶性全部 GT 正确类别 IoU']
for n,r in x['reports'].items():
    a=r['selected'];m=a['classes']['Malignant']
    lines.append(f"{names[n]} | {a['threshold']:.2f} | {m['tp']} | {a['image_alerts']['benign_fp']} | {m['iou']['all_gt_class_correct']['mean']:.4f}")
p=read(S/'policy_results_seed3710.json')['selected']
lines+=['',f"额外单框规则：相对分数阈值 {p['ratio_threshold']:.2f}（不是普通 confidence），恶性 TP {p['classes']['Malignant']['tp']}/41，良性误警报 {p['image_alerts']['benign_fp']}/124；未改善，弃用。",'',
    f"当前选择：{names[d['selected']]}，confidence={d['confidence']:.2f}。",
    f"第二个 seed 确认：{'通过' if d['confirmation_passed'] else '未通过' if d['confirmation_triggered'] else '无新候选达到触发条件，未新增确认训练'}。",
    '以上均是反复使用的 validation 结果，不是新的盲测；本轮不宣称达到课程 test IoU>=0.8。',
    'TP-only、忽略类别和包含漏检的 IoU 不能互换。扩大图片警报预算不是临床安全证明。',
    '当前模型仍有恶性漏检；本轮没有继续搜索参数直到出现有利数字。',
    '旧报告及旧实验记录保留；本轮完整结果在独立 PDF 补充报告中。',
    '代码、参数和结果尚未提交，也未发布到 GitHub。']
(S/'RESULTS_zh.txt').write_text('\n'.join(lines)+'\n')
lock=d['evaluation_path']+'/selection_lock.json'
command=(f"python recognition/YOLO_s4905914/predict.py --config {d['config_path']} "
    f"--checkpoint {d['checkpoint_path']} --lock {lock} --image /path/to/image.jpg "
    '--output outputs/followup_selected_prediction --device cpu')
(S/'SELECTED_MODEL_COMMAND.txt').write_text('Run from restart_20261017 in the pinned ML environment.\n'
    'Replace the image path and choose a fresh output directory. This command has not been executed on reassessment images.\n\n'+command+'\n')
print('\n'.join(lines))
