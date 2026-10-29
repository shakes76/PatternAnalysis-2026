"""Publish the verified, completed comparison to the local report and status notes.

Run with the document-generation Python after compare_confirmation.py and
finalize_decision.py. This does not train, infer, upload, or submit anything.
Visual review of the regenerated PDF remains a separate required step.
"""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import shlex
import subprocess
import sys

S = Path(__file__).resolve().parent
R = S.parents[1]
C = R / 'recognition/YOLO_s4905914'
read = lambda p: json.loads(p.read_text())
r = read(S / 'results.json')
d = read(S / 'decision.json')
c = read(S / 'confirmation_results.json')
assert r['status'] == c['status'] == d['status'] == 'completed'
assert r['final_decision'] == d and r['confirmation'] == c
assert not d['new_reassessment_inference'] and not d['further_training_planned']
assert d['confirmation_passed'] == c['passes_predeclared_confirmation_rule']
assert hashlib.sha256((S / 'confirmation_plan.json').read_bytes()).hexdigest() == c['confirmation_plan_sha256']

audit = R / 'outputs/known_optimization_final_20261022'
for rel in ['output/pdf/COMP3710_Project3_Report.pdf',
            'recognition/YOLO_s4905914/README.md',
            'recognition/YOLO_s4905914/known_optimization_results_20261021.json',
            'recognition/YOLO_s4905914/known_optimization_decision_20261021.json',
            'READINESS_STATUS_zh.txt', 'START_HERE_zh.txt',
            'SUBMISSION_DECISION_zh.txt', 'TEACHER_QUESTIONS.txt',
            'PRE_SUBMISSION_CHECK_zh.txt']:
    target = audit / 'before' / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(R / rel, target)

shutil.copy2(S / 'results.json', C / 'known_optimization_results_20261021.json')
shutil.copy2(S / 'decision.json', C / 'known_optimization_decision_20261021.json')
command=['python','recognition/YOLO_s4905914/predict.py',
         '--config',d['config_path'],'--checkpoint',d['checkpoint_path'],
         '--lock',d['selection_lock_path'],'--image','/path/to/image.jpg',
         '--output','outputs/current_model_prediction','--device','cpu']
(S / 'SELECTED_MODEL_COMMAND.txt').write_text(
    'Run from the restart_20261017 project root in the pinned ML environment.\n'
    'Replace /path/to/image.jpg with the intended dermoscopic photo and choose\n'
    'a new output directory. This command has not been run on reassessment images.\n\n'
    + shlex.join(command) + '\n')
now = datetime.now(ZoneInfo('Australia/Brisbane')).isoformat()
chosen = '轻度几何增强 n（seed=3710）' if d['candidate'] == 'n_geometry' else '原 n（seed=3710）'
outcome = ('两次 seed 比较均达到事先规定的改善标准，推荐初次几何增强模型。'
           if d['confirmation_passed'] else
           '几何增强的初次改善未通过第二个 seed 的确认标准，保留原 n 为主实验参考。')
rows = '\n'.join(' | '.join(map(str, row)) for row in c['report_rows'])
summary = f'''最新状态（2026-10-22，布里斯班）

三组已知病灶子集优化和两次 seed=3711 确认均已完成，结果按原规则比较。
当前选择：{chosen}，confidence={d['confidence']:.2f}。
{outcome}
没有从两个 seed 中挑较好的一个；当前选择仍指 seed=3710 的权重。

运行 / seed | confidence | 恶性 TP/41 | 良性误警报/124 | 恶性全 GT 正确类别 IoU
{rows}

以上都来自同一 validation。每次拟合按相同网格和最多 12 张良性误警报
的规则选择阈值。IoU 包括全部 41 个恶性 GT，无正确类别预测时记 0。
第二个 seed 是对训练随机性的一次检查，不是独立验证或新盲测。
两次 seed 不能证明普遍稳定性；这里的“通过确认”仅指达到本次预定规则。

当前仅使用 1,098 张已知 lesion_id 图片：train 768、validation 165、
reassessment 165。本次优化只使用 train/validation，没有新补充评估推理或调参。
老师建议使用这个子集的情况来自学生转述；708 张缺 patient_id，仍不能
证明完整患者独立。扩充数据模型只作为历史探索，不参与当前选择。

模型选择记录：recognition/YOLO_s4905914/known_optimization_decision_20261021.json
对应 validation 输出：{d['evaluation_path']}
checkpoint SHA-256：{d['checkpoint_sha256']}
预测命令模板：studies/known_lesion_optimization_20261021/SELECTED_MODEL_COMMAND.txt
报告：recognition/YOLO_s4905914/README.md
PDF：output/pdf/COMP3710_Project3_Report.pdf
结果和来源：studies/known_lesion_optimization_20261021/results.json
报告核验：outputs/known_optimization_report_20261021/verification.json
报告的逐页视觉检查是否完成，以该核验文件的最新状态为准。

本轮固定实验到此停止，不追加模型或阈值搜索。原训练配置、阈值锁和
原始预测记录均保留。没有改写历史结果，没有 commit、push、新 PR、
对外消息或正式提交；也没有创建定时自动任务。

剩余课程事项：IoU 分母/性能门槛及第三方组件要求；个人 feasibility
check-off、准确截止时间；学生审阅与课程仓库整合。当前英文问题稿
为 TEACHER_CONFIRMATION_EN.txt，尚未发送。子集选择不再重复询问。
旧转移 ZIP 和 18 日提交前检查属于历史版本，不能当作当前提交包。
'''
(S / 'RESULTS_zh.txt').write_text(summary)
marker = '以下保留此前记录；与上面的最新状态冲突时，以最新状态为准。\n\n'
for name in ['READINESS_STATUS_zh.txt', 'START_HERE_zh.txt', 'SUBMISSION_DECISION_zh.txt']:
    p = R / name
    historical = p.read_text().split(marker, 1)[-1]
    p.write_text(summary + '\n' + marker + historical)
p = R / 'TEACHER_QUESTIONS.txt'
marker_en = 'Historical drafts and earlier status follow:\n\n'
historical = p.read_text().split(marker_en, 1)[-1]
p.write_text('Current status - 22 October 2026\n\n'
    'The student reports that the teacher advised using the known-lesion-ID subset.\n'
    'The three initial validation ablations and two second-seed confirmation runs\n'
    'are complete. No new reassessment inference or tuning was performed.\n'
    'The old cohort question below is historical. Remaining questions concern\n'
    'IoU aggregation and performance requirements, imported model components,\n'
    'personal check-off and the deadline. Use TEACHER_CONFIRMATION_EN.txt as\n'
    'the current consolidated draft; no message has been sent.\n\n' + marker_en + historical)
p = R / 'PRE_SUBMISSION_CHECK_zh.txt'
marker_old = '提交前检查（2026-10-18，未提交）'
historical = marker_old + p.read_text().split(marker_old, 1)[1]
p.write_text('最新报告检查（2026-10-22，未提交）\n\n'
    '三个已知病灶子集对照和第二 seed 确认已完成，并按预定规则确定结果。\n'
    '本次完整记录见 studies/known_lesion_optimization_20261021/RESULTS_zh.txt。\n'
    '当前 README/PDF 核验见 outputs/known_optimization_report_20261021/verification.json。\n'
    '此次没有重跑全套 ML 单元测试、补充评估或更新旧转移 ZIP；没有提交。\n'
    '下方 18 日检查仅为当时版本的历史记录。\n\n' + historical)
status = read(S / 'RUN_STATUS.json')
status.update(status='completed', completed_at=now, remote_status_verified=True,
    latest_connection_result='connected; both confirmation outputs downloaded and verified',
    latest_slurm_state='COMPLETED', latest_slurm_reason=None,
    confirmation_runmap=read(S / 'confirmation_runmap.json'),
    baseline_confirmation_downloaded=True, candidate_confirmation_downloaded=True,
    confirmation_passed=d['confirmation_passed'], chosen_candidate=d['candidate'],
    chosen_confidence=d['confidence'], further_training_planned=False)
status.pop('estimated_start_brisbane', None)
(S / 'RUN_STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
(S / 'ACTIVE_WORK.txt').write_text('Completed known-lesion optimization and seed confirmation.\n\n'
    'Read RESULTS_zh.txt, decision.json and confirmation_results.json. No further\n'
    'training or reassessment inference is planned. Do not release or resubmit681542.\n'
    'README/PDF rebuilt from completed evidence; check report verification.json\n'
    'for final visual review status. Original configs/results/locks are preserved.\n'
    'No submission, push, commit, external messages or automation created.\n')
for args in [[C / 'build_final_report.py'],
             [C / 'export_readme_pdf.py', '--output', R / 'output/pdf/COMP3710_Project3_Report.pdf'],
             [R / 'outputs/known_optimization_report_20261021/verify_report.py']]:
    subprocess.run([sys.executable, *map(str, args)], check=True)
print('Local report and notes updated. Render and visually inspect the final PDF next.')
