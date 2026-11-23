"""Fixed revised A1 evaluation on the previously exposed patient-independent test.

No model training, calibration fitting, threshold search, or policy selection.
The existing sources/checkpoints are read-only. New commands are dispatched from
predict.py. Recovery reuses completed batches only under identical frozen inputs.

Development assistance: ChatGPT; review, verify, understand and disclose usage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image, ImageDraw
import torch

import selected_a1 as sa
import negative_policy as policy
from dataset import preprocessing_spec
from training_utils import (choose_device, seed_everything, synchronize,
    memory_info, reset_memory_peak, hardware_info)
from train_convnext_v2 import run_lock, environment, atomic_json
from confidence_analysis import scaled_probabilities, probability_scores, confidence_diagnostics
from final_evaluation import (VerifiedTestDataset, validate_progress, aggregate_test,
                             paired_comparison, select_failure_cases)

NEW_FILES = ('predict.py', 'revised_evaluation.py', 'REVISED_EVALUATION_METHOD.md',
             'tests_revised/test_revised_evaluation.py')
PLAN_FILE = 'A1_REVISED_EVALUATION_PLAN.json'
FOLDER = 'a1-revised-evaluation'
POINTER = 'latest_a1_revised_evaluation.txt'
SUMMARY = 'revised_test_summary.json'
BUNDLE = 'COMP3710_A1_Revised_Evaluation_Results.zip'
PLOTS = ('a1_revised_test_roc.png', 'a1_revised_test_reliability.png')
PUBLIC = (PLAN_FILE, SUMMARY, 'README_revised_test_results.md',
          'failure_review_status.json', *PLOTS)
MARKER = '<!-- COMP3710-A1-REVISED-EVALUATION -->'
CAUTION = (
    'Research prototype, not a clinical diagnostic system. This is a revised evaluation '
    'on the SAME patient-independent test cohort previously evaluated for Version 1, '
    'not a new untouched independent test. Later development used repeatedly consulted '
    'training/validation evidence; prior test exposure remains a limitation. Model, '
    'temperature, diagnosis cutoff and all three policies were fixed before this revised '
    'inference. Do not select another model/policy or tune using these results. '
    'Referrals are not correct diagnoses; no clinician accuracy is assumed.')

require = policy.require
read_json = policy.read_json
sha256 = policy.sha256


def now():
    return datetime.now(timezone.utc).isoformat()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def private_output(private):
    private = Path(private).resolve()
    require(not any((p / '.git').exists() for p in (private, *private.parents)),
            'Keep evaluation records outside Git.')
    folder = private / FOLDER
    require(not folder.is_symlink(), 'Do not redirect the revised output using a symlink.')
    folder.mkdir(exist_ok=True)
    return folder


def load_development(private):
    """Validate existing development evidence; no MRI decoding or test inference."""
    ctx, confidence, rules, _, input_hashes = policy.load_inputs(private)
    require(ctx['project'] == Path(__file__).resolve().parent,
            'Run the installed project entry point, not the extracted tools copy.')
    expected_policy_plan = policy.make_plan(ctx, confidence, rules, input_hashes)
    require(read_json(ctx['project'] / policy.PLAN_FILE) == expected_policy_plan,
            'The declared negative-policy plan differs.')
    negative_dir = sa.pointer(ctx['private'], 'latest_negative_policy.txt')
    policy.validate_output(negative_dir, expected_policy_plan)
    exported = read_json(negative_dir / 'negative_policy_rules.json')
    require(exported == policy.export_rules(rules, input_hashes),
            'Saved secondary policies differ from the declared comparison.')
    old_dir = sa.pointer(ctx['private'], 'latest_final_evaluation.txt')
    old_summary = old_dir / 'final_test_summary.json'
    require(read_json(old_summary).get('quantitative_evaluation_completed') is True,
            'Expected the preserved Version 1 evaluation; do not create or overwrite it.')
    files = {
        'a1_selection': ctx['project'] / sa.PLAN_FILE,
        'a1_confidence': sa.pointer(ctx['private'], sa.POINTER) / sa.SUMMARY,
        'a1_rules': sa.pointer(ctx['private'], sa.POINTER) / sa.RULES,
        'negative_plan': ctx['project'] / policy.PLAN_FILE,
        'negative_summary': negative_dir / 'negative_policy_summary.json',
        'negative_rules': negative_dir / 'negative_policy_rules.json',
        'version1_summary': old_summary,
    }
    audit = sa.pointer(ctx['private'], 'latest_adni_audit.txt') / 'private_inventory.json'
    inventory = read_json(audit)
    test_hashes = audited_hashes(ctx['bundle'].records['test'], inventory)
    sources = dict(ctx['sources'])
    sources.update(policy.source_hashes(ctx['project']))
    for name in (*NEW_FILES, 'final_evaluation.py'):
        sources[name] = sha256(ctx['project'] / name)
    ctx.update(confidence=confidence, rules=rules, negative_rules=exported,
               bound_evidence={name: sha256(path) for name, path in files.items()},
               audit_inventory_sha256=sha256(audit), test_hashes=test_hashes,
               revised_sources=sources)
    return ctx


def audited_hashes(records, inventory):
    require(bool(records) and all(r.split == 'test' for r in records), 'Expected test records.')
    index = {}
    for row in inventory.get('images', []):
        key = row.get('relative_path')
        require(isinstance(key, str) and key not in index, 'Duplicate or malformed audited path.')
        index[key] = row
    result = {}
    for r in records:
        item = index.get(r.image_path, {})
        h = item.get('file_sha256')
        require(item.get('status') == 'decoded' and isinstance(h, str) and len(h) == 64
                and all(c in '0123456789abcdef' for c in h), 'Missing original audited test-image hash.')
        require(r.image_path not in result, 'Repeated test-image path.')
        result[r.image_path] = h
    return result


def counts(records):
    subjects = {}
    scans = set()
    for r in records:
        require(r.subject_id not in subjects or subjects[r.subject_id] == r.label,
                'Patient labels disagree.')
        subjects[r.subject_id] = r.label
        scans.add(r.scan_id)
    return {'patients': len(subjects), 'scans': len(scans), 'images': len(records),
            'patients_NC': sum(v == 0 for v in subjects.values()),
            'patients_AD': sum(v == 1 for v in subjects.values())}


def make_plan(ctx, device_name):
    device = choose_device(device_name)
    env = environment(device)
    for alias, model in ctx['confidence']['models'].items():
        require(model['checkpoint_replay']['passed'] is True
                and model['checkpoint_replay']['binding']['environment'] == env,
                'Use the same device/software as the verified selected-checkpoint replay.')
    return {
        'format_version': 1, 'purpose': 'FIXED_REVISED_TEST_EVALUATION_PREVIOUS_COHORT_EXPOSURE_DISCLOSED',
        'models': {a: {'name': c.name, 'selected_epoch': c.report['selected_epoch'],
                      'configuration': c.report['configuration'], 'parameters': c.report['parameters'],
                      'input_hashes': c.hashes, 'temperature': ctx['rules']['models'][a]['temperature'],
                      'reference_confidence_threshold': ctx['rules']['models'][a]['confidence_threshold']}
                   for a, c in ctx['candidates'].items()},
        'manifest_sha256': ctx['bundle'].manifest_hashes,
        'test_counts': counts(ctx['bundle'].records['test']),
        'preprocessing': preprocessing_spec(0.0), 'model_labels': {'NC': 0, 'AD': 1},
        'configuration': {'device': device_name, 'batch_size': 8, 'workers': 0,
            'seed_for_inference': 42, 'precision': 'float32; mixed precision and TF32 disabled',
            'primary_unit': 'patient', 'secondary_units': ['scan', 'slice'],
            'diagnostic_threshold': 0.5, 'aggregation': 'mean of scan means of slice P(AD)',
            'temperature_applied_to': 'patient probability log odds only',
            'accuracy_target': 0.8, 'bootstrap_repeats': 5000, 'bootstrap_seed': 2026,
            'policies_reported_without_selection': list(policy.POLICIES),
            'strict_CN_maximum_calibrated_p_AD': policy.CN_AD_PROBABILITY_CAP,
            'failure_cases_maximum': 5,
            'failure_selection': 'Unique patients: confident A1 FN; confident A1 FP; A1 error corrected by CNN; low-confidence A1 error; remaining confident errors; CNN errors if needed.',
            'case_scan_selection': 'scan probability nearest patient probability; scan ID breaks ties',
            'case_slice_positions': [0.25, 0.5, 0.75]},
        'environment': env, 'source_code_sha256': ctx['revised_sources'],
        'development_evidence_sha256': ctx['bound_evidence'],
        'audit_inventory_sha256': ctx['audit_inventory_sha256'],
        'test_image_hash_map_sha256': object_hash(ctx['test_hashes']),
        'prior_test_exposure_disclosed': True, 'fresh_independent_test': False,
        'no_training_calibration_fitting_or_policy_search': True,
        'no_automatic_clinical_clearance_recommended': True,
        'manual_failure_analysis_required': True, 'caution': CAUTION}


def set_pointer(private, folder):
    p = Path(private) / POINTER
    temp = p.with_suffix('.tmp')
    temp.write_text(str(folder.resolve()) + '\n', encoding='utf-8')
    temp.replace(p)


def freeze(private, device_name='cuda'):
    ctx = load_development(Path(private))
    folder = private_output(ctx['private'])
    with run_lock(folder):
        plan = make_plan(ctx, device_name)
        pub, saved = ctx['project'] / PLAN_FILE, folder / PLAN_FILE
        require(not (folder / 'REVISED_TEST_ACCESS_STARTED.json').exists() or saved.exists(),
                'Test access exists without its plan. Preserve all evidence.')
        for path in (pub, saved):
            if path.exists():
                require(read_json(path) == plan, 'An existing revised plan differs. Do not replace it.')
        existed = saved.exists() and pub.exists()
        if not saved.exists(): atomic_json(saved, plan)
        if not pub.exists(): shutil.copy2(saved, pub)
        set_pointer(ctx['private'], folder)
    print('A1_REVISED_PLAN_ALREADY_FROZEN' if existed else 'A1_REVISED_PLAN_FROZEN')
    print('REVIEW_AND_COMMIT_PLAN_AND_CODE_BEFORE_REVISED_EVALUATION')
    return folder


def require_committed(project, plan):
    sa.project_check(project)
    prefix = sa.git(project, 'rev-parse', '--show-prefix').decode().strip()
    for name in (*NEW_FILES, PLAN_FILE):
        sa.git(project, 'ls-files', '--error-unmatch', '--', name)
    committed = json.loads(sa.git(project, 'show', f'HEAD:{prefix}{PLAN_FILE}').decode('utf-8-sig'))
    require(committed == plan, 'Commit the unchanged revised plan before test inference.')
    sa.git(project, 'diff', '--exit-code', 'HEAD', '--', *NEW_FILES, PLAN_FILE)
    return sa.git(project, 'rev-parse', 'HEAD').decode().strip()


def checked(private, committed=True):
    ctx = load_development(Path(private))
    folder = private_output(ctx['private'])
    require((folder / PLAN_FILE).is_file(), 'Run revised-freeze first.')
    plan = read_json(folder / PLAN_FILE)
    require(plan == read_json(ctx['project'] / PLAN_FILE), 'Private and repository plans differ.')
    require(plan == make_plan(ctx, plan['configuration']['device']),
            'Frozen models, sources, evidence, environment, or manifests have changed.')
    commit = require_committed(ctx['project'], plan) if committed else None
    return ctx, folder, plan, commit


def infer_cached(model, data, device, path, identity, batch_size):
    """Commit predictions after each batch; verify cached-image bytes on recovery."""
    total = len(data)
    if path.exists():
        cache = read_json(path)
        p = validate_progress(cache, identity, total, batch_size)
        require(cache.get('completed_prediction_sha256') == object_hash(p), 'Cached predictions changed.')
        for i in range(len(p)): data.verified_bytes(i)
        for key in ('batch_seconds', 'forward_seconds'):
            require(isinstance(cache.get(key), (int, float)) and math.isfinite(cache[key])
                    and cache[key] >= 0, 'Invalid saved timing.')
    else:
        p = []
        cache = {'identity': identity, 'probabilities': [], 'batch_seconds': 0.,
                 'forward_seconds': 0., 'peak_allocated_gib': None, 'peak_reserved_gib': None}
    reset_memory_peak(device)
    model.eval()
    print(f"{identity['alias']}: reusing {len(p)}/{total} completed slice predictions", flush=True)
    with torch.inference_mode():
        for start in range(len(p), total, batch_size):
            synchronize(device); t0 = time.perf_counter()
            images = torch.stack([data[i]['image'] for i in range(start, min(start+batch_size, total))]).to(device)
            synchronize(device); t1 = time.perf_counter()
            logits = model(images)
            synchronize(device); t2 = time.perf_counter()
            require(logits.shape == (len(images), 2) and bool(torch.isfinite(logits).all()), 'Invalid inference logits.')
            values = torch.softmax(logits, 1)[:, 1].cpu().tolist()
            require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), 'Invalid inference probability.')
            synchronize(device); t3 = time.perf_counter()
            p.extend(values)
            cache.update(probabilities=p, completed_prediction_sha256=object_hash(p))
            cache['batch_seconds'] += t3-t0
            cache['forward_seconds'] += t2-t1
            for key, val in memory_info(device).items():
                if key in ('peak_allocated_gib', 'peak_reserved_gib') and val is not None:
                    cache[key] = max(cache[key] or 0., val)
            atomic_json(path, cache)
            if len(p) % 800 == 0 or len(p) == total:
                print(f"{identity['alias']}: saved {len(p)}/{total} test slices", flush=True)
    resources = {k: v for k, v in cache.items() if k not in ('identity', 'probabilities', 'completed_prediction_sha256')}
    resources['timing_scope'] = 'Sum of completed batches: includes hash/decode/preprocess/transfer/forward/output conversion; excludes checkpoint load, cache writes, resume verification and interrupted work.'
    resources['memory_scope'] = 'PyTorch process allocation peaks, not total Windows VRAM; inference, not training.'
    return p, resources


def score(records, probabilities, frozen_model):
    metrics, rows = aggregate_test(records, probabilities)
    y = [r['label'] for r in rows['patients']]
    p = [r['p_ad'] for r in rows['patients']]
    q = scaled_probabilities(p, frozen_model['temperature']).tolist()
    require(all((a >= .5) == (b >= .5) for a, b in zip(p, q)), 'Calibration changed class predictions.')
    policies, choices = policy.compare_actions(y, q,
        [frozen_model['reference_confidence_threshold']] * len(q))
    for i, row in enumerate(rows['patients']):
        row.update(prediction=int(p[i] >= .5), calibrated_p_ad=q[i],
                   confidence=max(q[i], 1-q[i]), actions={name: choices[name][i] for name in policy.POLICIES},
                   action=choices['reference'][i])
    cm = metrics['patient']['confusion_matrix_rows_true_columns_predicted_NC_AD']
    correct = cm[0][0] + cm[1][1]
    report = {'name': frozen_model['name'], 'selected_epoch': frozen_model['selected_epoch'],
              'checkpoint_sha256': frozen_model['input_hashes']['checkpoint_sha256'],
              'metrics': metrics, 'patient_accuracy_wilson95': policy.interval(correct, len(y)),
              'AD_recall_wilson95': policy.interval(cm[1][1], sum(cm[1])),
              'NC_recall_wilson95': policy.interval(cm[0][0], sum(cm[0])),
              'raw_probability_scores': probability_scores(y, p),
              'fixed_temperature_probability_scores': probability_scores(y, q),
              'raw_confidence_diagnostics': confidence_diagnostics(y, p),
              'fixed_temperature_confidence_diagnostics': confidence_diagnostics(y, q),
              'fixed_temperature': frozen_model['temperature'], 'diagnostic_threshold': .5,
              'policies': policies, 'no_test_fitting_or_policy_selection': True,
              'observed_accuracy_at_least_0_80_by_unit': {k: v['accuracy'] >= .8 for k, v in metrics.items()},
              'clinical_safety_established': False}
    return report, rows


def build_cases(output, data, predictions):
    """Generate PRIVATE descriptive input panels; never fabricate manual findings."""
    cases = select_failure_cases({'cnn': predictions['cnn'], 'convnext': predictions['a1']}, maximum=5)
    folder = output / 'PRIVATE_failure_review'
    folder.mkdir(exist_ok=True)
    text = ['# PRIVATE revised-evaluation failure review', '', CAUTION, '',
            'Complete 3–5 cases where available. These are input panels, NOT saliency maps or causal explanations.',
            'Keep subject IDs, source paths and MRI images private. Public image permission is separate from local access.', '']
    path_index = {r.image_path: i for i, r in enumerate(data.records)}
    for case in cases:
        subject = case['subject_id']
        case['selection_reason'] = case['selection_reason'].replace('ConvNeXt', 'A1')
        text.extend([f"## {case['case']}: {case['selection_reason']}", '',
                     f"Private subject: `{subject}`", '',
                     '| Model | True class | Prediction | Raw P(AD) | Calibrated P(AD) | Reference | Strict CN | No auto CN |',
                     '|---|---|---|---:|---:|---|---|---|'])
        for alias in ('cnn', 'a1'):
            r = next(r for r in predictions[alias]['patients'] if r['subject_id'] == subject)
            text.append(f"| {alias} | {['CN','AD'][r['label']]} | {['CN','AD'][r['prediction']]} | {r['p_ad']:.4f} | {r['calibrated_p_ad']:.4f} | " + ' | '.join(r['actions'][k] for k in policy.POLICIES) + ' |')
        patient = next(r for r in predictions['a1']['patients'] if r['subject_id'] == subject)
        scans = [r for r in predictions['a1']['scans'] if r['subject_id'] == subject]
        chosen_scan = min(scans, key=lambda r: (abs(r['p_ad']-patient['p_ad']), r['scan_id']))
        slices = sorted([r for r in predictions['a1']['slices'] if r['scan_id'] == chosen_scan['scan_id']],
                        key=lambda r: r['slice_index'])
        panel = Image.new('L', (720, 272), 255)
        draw = ImageDraw.Draw(panel)
        draw.text((8, 6), f"{case['case']} | PRIVATE representative inputs, not a feature explanation", fill=0)
        for col, fraction in enumerate((.25, .5, .75)):
            r = slices[round((len(slices)-1)*fraction)]
            tensor = data[path_index[r['image_path']]]['image']
            pixels = ((tensor[0]+1)*127.5).clamp(0, 255).byte().numpy()
            panel.paste(Image.fromarray(pixels), (240*col+8, 28))
            draw.text((240*col+8, 254), f"Slice {r['slice_index']}; P(AD)={r['p_ad']:.3f}", fill=0)
        name = case['case'] + '_PRIVATE_inputs.png'
        panel.save(folder / name)
        text.extend(['', f'![Private input panel]({name})', '',
            f"A1 scan probability range: {min(r['p_ad'] for r in scans):.3f}–{max(r['p_ad'] for r in scans):.3f}.", '',
            '**Observed image/data properties:** [COMPLETE — do not invent findings.]', '',
            '**Possible explanation and uncertainty:** [COMPLETE — separate hypothesis from evidence.]', '',
            '**CNN comparison and policy consequences:** [COMPLETE — false-negative/false-positive implications and all policies.]', '',
            '**Recommended limitation or safeguard:** [COMPLETE — no tuning against these test cases.]', ''])
    worksheet = folder / 'PRIVATE_failure_review.md'
    if not worksheet.exists(): worksheet.write_text('\n'.join(text), encoding='utf-8')
    atomic_json(folder / 'PRIVATE_case_selection.json', cases)
    status = {'cases_selected': len(cases), 'manual_failure_review_completed': False,
              'review_for_selected_A1_not_V1': True, 'medical_images_remain_private': True,
              'panels_are_not_causal_explanations': True,
              'note': 'Generated examples are not a completed qualitative analysis. Write your observations separately.'}
    atomic_json(output / 'failure_review_status.json', status)
    return status


def plots(output, predictions, reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(7, 5))
    for alias in ('cnn', 'a1'):
        rows = predictions[alias]['patients']
        x, y, _ = roc_curve([r['label'] for r in rows], [r['p_ad'] for r in rows])
        ax.plot(x, y, label=f"{alias.upper()} (AUROC {reports[alias]['metrics']['patient']['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], '--', label='Chance ranking')
    ax.set(xlabel='False-positive rate (CN)', ylabel='True-positive rate (AD)',
           title='Revised test — prior cohort exposure disclosed', xlim=(0, 1), ylim=(0, 1))
    ax.legend(); fig.tight_layout(); fig.savefig(output / PLOTS[0], dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for alias in ('cnn', 'a1'):
        for key, style in (('raw_probability_scores', '--'), ('fixed_temperature_probability_scores', '-')):
            bins = [b for b in reports[alias][key]['reliability_bins'] if b['n']]
            ax.plot([b['mean_predicted_ad_probability'] for b in bins], [b['observed_ad_fraction'] for b in bins],
                    marker='o', linestyle=style, label=f"{alias.upper()} {'raw' if key.startswith('raw') else 'fixed T'}")
    ax.plot([0, 1], [0, 1], ':', label='Perfect calibration')
    ax.set(xlabel='Mean predicted P(AD)', ylabel='Observed AD fraction', xlim=(0, 1), ylim=(0, 1),
           title='Revised test — fixed validation-fitted temperatures')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output / PLOTS[1], dpi=160); plt.close(fig)


def fmt(value, percent=False):
    if value is None: return 'Not estimable'
    return f'{100*value:.2f}%' if percent else f'{value:.4f}'


def markdown(summary):
    lines = ['## Revised evaluation: selected ConvNeXt A1', '', CAUTION, '',
             'A1 seed 42 / epoch 13 and the regularized CNN epoch 14 were fixed before revised inference. '
             'Pretraining was not used. The primary classification cutoff remains 0.5. All three declared '
             'referral policies are reported; none is selected according to revised test performance.', '',
             '### All-patient classification (primary)', '',
             '| Model | Correct / N | Accuracy | Macro-F1 | AUROC | AD recall | CN recall |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for alias, r in summary['models'].items():
        m, ci = r['metrics']['patient'], r['patient_accuracy_wilson95']
        lines.append(f"| {r['name']} | {ci['count']}/{ci['denominator']} | {fmt(m['accuracy'],True)} | {fmt(m['macro_f1'])} | {fmt(m['auroc'])} | {fmt(m['per_class']['AD']['recall'],True)} | {fmt(m['per_class']['NC']['recall'],True)} |")
    lines += ['', '### Per-class precision and confusion counts', '',
              '| Model | CN precision | AD precision | TN | FP | FN | TP |', '|---|---:|---:|---:|---:|---:|---:|']
    for alias, r in summary['models'].items():
        m = r['metrics']['patient']; cm = m['confusion_matrix_rows_true_columns_predicted_NC_AD']
        lines.append(f"| {alias} | {fmt(m['per_class']['NC']['precision'])} | {fmt(m['per_class']['AD']['precision'])} | {cm[0][0]} | {cm[0][1]} | {cm[1][0]} | {cm[1][1]} |")
    lines += ['', '### Other units and target reporting', '',
              '| Model | Patient accuracy | Scan accuracy | Slice accuracy |', '|---|---:|---:|---:|']
    for alias, r in summary['models'].items():
        lines.append('| ' + alias + ' | ' + ' | '.join(fmt(r['metrics'][k]['accuracy'], True) for k in ('patient','scan','slice')) + ' |')
        ci = r['patient_accuracy_wilson95']
        lines.append(f"\n{alias}: descriptive Wilson 95% patient-accuracy interval {fmt(ci['lower'],True)}–{fmt(ci['upper'],True)}.\n")
    target = summary['models']['a1']['observed_accuracy_at_least_0_80_by_unit']
    lines += [f"**Observed A1 accuracy >=0.80:** patient={target['patient']}, scan={target['scan']}, slice={target['slice']}. "
              'The primary unit was declared as patient. The brief does not explicitly settle the threshold unit. '
              'These observed flags do not guarantee marks, a fresh independent estimate or clinical suitability.', '',
              '### All fixed policies: safety and workload', '',
              '| Model | Policy | Auto CN | AD flagged | Refer | AD incorrectly auto CN / all AD | Accepted error | Clinical reviews* |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for alias, r in summary['models'].items():
        for name, m in r['policies'].items():
            lines.append(f"| {alias} | {name} | {m['n_auto_CN']} | {m['n_AD_flags']} | {m['n_unresolved_referrals']} | {m['n_AD_auto_CN']}/{m['n_true_AD']} | {fmt(m['automated_research_label_error']['estimate'],True)} | {m['n_clinical_reviews_if_all_AD_flags_also_reviewed']} |")
    lines += ['', '*Clinical reviews assume all AD flags also need confirmation; this workload is calculated, not observed. '
              'Referrals are not successful diagnoses. No-auto-CN removes automated negatives by construction. '
              'Accepted-subset accuracy does not replace all-patient accuracy. The 0.05 strict-CN cap is a fixed research choice, not a clinical safety bound. '
              'Descriptive policy confidence intervals and denominators are in the JSON.', '',
              '### Probability quality', '', '| Model | Raw NLL | Fixed-T NLL | Raw Brier | Fixed-T Brier |', '|---|---:|---:|---:|---:|']
    for alias, r in summary['models'].items():
        a,b = r['raw_probability_scores'], r['fixed_temperature_probability_scores']
        lines.append(f"| {alias} | {fmt(a['nll'])} | {fmt(b['nll'])} | {fmt(a['brier'])} | {fmt(b['brier'])} |")
    paired = summary['paired_patient_comparison']
    lines += ['', f"Paired A1 minus CNN accuracy: {fmt(paired['convnext_minus_cnn_accuracy'],True)}; "
              f"exploratory patient-bootstrap interval {fmt(paired['paired_patient_bootstrap_95_percentile'][0],True)} to {fmt(paired['paired_patient_bootstrap_95_percentile'][1],True)}. "
              'This interval excludes training-seed/site uncertainty and the effects of previous test exposure.', '',
              '### Resources and provenance', '',
              '| Model | Parameters | Completed test-batch seconds | Peak allocated GiB | Batch-1 forward median ms** |',
              '|---|---:|---:|---:|---:|']
    for alias, r in summary['models'].items():
        x=r['resources']; prior=x['prior_validation_model_only_inference']
        lines.append(f"| {alias} | {x['parameters']:,} | {x['revised_test_batches']['batch_seconds']:.3f} | {x['revised_test_batches']['peak_allocated_gib']} | {prior['median_ms_per_slice']:.3f} |")
    lines += ['', '**Previously measured for these exact checkpoints during the successful validation replay; not remeasured or mislabeled as test latency. '
              'New batch timings include hash/decode/preprocess/device transfer and exclude cache writes, recovery verification and setup. '
              'Memory is the process allocator peak for inference, not total Windows VRAM or training peak. Earlier training costs remain in the development reports.', '',
              '### Failure analysis still requires manual work', '',
              f"{summary['failure_review']['cases_selected']} unique cases were selected using the predeclared rule. "
              'Complete the private worksheet for the selected A1 checkpoint, including the CNN and all three policy actions. '
              'Input panels do not explain model causality. Do not publish medical images or patient identifiers without applicable permission.', '',
              '### Artificial Intelligence Usage Disclosure — update', '',
              'ChatGPT assisted with integration, tests and this generated draft. Record the code review, tests, '
              'manual observations and corrections actually performed. Automated success is not a clinical assessment.', '',
              f'![Revised patient ROC](figures/{PLOTS[0]})', '', f'![Revised fixed-temperature reliability](figures/{PLOTS[1]})', '']
    return '\n'.join(lines)


def bundle(output):
    temp = output / (BUNDLE + '.tmp')
    with ZipFile(temp, 'w', ZIP_DEFLATED) as z:
        for name in PUBLIC: z.write(output / name, name)
    temp.replace(output / BUNDLE)


def validate_completed(folder, plan):
    summary = read_json(folder / SUMMARY)
    require(summary.get('quantitative_evaluation_completed') is True
            and summary.get('prior_test_exposure_disclosed') is True
            and summary.get('plan_sha256') == sha256(folder / PLAN_FILE), 'Completion evidence differs.')
    require(read_json(folder / PLAN_FILE) == plan, 'Completed plan changed.')
    hashes = summary.get('aggregate_output_sha256', {})
    require(set(hashes) == set(PUBLIC) - {SUMMARY}, 'Incomplete aggregate output manifest.')
    for name, digest in hashes.items():
        require(sha256(folder / name) == digest, 'Completed aggregate output changed; preserve it.')
    return summary


def evaluate(private, *, enforce_git=True, factories=None):
    """Factories are a synthetic-test hook; the CLI has no alternate-model option."""
    ctx, folder, plan, commit = checked(private, committed=enforce_git)
    with run_lock(folder):
        if (folder / SUMMARY).exists():
            validate_completed(folder, plan)
            bundle(folder); set_pointer(ctx['private'], folder)
            print('A1_REVISED_EVALUATION_ALREADY_COMPLETED'); return folder
        identity = {'plan_sha256': sha256(folder / PLAN_FILE), 'environment': plan['environment']}
        access = folder / 'REVISED_TEST_ACCESS_STARTED.json'
        if access.exists():
            require(read_json(access)['identity'] == identity, 'Existing inference belongs to a different plan.')
        else:
            atomic_json(access, {'started_utc': now(), 'identity': identity, 'git_commit_at_first_access': commit,
                                 'prior_test_exposure_disclosed': True})
        seed_everything(42)
        device = choose_device(plan['configuration']['device'])
        data = VerifiedTestDataset(ctx['bundle'], ctx['test_hashes'])
        reports, predictions = {}, {}
        for alias, candidate in ctx['candidates'].items():
            model = sa.load_selected_checkpoint(candidate, ctx['bundle'].manifest_hashes, device, factories)
            this_identity = {**identity, 'alias': alias, 'checkpoint_sha256': candidate.hashes['checkpoint_sha256']}
            p, timing = infer_cached(model, data, device, folder / f'PRIVATE_{alias}_progress.json',
                                    this_identity, plan['configuration']['batch_size'])
            reports[alias], predictions[alias] = score(data.records, p, plan['models'][alias])
            replay = ctx['confidence']['models'][alias]['checkpoint_replay']
            reports[alias]['resources'] = {'parameters': candidate.report['parameters'],
                'hardware': hardware_info(device), 'revised_test_batches': timing,
                'prior_validation_model_only_inference': replay['model_only_inference'],
                'batch1_latency_source': 'Earlier successful checkpoint replay on validation inputs; not newly measured on test.'}
            atomic_json(folder / f'PRIVATE_{alias}_predictions.json', predictions[alias])
            del model; gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()
        paired = paired_comparison(predictions['cnn']['patients'], predictions['a1']['patients'],
                                   plan['configuration']['bootstrap_repeats'], plan['configuration']['bootstrap_seed'])
        review = build_cases(folder, data, predictions)
        plots(folder, predictions, reports)
        # Recheck bound sources/evidence to catch a concurrent edit before declaring completion.
        again = load_development(ctx['private'])
        require(make_plan(again, plan['configuration']['device']) == plan, 'Evidence changed during revised evaluation.')
        summary = {'format_version': 1, 'created_utc': now(), 'quantitative_evaluation_completed': True,
            'prior_test_exposure_disclosed': True, 'fresh_independent_test': False,
            'model_training_performed': False, 'test_used_for_fitting_or_selection': False,
            'calibration_fitted': False, 'threshold_search_performed': False,
            'old_run_directories_or_pointers_modified': False,
            'new_test_model_evaluation_performed': True,
            'test_counts': plan['test_counts'], 'manifest_sha256': plan['manifest_sha256'],
            'plan_sha256': sha256(folder / PLAN_FILE), 'source_code_sha256': plan['source_code_sha256'],
            'environment': plan['environment'], 'git_commit_at_first_access': read_json(access)['git_commit_at_first_access'],
            'models': reports, 'paired_patient_comparison': paired, 'failure_review': review,
            'manual_failure_review_completed': False, 'no_clinical_safety_claim': True, 'caution': CAUTION}
        (folder / 'README_revised_test_results.md').write_text(markdown(summary), encoding='utf-8')
        summary['aggregate_output_sha256'] = {name: sha256(folder / name) for name in PUBLIC if name != SUMMARY}
        atomic_json(folder / SUMMARY, summary)
        bundle(folder); set_pointer(ctx['private'], folder)
    print('A1_REVISED_EVALUATION_COMPLETED')
    print('PRIOR_TEST_EXPOSURE_DISCLOSED_MANUAL_FAILURE_REVIEW_REQUIRED')
    print(folder / BUNDLE)
    return folder


def publish(private):
    ctx, folder, plan, _ = checked(private)
    with run_lock(folder):
        validate_completed(folder, plan)
        readme = ctx['project'] / 'README.md'
        old = readme.read_bytes()
        figure_dir = ctx['project'] / 'figures'
        if MARKER.encode() in old:
            for name in PLOTS:
                require((figure_dir / name).is_file() and sha256(figure_dir / name) == sha256(folder / name),
                        'Published figures differ; preserve them for review.')
            print('A1_REVISED_REPORT_ALREADY_PRESENT'); return
        for name in PLOTS:
            dest = figure_dir / name
            require(not dest.exists() or sha256(dest) == sha256(folder / name), 'Conflicting existing figure; not overwritten.')
        backup = ctx['private'] / ('a1-revised-readme-backup-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        backup.mkdir(); shutil.copy2(readme, backup / 'README.md')
        figure_dir.mkdir(exist_ok=True)
        for name in PLOTS: shutil.copy2(folder / name, figure_dir / name)
        new = old + b'\n\n' + MARKER.encode() + b'\n' + (folder / 'README_revised_test_results.md').read_bytes()
        temp = readme.with_suffix('.md.tmp'); temp.write_bytes(new); temp.replace(readme)
    print('A1_REVISED_REPORT_APPENDED')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('revised-freeze', 'revised-evaluate', 'revised-publish'):
        p = sub.add_parser(name)
        p.add_argument('--private-root', type=Path, required=True)
        if name == 'revised-freeze': p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
        if name == 'revised-evaluate': p.add_argument('--confirm-reused-test', action='store_true')
    a = parser.parse_args(argv)
    try:
        if a.command == 'revised-freeze': freeze(a.private_root, a.device)
        elif a.command == 'revised-evaluate':
            require(a.confirm_reused_test, 'Explicitly acknowledge the previously exposed test using --confirm-reused-test.')
            evaluate(a.private_root)
        else: publish(a.private_root)
    except (Exception, KeyboardInterrupt) as exc:
        # Error trace may contain local paths; keep it private and out of the result bundle.
        folder = Path(a.private_root) / FOLDER
        if folder.is_dir():
            (folder / 'PRIVATE_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print(f'A1_REVISED_STOPPED: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('Preserve all records. Do not retrain, alter cutoffs, replace the plan, or delete progress.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
