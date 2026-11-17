"""Frozen final-test evaluation for COMP3710; no fitting, search, or training.

New code only: existing dataset/model/training/calibration sources are unchanged.
Patient predictions are mean(scan means(slice P(AD))). Calibrate that aggregate,
not individual slices. Test labels are used for reporting only. Partial inference
can resume for the SAME frozen plan, device/software and batch settings.

Development assistance: ChatGPT. Review, test, and disclose assistance accurately.
The private image panels are descriptive error-review aids, not clinical findings.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any, Sequence
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image, ImageDraw
import torch

from dataset import (ADNISliceDataset, SliceRecord, image_to_tensor,
                     load_finalized_splits, preprocessing_spec)
from training_utils import (binary_metrics, choose_device, hardware_info,
                            inference_benchmark, memory_info, reset_memory_peak,
                            seed_everything, synchronize, load_checkpoint)
from train_experiment import load_model
from analyse_confidence import (check_inputs, pointer, read_json, same_numbers,
                                sha256, write_json)
from confidence_analysis import (apply_referral, confidence_diagnostics,
    probability_scores, scaled_probabilities, triage_metrics, wilson_interval)

NEW_FILES = ('predict.py', 'final_evaluation.py', 'FINAL_EVALUATION_METHOD.md',
             'tests_final/test_final_evaluation.py')
PLAN_NAME = 'FINAL_EVALUATION_PLAN.json'
PLOTS = ('step10_test_roc.png', 'step10_test_reliability.png')
PUBLIC_FILES = (PLAN_NAME, 'final_test_summary.json', 'README_final_test_results.md',
                'failure_review_status.json', *PLOTS)
CAUTION = ('Research prototype, not a clinical diagnostic system. One development/test split and '
           'one training seed do not establish clinical validity or generalisation to other sites. '
           'The held-out test is now exposed: do not tune, refit temperatures, change referral '
           'cutoffs, swap checkpoints, or change the patient split using these results.')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_pointer(private: Path, name: str, directory: Path) -> None:
    path = private / name
    temp = path.with_suffix('.tmp')
    temp.write_text(str(directory.resolve()) + '\n', encoding='utf-8')
    os.replace(temp, path)


def git(project: Path, *args: str) -> bytes:
    r = subprocess.run(['git', '-C', str(project), *args], capture_output=True)
    require(r.returncode == 0, 'Git check failed; preserve the working tree and inspect locally.')
    return r.stdout


def project_check(project: Path) -> None:
    prefix = git(project, 'rev-parse', '--show-prefix').decode().strip()
    require(prefix.startswith('recognition/') and prefix != 'recognition/'
            and git(project, 'branch', '--show-current').decode().strip() == 'topic-recognition',
            'Expected the saved assignment folder inside recognition on topic-recognition.')


def software_versions() -> dict[str, str]:
    import PIL, sklearn
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': str(np.__version__), 'pillow': str(PIL.__version__),
            'sklearn': str(sklearn.__version__)}


def current_sources(project: Path, inputs: dict, step9: dict) -> dict[str, str]:
    expected = dict(inputs['models']['convnext']['report']['source_code_sha256'])
    expected.update(step9['source_code_sha256'])
    for name, digest in expected.items():
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe source path.')
        require(sha256(project / rel) == digest, 'Historical source changed; do not reinstall or retrain blindly.')
    expected.update({name: sha256(project / name) for name in NEW_FILES})
    return expected


def frozen_models(inputs: dict, summary: dict, rules: dict) -> dict:
    require(set(rules.get('models', {})) == {'cnn', 'convnext'}, 'Unexpected saved rule models.')
    require(rules.get('manifest_sha256') == inputs['manifest_sha256']
            and rules.get('input_probability_clip') == [1e-7, 1-1e-7], 'Operating-rule inputs differ.')
    result = {}
    for alias, info in inputs['models'].items():
        prior = summary['models'][alias]
        rule = rules['models'][alias]
        same_numbers(info['input_hashes'], prior['input_hashes'], 'Step 9 input hashes')
        same_numbers(info['report']['selected_validation_metrics']['patient'],
                     prior['original_validation_patient_metrics'], 'Step 9 selected model')
        require(rule['checkpoint_sha256'] == info['input_hashes']['selected_checkpoint_sha256']
                and rule['selected_epoch'] == info['report']['selected_epoch']
                and rule['name'] == info['name'] and rule['fit_partition'] == 'validation'
                and rule['fit_patients'] == len(info['subjects'])
                and rule['diagnostic_threshold'] == 0.5,
                'Saved rule refers to a different checkpoint or partition.')
        same_numbers(rule['temperature'], prior['full_validation_temperature_fit']['temperature'], 'temperature')
        same_numbers(rule['confidence_threshold'], prior['operating_rule']['confidence_threshold'], 'referral cutoff')
        require(rule['refer_all'] == (rule['confidence_threshold'] is None), 'Refer-all rule is inconsistent.')
        require(math.isfinite(rule['temperature']) and 0.5 <= rule['temperature'] <= 10, 'Invalid saved temperature.')
        if rule['confidence_threshold'] is not None:
            require(0.5 <= rule['confidence_threshold'] <= 1, 'Invalid saved cutoff.')
        result[alias] = dict(rule)
    return result


def freeze(private: Path, device: str = 'cuda') -> Path:
    """Freeze before inference. Reads metadata, cached VALIDATION data and hashes only."""
    private = private.resolve()
    require(device in ('cpu', 'cuda'), 'Expected cpu or cuda.')
    existing = private / 'latest_final_evaluation.txt'
    require(not existing.exists(), 'A final-evaluation plan already exists. Use evaluate for that same plan, not freeze again.')
    require(not (private / 'FINAL_TEST_ACCESS_STARTED.json').exists(), 'Final test has already been opened. Preserve the existing plan.')
    inputs = check_inputs(private)  # Existing validated Step 9 input verification, no image decoding.
    project = inputs['project']; project_check(project)
    require(project == Path(__file__).resolve().parent, 'Run the installed project version, not the tools copy.')
    require(not (project / PLAN_NAME).exists(), 'A project evaluation plan already exists; do not replace it.')
    step9_dir = pointer(private, 'latest_step9_confidence.txt')
    step9 = read_json(step9_dir / 'confidence_summary.json')
    rules = read_json(step9_dir / 'operating_rules.json')
    require(step9.get('step9_completed') is True and step9.get('final_test_model_evaluation_performed') is False
            and step9.get('comparison_sha256') == inputs['comparison_sha256']
            and step9.get('manifest_sha256') == inputs['manifest_sha256'], 'Step 9 evidence does not match this experiment.')
    models = frozen_models(inputs, step9, rules)
    sources = current_sources(project, inputs, step9)
    splits = pointer(private, 'latest_patient_splits.txt')
    root = private / 'data' / 'ADNI'
    bundle = load_finalized_splits(root, splits)
    audit = pointer(private, 'latest_adni_audit.txt')
    inventory_path = audit / 'private_inventory.json'
    inventory = read_json(inventory_path)
    index = {}
    for row in inventory.get('images', []):
        key = row.get('relative_path')
        require(isinstance(key, str) and key not in index, 'Audit inventory has missing/duplicate image keys.')
        index[key] = row
    test_hashes = {}
    for r in bundle.records['test']:
        item = index.get(r.image_path, {})
        h = item.get('file_sha256', '')
        require(item.get('status') == 'decoded' and isinstance(h, str) and len(h) == 64
                and all(c in '0123456789abcdef' for c in h), 'Missing audited test-image byte hash.')
        test_hashes[r.image_path] = h
    model_locations = {}
    chosen_pointer = ('latest_step8_baseline.txt' if models['cnn']['name'] == 'Regularized CNN'
                      else 'latest_baseline_run.txt')
    for alias, pfile in [('cnn', chosen_pointer), ('convnext', 'latest_step8_convnext.txt')]:
        run = pointer(private, pfile)
        original = pfile == 'latest_baseline_run.txt'
        cp = run / ('PRIVATE_best_baseline.pt' if original else 'PRIVATE_best_model.pt')
        report = inputs['models'][alias]['report']
        model_locations[alias] = {'checkpoint': str(cp), 'original_baseline_format': original,
            'run_report': str(run / ('baseline_summary.json' if original else 'experiment_summary.json'))}
        models[alias]['training_resources'] = {
            'parameters': report.get('parameters'), 'training_loop_seconds': report.get('training_loop_seconds'),
            'gpu_memory': report.get('training_loop_gpu_memory'), 'model_only_inference': report.get('model_only_inference')}
    plan = {'format_version': 1, 'status': 'FROZEN_BEFORE_FINAL_TEST', 'created_utc': utc_now(),
        'models': models, 'manifest_sha256': bundle.manifest_hashes,
        'preprocessing': preprocessing_spec(0.0),
        'test_counts': bundle.summary['proposed_or_final_counts']['test'],
        'source_code_sha256': sources, 'software_versions': software_versions(),
        'step9_summary_sha256': sha256(step9_dir / 'confidence_summary.json'),
        'operating_rules_sha256': sha256(step9_dir / 'operating_rules.json'),
        'prior_audit_inventory_sha256': sha256(inventory_path),
        'configuration': {'device': device, 'batch_size': 8, 'num_workers': 0, 'seed': 42,
            'precision': 'float32; no mixed precision or TF32',
            'diagnostic_threshold': 0.5, 'patient_aggregation': 'mean of scan means of slice P(AD)',
            'temperature_application': 'patient aggregate log-odds only; not slice/scan calibration',
            'accuracy_target': 0.8, 'primary_unit': 'patient', 'secondary_units': ['scan', 'slice'],
            'paired_accuracy_bootstrap_replicates': 5000, 'paired_accuracy_bootstrap_seed': 2026,
            'failure_selection': 'Up to 5 unique patients: confident ConvNeXt FN; confident FP; error corrected by CNN; lowest-confidence remaining error; highest-confidence remaining errors; then CNN errors if needed.',
            'case_scan_choice': 'scan probability nearest the patient probability; scan ID breaks ties',
            'case_slices': 'ordered slice positions 25%, 50%, 75%; images stay private'},
        'test_used_for_fitting': False, 'no_post_test_tuning': True,
        'caution': CAUTION, 'manual_failure_review_required': True}
    out = private / ('final-evaluation-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    require(not any((p / '.git').exists() for p in (private, *private.parents)), 'Private output must be outside Git.')
    out.mkdir()
    write_json(out / PLAN_NAME, plan)
    write_json(out / 'PRIVATE_locations.json', {'project': str(project), 'data_root': str(root),
        'splits': str(splits), 'step9_dir': str(step9_dir), 'audit_inventory': str(inventory_path),
        'models': model_locations, 'test_image_sha256': test_hashes})
    shutil.copy2(out / PLAN_NAME, project / PLAN_NAME)
    set_pointer(private, 'latest_final_evaluation.txt', out)
    print('FINAL_EVALUATION_PLAN_FROZEN')
    print('COMMIT_THE_PLAN_AND_NEW_CODE_BEFORE_EVALUATING')
    return out


def require_committed_plan(project: Path, plan: dict) -> str:
    project_check(project)
    prefix = git(project, 'rev-parse', '--show-prefix').decode().strip()
    try:
        committed = json.loads(git(project, 'show', 'HEAD:' + prefix + PLAN_NAME).decode('utf-8-sig'))
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Commit the frozen evaluation plan before opening the final test.') from exc
    require(committed == plan, 'Committed evaluation plan does not match the private frozen plan.')
    git(project, 'diff', '--exit-code', 'HEAD', '--', *NEW_FILES, PLAN_NAME)
    for name in NEW_FILES:
        git(project, 'ls-files', '--error-unmatch', '--', name)
    return git(project, 'rev-parse', 'HEAD').decode().strip()


def checked_plan(private: Path, *, committed: bool = True):
    private = private.resolve(); out = pointer(private, 'latest_final_evaluation.txt')
    plan = read_json(out / PLAN_NAME); loc = read_json(out / 'PRIVATE_locations.json')
    require(plan.get('status') == 'FROZEN_BEFORE_FINAL_TEST' and plan.get('no_post_test_tuning') is True,
            'Invalid frozen plan.')
    project = pointer(private, 'project_path.txt')
    require(project == Path(loc['project']).resolve() == Path(__file__).resolve().parent, 'Saved project changed.')
    require(read_json(project / PLAN_NAME) == plan, 'Project plan changed after freezing.')
    for name, h in plan['source_code_sha256'].items():
        require(sha256(project / name) == h, 'Frozen source changed; preserve results and diagnose before modifying experiments.')
    require(software_versions() == plan['software_versions'], 'Software versions changed after freezing.')
    step9 = Path(loc['step9_dir'])
    require(sha256(step9 / 'confidence_summary.json') == plan['step9_summary_sha256']
            and sha256(step9 / 'operating_rules.json') == plan['operating_rules_sha256'], 'Step 9 evidence changed.')
    require(sha256(Path(loc['audit_inventory'])) == plan['prior_audit_inventory_sha256'], 'Audited image inventory changed.')
    bundle = load_finalized_splits(loc['data_root'], loc['splits'])
    require(bundle.manifest_hashes == plan['manifest_sha256']
            and bundle.summary['proposed_or_final_counts']['test'] == plan['test_counts'], 'Frozen partition changed.')
    for alias, m in plan['models'].items():
        require(sha256(Path(loc['models'][alias]['checkpoint'])) == m['checkpoint_sha256'], 'Frozen checkpoint changed.')
    commit = require_committed_plan(project, plan) if committed else None
    return out, plan, loc, bundle, commit


class VerifiedTestDataset(ADNISliceDataset):
    """Hash the exact bytes being decoded against the old audit; no silent replacement."""
    def __init__(self, bundle, hashes):
        super().__init__(bundle, 'test', allow_test=True)
        require(set(hashes) == {r.image_path for r in self.records}, 'Test-image hashes do not cover the frozen manifest.')
        self.hashes = hashes

    def verified_bytes(self, i: int) -> bytes:
        raw = self.source_path(i).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == self.hashes[self.records[i].image_path],
                'A test image differs from the pre-modelling audit; do not substitute images.')
        return raw

    def __getitem__(self, i: int):
        with Image.open(io.BytesIO(self.verified_bytes(i))) as img:
            img.load(); tensor = image_to_tensor(img)
        return {'image': tensor, 'image_path': self.records[i].image_path}


def validate_progress(cache: dict, identity: dict, total: int, batch: int) -> list[float]:
    require(cache.get('identity') == identity, 'Partial inference belongs to different frozen inputs/device/software.')
    p = cache.get('probabilities')
    require(isinstance(p, list) and len(p) <= total
            and (len(p) % batch == 0 or len(p) == total), 'Partial inference has an invalid completed-batch boundary.')
    require(all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1 for x in p),
            'Partial inference contains invalid probabilities.')
    return list(p)


def cached_inference(model: torch.nn.Module, data: VerifiedTestDataset, device: torch.device,
                     cache_path: Path, identity: dict, batch_size: int = 8) -> tuple[list, dict]:
    total = len(data)
    if cache_path.exists():
        cache = read_json(cache_path)
        probabilities = validate_progress(cache, identity, total, batch_size)
        # Verify previously consumed bytes before reusing predictions after interruption.
        for i in range(len(probabilities)):
            data.verified_bytes(i)
    else:
        probabilities = []; cache = {'identity': identity, 'probabilities': [],
            'end_to_end_seconds_completed_batches': 0.0, 'model_forward_seconds_completed_batches': 0.0,
            'peak_allocated_gib': None, 'peak_reserved_gib': None}
    model.eval(); reset_memory_peak(device)
    print(f"{identity['alias']}: using {len(probabilities)}/{total} saved predictions", flush=True)
    with torch.inference_mode():
        for start in range(len(probabilities), total, batch_size):
            synchronize(device); t0 = time.perf_counter()
            images = torch.stack([data[i]['image'] for i in range(start, min(start+batch_size, total))]).to(device)
            synchronize(device); t1 = time.perf_counter()
            logits = model(images)
            synchronize(device); t2 = time.perf_counter()
            require(logits.shape == (len(images), 2) and torch.isfinite(logits).all().item(), 'Invalid test logits.')
            values = torch.softmax(logits, 1)[:, 1].cpu().tolist()
            synchronize(device); t3 = time.perf_counter()
            probabilities.extend(values)
            cache['probabilities'] = probabilities
            cache['end_to_end_seconds_completed_batches'] += t3-t0
            cache['model_forward_seconds_completed_batches'] += t2-t1
            mem = memory_info(device)
            for key in ('peak_allocated_gib', 'peak_reserved_gib'):
                if mem[key] is not None:
                    cache[key] = max(cache[key] or 0, mem[key])
            write_json(cache_path, cache)  # atomic: an interrupted batch alone may be recomputed
            if len(probabilities) % 800 == 0 or len(probabilities) == total:
                print(f"{identity['alias']}: saved {len(probabilities)}/{total} test slices", flush=True)
    return probabilities, {k:v for k,v in cache.items() if k not in ('identity', 'probabilities')}


def aggregate_test(records: Sequence[SliceRecord], probabilities: Sequence[float]):
    """Same aggregation as Step 7, explicitly test-only. Do not relax its old guard."""
    require(bool(records) and len(records) == len(probabilities), 'Incomplete test predictions.')
    require(all(r.split == 'test' for r in records), 'This aggregate accepts final-test records only.')
    require(len({r.image_path for r in records}) == len(records), 'Repeated test image.')
    require(all(math.isfinite(float(p)) and 0 <= p <= 1 for p in probabilities), 'Invalid probability.')
    scans = defaultdict(list)
    for r,p in zip(records, probabilities): scans[r.scan_id].append((r, float(p)))
    scan_rows = []
    for sid in sorted(scans):
        rows = scans[sid]; first = rows[0][0]
        require(len({(r.subject_id, r.label) for r,p in rows}) == 1, 'Inconsistent scan grouping.')
        scan_rows.append({'scan_id':sid,'subject_id':first.subject_id,'label':first.label,
            'p_ad':math.fsum(p for r,p in rows)/len(rows),'slices':len(rows)})
    groups = defaultdict(list)
    for r in scan_rows: groups[r['subject_id']].append(r)
    patients = []
    for subject in sorted(groups):
        rows = groups[subject]; require(len({r['label'] for r in rows}) == 1, 'Inconsistent patient label.')
        patients.append({'subject_id':subject,'label':rows[0]['label'],
            'p_ad':math.fsum(r['p_ad'] for r in rows)/len(rows),'scans':len(rows)})
    metrics = {'slice':binary_metrics([r.label for r in records], probabilities),
        'scan':binary_metrics([r['label'] for r in scan_rows], [r['p_ad'] for r in scan_rows]),
        'patient':binary_metrics([r['label'] for r in patients], [r['p_ad'] for r in patients])}
    private = {'slices':[{'image_path':r.image_path, 'subject_id':r.subject_id,
              'scan_id':r.scan_id,'slice_index':r.slice_index,'label':r.label,'p_ad':float(p)} for r,p in zip(records,probabilities)],
              'scans':scan_rows, 'patients':patients}
    return metrics, private


def score_model(records, probabilities, rule):
    metrics, private = aggregate_test(records, probabilities)
    y = np.array([r['label'] for r in private['patients']]); p = np.array([r['p_ad'] for r in private['patients']])
    calibrated = scaled_probabilities(p, rule['temperature'])
    require(np.array_equal(p >= .5, calibrated >= .5), 'Unexpected change in class predictions after temperature scaling.')
    accepted = apply_referral(calibrated, rule['confidence_threshold'])
    triage = triage_metrics(y, calibrated, accepted)
    cm = metrics['patient']['confusion_matrix_rows_true_columns_predicted_NC_AD']
    correct = cm[0][0] + cm[1][1]
    result = {'name':rule['name'], 'selected_epoch':rule['selected_epoch'],
        'metrics':metrics, 'raw_probability_scores':probability_scores(y,p),
        'calibrated_probability_scores':probability_scores(y,calibrated),
        'raw_confidence_diagnostics':confidence_diagnostics(y,p),
        'calibrated_confidence_diagnostics':confidence_diagnostics(y,calibrated),
        'patient_accuracy_wilson95':wilson_interval(correct,len(y)),
        'ad_recall_wilson95':wilson_interval(cm[1][1],sum(cm[1])),
        'nc_recall_wilson95':wilson_interval(cm[0][0],sum(cm[0])),
        'frozen_rule':{k:rule[k] for k in ('temperature','diagnostic_threshold','confidence_threshold','refer_all')},
        'triage':triage,
        'observed_all_patient_accuracy_at_least_0_80':metrics['patient']['accuracy'] >= .8,
        'descriptive_referral_constraints_met':bool(triage['n_accepted'] >= max(20, math.ceil(.25*len(y)))
            and triage['accepted_error_rate'] is not None and triage['accepted_error_rate'] <= .1
            and triage['ad_automatically_cleared_fraction_of_all_ad'] <= .1),
        'clinical_safety_established':False, 'calibration_fitted_on_test':False}
    for r,q,a in zip(private['patients'], calibrated, accepted):
        r.update(calibrated_p_ad=float(q),confidence=float(max(q,1-q)),prediction=int(q>=.5),
                 accepted=bool(a),action=('AUTO_AD' if q>=.5 else 'AUTO_NC') if a else 'REFER')
    return result, private


def paired_comparison(cnn: list, convnext: list, repeats: int = 5000, seed: int = 2026):
    require([(r['subject_id'],r['label']) for r in cnn] == [(r['subject_id'],r['label']) for r in convnext],
            'Paired patient cohorts differ.')
    a = np.array([(r['p_ad']>=.5)==r['label'] for r in cnn])
    b = np.array([(r['p_ad']>=.5)==r['label'] for r in convnext])
    delta = b.astype(float)-a.astype(float)
    sample = np.random.default_rng(seed).integers(0,len(delta),(repeats,len(delta)))
    estimates = delta[sample].mean(axis=1)
    return {'patients':len(delta),'both_correct':int((a&b).sum()),'both_wrong':int((~a&~b).sum()),
        'cnn_only_correct':int((a&~b).sum()),'convnext_only_correct':int((~a&b).sum()),
        'convnext_minus_cnn_accuracy':float(delta.mean()),
        'paired_patient_bootstrap_95_percentile':[float(x) for x in np.quantile(estimates,[.025,.975])],
        'replicates':repeats,'seed':seed,
        'note':'Exploratory patient-resampling interval for fixed models on this cohort, not training/site uncertainty.'}


def select_failure_cases(predictions: dict, maximum: int = 5) -> list:
    a = {r['subject_id']:r for r in predictions['cnn']['patients']}
    b = {r['subject_id']:r for r in predictions['convnext']['patients']}
    require(set(a)==set(b),'Failure-review cohorts differ.')
    bad = [r for r in b.values() if r['prediction']!=r['label']]
    chosen=[]; seen=set()
    def pick(rows, reason, low=False, all_rows=False):
        for r in sorted(rows,key=lambda r: ((r['confidence'] if low else -r['confidence']),r['subject_id'])):
            if r['subject_id'] in seen:continue
            chosen.append({'case':f'F{len(chosen)+1:02d}','subject_id':r['subject_id'],'selection_reason':reason})
            seen.add(r['subject_id'])
            if not all_rows or len(chosen)>=maximum:break
    pick([r for r in bad if r['label']==1],'Confident ConvNeXt false negative')
    if len(chosen)<maximum:pick([r for r in bad if r['label']==0],'Confident ConvNeXt false positive')
    if len(chosen)<maximum:pick([r for r in bad if a[r['subject_id']]['prediction']==r['label']], 'ConvNeXt error corrected by CNN')
    if len(chosen)<maximum:pick(bad,'Low-confidence ConvNeXt error',low=True)
    if len(chosen)<maximum:pick(bad,'Remaining confident ConvNeXt error',all_rows=True)
    if len(chosen)<maximum:pick([r for r in a.values() if r['prediction']!=r['label']], 'CNN error (additional distinct case)',all_rows=True)
    return chosen[:maximum]


def build_failure_review(output: Path, data: VerifiedTestDataset, predictions: dict) -> dict:
    cases = select_failure_cases(predictions)
    private_dir = output / 'PRIVATE_failure_review'; private_dir.mkdir(exist_ok=True)
    paths = {r.image_path:i for i,r in enumerate(data.records)}
    text = ['# PRIVATE final-test failure review — complete manually', '',
        'Do not upload this folder or commit these medical images/identifiers. Describe observable properties; '
        'do not infer a clinical diagnosis or claim a causal anatomical explanation from these slices alone. '
        'Test inspection occurs AFTER model/policy freezing and must not lead to tuning on this test set.', '',
        'Complete 3–5 distinct cases where available. If fewer exist, report that honestly. '
        'Remove individual identifiers from any final report, and obtain applicable permission before publishing medical images.','']
    for case in cases:
        subject=case['subject_id']; label=None
        text += [f"## {case['case']}: {case['selection_reason']}", '',
                 f"Private subject identifier: `{subject}`", '',
                 '| Model | True class | Predicted class | Raw P(AD) | Calibrated P(AD) | Confidence | Frozen action |',
                 '|---|---|---|---:|---:|---:|---|']
        for alias in ('cnn','convnext'):
            r=next(r for r in predictions[alias]['patients'] if r['subject_id']==subject)
            label=r['label']
            text.append(f"| {alias} | {['NC','AD'][label]} | {['NC','AD'][r['prediction']]} | {r['p_ad']:.4f} | {r['calibrated_p_ad']:.4f} | {r['confidence']:.4f} | {r['action']} |")
        cp=next(r for r in predictions['convnext']['patients'] if r['subject_id']==subject)
        scans=[r for r in predictions['convnext']['scans'] if r['subject_id']==subject]
        scan=min(scans,key=lambda r:(abs(r['p_ad']-cp['p_ad']),r['scan_id']))
        rows=sorted([r for r in predictions['convnext']['slices'] if r['scan_id']==scan['scan_id']],key=lambda r:r['slice_index'])
        positions=[round((len(rows)-1)*q) for q in (.25,.5,.75)]
        canvas=Image.new('L',(3*240,272),255); draw=ImageDraw.Draw(canvas)
        draw.text((8,6),f"{case['case']} | Representative processed slices | PRIVATE",fill=0)
        for col,pos in enumerate(positions):
            row=rows[pos]; tensor=data[paths[row['image_path']]]['image']
            arr=np.clip((tensor[0].numpy()+1)*127.5,0,255).astype(np.uint8)
            canvas.paste(Image.fromarray(arr),(col*240+8,28))
            draw.text((col*240+8,254),f"Slice {row['slice_index']}; P(AD)={row['p_ad']:.3f}",fill=0)
        image_name=case['case']+'_PRIVATE_inputs.png';canvas.save(private_dir/image_name)
        text += ['',f'![Private inputs]({image_name})','',
            f"Representative scan selected by proximity to patient probability. Across this patient's scans, "
            f"ConvNeXt P(AD) ranges from {min(r['p_ad'] for r in scans):.3f} to {max(r['p_ad'] for r in scans):.3f}.",'',
            '**Observed visual/data properties:** [COMPLETE — orientation, contrast, cropping, visible artefacts, slice/scan disagreement; do not invent findings.]','',
            '**Possible explanation and uncertainty:** [COMPLETE — separate evidence from hypotheses; these panels do not show which features caused the error.]','',
            '**Comparison and operational consequence:** [COMPLETE — discuss the CNN, AD false negative/NC false positive, confidence, and the frozen referral action.]','',
            '**Recommended boundary or safeguard:** [COMPLETE — propose future work without modifying this frozen test experiment.]','']
    worksheet=private_dir/'PRIVATE_failure_review.md'
    if not worksheet.exists():worksheet.write_text('\n'.join(text),encoding='utf-8')
    write_json(private_dir/'PRIVATE_case_selection.json',cases)
    status={'cases_selected':len(cases),'manual_failure_review_completed':False,
        'medical_images_are_private':True,'image_panels_explain_model_causality':False,
        'selection_rule':'As frozen in FINAL_EVALUATION_PLAN.json; no post-hoc cherry picking.',
        'note':'Complete the private worksheet; generating it does not satisfy the qualitative analysis requirement.'}
    write_json(output/'failure_review_status.json',status)
    return status


def final_plots(output: Path, predictions: dict, reports: dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig,ax=plt.subplots(figsize=(7,5))
    for alias in ('cnn','convnext'):
        rows=predictions[alias]['patients']
        x,y,_=roc_curve([r['label'] for r in rows],[r['p_ad'] for r in rows])
        ax.plot(x,y,label=f"{reports[alias]['name']} (AUROC {reports[alias]['metrics']['patient']['auroc']:.3f})")
    ax.plot([0,1],[0,1],linestyle='--',label='Chance ranking')
    ax.set(xlabel='False-positive rate (NC)',ylabel='True-positive rate (AD)',title='Frozen final test: patient-level ROC',xlim=(0,1),ylim=(0,1))
    ax.legend();fig.tight_layout();fig.savefig(output/PLOTS[0],dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for alias in ('cnn','convnext'):
        for kind,style in [('raw_probability_scores','--'),('calibrated_probability_scores','-')]:
            bins=[r for r in reports[alias][kind]['reliability_bins'] if r['n']]
            ax.plot([r['mean_predicted_ad_probability'] for r in bins],[r['observed_ad_fraction'] for r in bins],
                marker='o',linestyle=style,label=f"{reports[alias]['name']} {'raw' if kind.startswith('raw') else 'fixed T'}")
    ax.plot([0,1],[0,1],linestyle=':',label='Perfect calibration')
    ax.set(xlabel='Mean predicted P(AD)',ylabel='Observed AD fraction',title='Final test: fixed validation-fitted temperatures',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/PLOTS[1],dpi=160);plt.close(fig)


def report_markdown(summary: dict) -> str:
    def percent(x):return 'Not estimable' if x is None else f'{x:.2%}'
    lines=['## Step 10: frozen final-test evaluation','',CAUTION,'',
        'The preferred CNN and ConvNeXt checkpoints, temperatures and referral rules were fixed using development evidence and committed before test inference. '
        'No checkpoint selection, training, temperature fitting or cutoff search used this test set. '
        'The primary unit is a patient; scan and slice results are secondary. Patient probabilities are means of scan means of slice P(AD).','',
        '| Model | Patient accuracy | Macro-F1 | AUROC | AD recall | NC recall |','|---|---:|---:|---:|---:|---:|']
    intervals=[]
    for r in summary['models'].values():
        m=r['metrics']['patient'];lines.append(f"| {r['name']} | {m['accuracy']:.2%} | {m['macro_f1']:.4f} | {m['auroc']:.4f} | {m['per_class']['AD']['recall']:.2%} | {m['per_class']['NC']['recall']:.2%} |")
        ci=r['patient_accuracy_wilson95'];intervals += [f"\n{r['name']}: {ci['count']}/{ci['denominator']} patients correct; descriptive 95% Wilson accuracy interval {ci['lower']:.2%}–{ci['upper']:.2%}."]
    lines += intervals
    # A second table is separate from the primary metric table.
    lines += ['','### Secondary evaluation units','', '| Model | Scan accuracy | Slice accuracy |', '|---|---:|---:|']
    for r in summary['models'].values():lines.append(f"| {r['name']} | {r['metrics']['scan']['accuracy']:.2%} | {r['metrics']['slice']['accuracy']:.2%} |")
    reached=summary['models']['convnext']['observed_all_patient_accuracy_at_least_0_80']
    lines += ['', '**ConvNeXt all-patient 0.80 target: '+('met on this test split.' if reached else 'NOT met on this test split.')+'** '
        'This is a measured result, not a guarantee of marks or clinical performance. Accepted-subset accuracy is not a substitute.', '',
        '### Frozen referral and calibration','',
        '| Model | Coverage | Accepted error | Referred | AD auto-cleared as NC / all AD |','|---|---:|---:|---:|---:|']
    for r in summary['models'].values():
        t=r['triage'];lines.append(f"| {r['name']} | {t['coverage']:.2%} | {percent(t['accepted_error_rate'])} | {t['n_referred']}/{t['n_total']} | {t['n_ad_automatically_cleared_as_nc']}/{t['n_ad_total']} |")
    lines += ['','Refer-all means no automation benefit and no estimable accepted accuracy; it is not perfect performance. '
        'Referred patients are not counted as correctly diagnosed. No human-review accuracy is assumed. '
        'Temperature scaling does not change 0.5-threshold class decisions. The policies are research-only; passing an empirical fraction is not clinical validation.','',
        '| Model | Raw NLL | Fixed-T NLL | Raw Brier | Fixed-T Brier |','|---|---:|---:|---:|---:|']
    for r in summary['models'].values():
        a=r['raw_probability_scores'];b=r['calibrated_probability_scores']
        lines.append(f"| {r['name']} | {a['nll']:.4f} | {b['nll']:.4f} | {a['brier']:.4f} | {b['brier']:.4f} |")
    pair=summary['paired_comparison'];ci=pair['paired_patient_bootstrap_95_percentile']
    lines += ['',f"Paired ConvNeXt minus CNN accuracy: {pair['convnext_minus_cnn_accuracy']:.2%}; exploratory patient-bootstrap 95% percentile interval {ci[0]:.2%}–{ci[1]:.2%}. "
        'This interval does not incorporate training-seed or cross-site variation.','',
        '### Reproducibility and resources','',
        'Checkpoint/source/manifest/plan hashes are recorded in the JSON report. Test inference used float32, batch size 8 and no augmentation. '
        'The earlier training-memory/runtime and batch-1 latency measurements remain in the report. '
        'Additional measured test-batch times exclude result-file writing and resume verification; GPU allocator peaks are not total Windows VRAM use.','',
        '### Failure analysis — manual work outstanding','',
        f"{summary['failure_review']['cases_selected']} distinct cases were selected using the frozen rule. The private worksheet and three-slice panels support manual review. "
        'Generating those files is NOT a completed failure autopsy. Add 3–5 evidence-based case analyses and a final engineering recommendation after reviewing them. '
        'Do not publish patient identifiers or medical images without the applicable authorisation.','',
        '### AI assistance','',
        'ChatGPT assisted with code, synthetic tests and this generated draft. Record the code review, local tests and manual failure analysis actually performed; do not claim independent clinical verification.','']
    for name in PLOTS:lines += [f'![Frozen final-test diagnostic](figures/{name})','']
    return '\n'.join(lines)


def access_marker(private: Path, output: Path, plan_hash: str):
    marker=private/'FINAL_TEST_ACCESS_STARTED.json'
    if marker.exists():
        previous=read_json(marker)
        require(previous.get('plan_sha256')==plan_hash and Path(previous['output_dir']).resolve()==output.resolve(),
                'Another final-test plan was opened. Do not start a fresh evaluation or choose a different run.')
    else:
        # Exclusive creation: concurrent first evaluation cannot silently replace the record.
        with marker.open('x',encoding='utf-8') as f:
            json.dump({'created_utc':utc_now(),'plan_sha256':plan_hash,'output_dir':str(output.resolve())},f,indent=2)


def evaluate(private: Path) -> Path:
    out,plan,loc,bundle,commit=checked_plan(private)
    lock=out/'EVALUATION_RUNNING.lock'
    require(not lock.exists(),'An evaluation lock exists. Do not run concurrently. After a reboot, use recover-lock before evaluate.')
    summary_path=out/'final_test_summary.json'
    if summary_path.exists() and read_json(summary_path).get('quantitative_evaluation_completed') is True:
        if (out/'COMP3710_Step10_Results.zip').is_file():
            print('FINAL_TEST_ALREADY_COMPLETED_NO_INFERENCE_REPEATED');print(out);return out
        print('RESUMING_REPORT_EXPORT_FROM_THE_SAME_SAVED_PREDICTIONS', flush=True)
    plan_hash=sha256(out/PLAN_NAME)
    with lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'started_utc':utc_now()},f)
    try:
        seed_everything(plan['configuration']['seed']); device=choose_device(plan['configuration']['device'])
        access_marker(private,out,plan_hash)
        hardware=hardware_info(device)
        data=VerifiedTestDataset(bundle,loc['test_image_sha256'])
        reports={};predictions={}
        for alias in ('cnn','convnext'):
            m=plan['models'][alias];where=loc['models'][alias]
            if device.type=='cuda':torch.cuda.empty_cache()
            cp=Path(where['checkpoint'])
            if where['original_baseline_format']:
                model,ck=load_checkpoint(cp,device,bundle.manifest_hashes)
            else:
                model,ck=load_model(cp,device,bundle.manifest_hashes,'baseline' if alias=='cnn' else 'convnext')
            require(ck['epoch']==m['selected_epoch'],'Checkpoint epoch differs from frozen model.')
            identity={'plan_sha256':plan_hash,'checkpoint_sha256':m['checkpoint_sha256'],'alias':alias,
                      'manifest_sha256':bundle.manifest_hashes,'hardware':hardware,'batch_size':8}
            p,resources=cached_inference(model,data,device,out/f'PRIVATE_{alias}_progress.json',identity)
            r,private_rows=score_model(bundle.records['test'],p,m)
            r['inference_resources']=resources;r['historical_training_resources']=m['training_resources']
            reports[alias]=r;predictions[alias]=private_rows
            write_json(out/f'PRIVATE_{alias}_test_predictions.json',private_rows)
            del model
        pair=paired_comparison(predictions['cnn']['patients'],predictions['convnext']['patients'])
        failure=build_failure_review(out,data,predictions)
        final_plots(out,predictions,reports)
        summary={'format_version':1,'status':'COMPLETED','created_utc':utc_now(),
            'quantitative_evaluation_completed':True,'manual_failure_review_completed':False,
            'final_test_model_evaluation_performed':True,'unique_test_images_per_model':len(data),
            'evaluation_plan_sha256':plan_hash,'committed_plan_git_commit':commit,
            'manifest_sha256':bundle.manifest_hashes,'source_code_sha256':plan['source_code_sha256'],
            'hardware':hardware,'models':reports,'paired_comparison':pair,'failure_review':failure,
            'test_used_for_fitting_or_selection':False,'clinical_validation':False,
            'additional_model_training_performed':False,'caution':CAUTION}
        write_json(summary_path,summary)
        (out/'README_final_test_results.md').write_text(report_markdown(summary),encoding='utf-8')
        temp=out/'COMP3710_Step10_Results.zip.tmp'
        with ZipFile(temp,'w',ZIP_DEFLATED) as z:
            for name in PUBLIC_FILES:z.write(out/name,name)
        os.replace(temp,out/'COMP3710_Step10_Results.zip')
        print('FINAL_TEST_QUANTITATIVE_EVALUATION_COMPLETED')
        print('MANUAL_FAILURE_ANALYSIS_STILL_REQUIRED')
        print(f'Upload only: {out / "COMP3710_Step10_Results.zip"}')
    except BaseException:
        import traceback
        (out/'PRIVATE_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        # A report-generation interruption must be recoverable from saved predictions.
        if summary_path.exists():
            s=read_json(summary_path);s['quantitative_evaluation_completed']=False;s['status']='INTERRUPTED_OR_FAILED'
            write_json(summary_path,s)
        print('FINAL_TEST_STOPPED_PRESERVE_DIRECTORY_AND_RESUME_SAME_PLAN',flush=True)
        raise
    finally:
        if lock.exists():lock.unlink()
    return out


def recover_lock(private: Path, confirm: bool):
    require(confirm,'Explicitly confirm that no evaluation process is running before clearing a stale lock.')
    out=pointer(private,'latest_final_evaluation.txt'); lock=out/'EVALUATION_RUNNING.lock'
    if not lock.exists():print('NO_STALE_LOCK');return
    record=read_json(lock);pid=record.get('pid')
    # Additional local PID check; cannot detect all races/PID reuse. Never kill a process.
    if isinstance(pid,int):
        if os.name=='nt':
            r=subprocess.run(['tasklist','/FI',f'PID eq {pid}','/FO','CSV','/NH'],capture_output=True,text=True)
            require(r.returncode==0 and f'"{pid}"' not in r.stdout, 'The recorded PID is still present. Do not clear the lock.')
        else:
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            except PermissionError:raise ValueError('Recorded process may still be running.')
            else:raise ValueError('Recorded process is still present; do not clear its lock.')
    saved=out/('STALE_LOCK_'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.json')
    os.replace(lock,saved)
    print('STALE_LOCK_ARCHIVED_RERUN_EVALUATE_WITH_THE_SAME_PLAN')
