"""Selected A1 checkpoint integration and validation-only confidence analysis.

No training, test-image inference, downloads or alteration of previous runs.
Reuses the project's established aggregation and confidence methodology. The
V1 predict.py is intentionally unchanged until the revised evaluation is ready.
The exported load_selected_checkpoint() is the adapter for that later interface.

Assistance: ChatGPT drafted code/tests/instructions. The student must review,
test, understand and disclose actual assistance. Calibration reference is in
confidence_analysis.py; this file does not introduce another fitting method.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ADNISliceDataset, preprocessing_spec
from modules import BaselineCNN, ConvNeXtTiny, parameter_count
from training_utils import (checked_bundle, seed_everything, choose_device,
    aggregate_predictions, inference_benchmark, memory_info, reset_memory_peak,
    synchronize, hardware_info)
from analyse_confidence import (read_json, write_json, pointer, sha256,
                                 same_numbers, verify_cached_predictions)
from confidence_analysis import (analyse_patients, TEMPERATURE_BOUNDS,
    CONFIDENCE_GRID, N_FOLDS, FOLD_SEED, N_BINS, MIN_ACCEPTED, MIN_COVERAGE,
    MAX_EMPIRICAL_ERROR, MAX_AD_AUTOCLEARED_FRACTION)
from train_convnext_v2 import run_lock, environment

NEW_FILES = ('selected_a1.py', 'A1_CONFIDENCE_METHOD.md',
             'tests_selected_a1/test_selected_a1.py')
PLAN_FILE = 'A1_SELECTED_MODEL.json'
FOLDER = 'a1-selected-confidence'
SUMMARY = 'a1_confidence_summary.json'
RULES = 'a1_operating_rules.json'
BUNDLE = 'COMP3710_A1_Confidence_Results.zip'
PLOTS = ('a1_cnn_reliability.png', 'a1_convnext_reliability.png', 'a1_risk_coverage.png')
PUBLIC = (PLAN_FILE, SUMMARY, RULES, 'README_a1_confidence.md', *PLOTS)
POINTER = 'latest_a1_confidence.txt'
MARKER = '<!-- COMP3710-A1-SELECTED-CONFIDENCE -->'
CAUTION = (
    'Selected development results only. Model/recipe/checkpoint selection repeatedly used '
    'these validation patients. Cross-fitting separates temperature/referral fitting from '
    'their fold diagnostics, not from prior model selection. Version 1 already exposed the '
    'existing test cohort; this step performs no new test evaluation. Referral constraints '
    'are empirical project choices, not clinical safety guarantees.')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(project: Path, *args: str) -> bytes:
    r = subprocess.run(['git', '-C', str(project), *args], capture_output=True)
    require(r.returncode == 0, 'Git operation failed; inspect the repository without discarding changes.')
    return r.stdout


def project_check(project: Path) -> None:
    prefix = git(project, 'rev-parse', '--show-prefix').decode().strip()
    require(prefix.startswith('recognition/') and prefix != 'recognition/' and
            git(project, 'branch', '--show-current').decode().strip() == 'topic-recognition',
            'Expected your existing assignment folder inside recognition on topic-recognition.')


def require_committed(project: Path) -> None:
    prefix = git(project, 'rev-parse', '--show-prefix').decode().strip()
    for name in (*NEW_FILES, PLAN_FILE):
        historical = git(project, 'show', 'HEAD:' + prefix + name)
        current = (project / name).read_bytes()
        require(historical.replace(b'\r\n', b'\n') == current.replace(b'\r\n', b'\n'),
                f'Commit the selected-model additions and plan before running: {name}')


def check_sources(project: Path, expected: dict[str, str]) -> None:
    require(bool(expected), 'Source-hash evidence is missing.')
    for name, digest in expected.items():
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts and (project / p).is_file(),
                'A historical source file is missing or its path is invalid.')
        require(sha256(project / p) == digest,
                f'Historical source differs: {name}. Do not overwrite old code or rerun training.')


@dataclass
class Candidate:
    alias: str
    name: str
    folder: Path
    report: dict[str, Any]
    report_name: str
    checkpoint: Path
    prediction_file: Path
    subjects: list[str]
    labels: list[int]
    p: list[float]
    slice_probabilities: list[float]
    hashes: dict[str, str]


def validate_report(report: dict, alias: str, manifests: dict) -> None:
    require(report.get('experiment_completed') is True and report.get('status') == 'COMPLETED',
            'Selected training run is incomplete.')
    require(report.get('manifest_sha256') == manifests, 'Selected run uses different patient manifests.')
    if alias == 'a1':
        require(report.get('recipe') == 'scratch_augmented' and
                report['configuration'].get('recipe') == 'scratch_augmented' and
                report['configuration'].get('seed') == 42 and report.get('selected_epoch') == 13 and
                report.get('pretrained', {}).get('used') is False and
                report.get('operating_threshold') == 0.5 and report.get('parameters') == 27818594,
                'Expected A1 scratch_augmented, seed 42, selected epoch 13, without pretraining.')
    else:
        require(alias == 'cnn' and report.get('model_name') == 'baseline' and
                report.get('selected_epoch') == 14 and
                report.get('preprocessing') == preprocessing_spec(0.0) and
                report.get('parameters') == 359922,
                'Expected the retained regularized CNN, epoch 14, with unchanged preprocessing.')


def candidate(folder: Path, alias: str, bundle, project: Path) -> Candidate:
    name = 'ConvNeXt A1 (seed 42, epoch 13)' if alias == 'a1' else 'Regularized CNN (epoch 14)'
    report_name = 'v2_summary.json' if alias == 'a1' else 'experiment_summary.json'
    report = read_json(folder / report_name)
    validate_report(report, alias, bundle.manifest_hashes)
    check_sources(project, report['source_code_sha256'])
    checkpoint = folder / 'PRIVATE_best_model.pt'
    prediction_file = folder / 'PRIVATE_validation_predictions.json'
    require(checkpoint.is_file() and prediction_file.is_file(), 'Saved selected checkpoint/predictions are missing.')
    digest = sha256(checkpoint)
    if alias == 'cnn':
        require(report.get('selected_checkpoint_sha256') == digest and
                report.get('selected_checkpoint_reloaded_and_checked') is True,
                'CNN checkpoint does not match its completed experiment.')
    subjects, labels, probabilities = verify_cached_predictions(
        prediction_file, bundle.records['validation'], report)
    cached = read_json(prediction_file)
    indexed = {r['image_path']: r['p_ad'] for r in cached['slices']}
    slices = [float(indexed[r.image_path]) for r in bundle.records['validation']]
    return Candidate(alias, name, folder, report, report_name, checkpoint,
        prediction_file, subjects, labels, probabilities, slices,
        {'run_summary_sha256': sha256(folder / report_name),
         'checkpoint_sha256': digest, 'validation_predictions_sha256': sha256(prediction_file)})


def context(private: Path) -> dict:
    private = private.resolve()
    require(private.is_dir() and not any((p / '.git').exists() for p in (private, *private.parents)),
            'Private outputs must be outside the Git repository.')
    project = pointer(private, 'project_path.txt')
    require(project == Path(__file__).resolve().parent, 'Run the installed selected_a1.py, not the tools copy.')
    project_check(project)
    splits = pointer(private, 'latest_patient_splits.txt')
    pipeline = pointer(private, 'latest_data_pipeline.txt') / 'data_pipeline_summary.json'
    bundle = checked_bundle(private / 'data' / 'ADNI', splits, pipeline)
    candidates = {
        'cnn': candidate(pointer(private, 'latest_step8_baseline.txt'), 'cnn', bundle, project),
        'a1': candidate(pointer(private, 'latest_v2_scratch_augmented.txt'), 'a1', bundle, project)}
    a, b = candidates['a1'], candidates['cnn']
    require(a.subjects == b.subjects and a.labels == b.labels, 'Models do not share the same validation patients.')
    # Pin the already-made baseline choice; do not reselect it using calibration.
    comparison = read_json(pointer(private, 'latest_step8_comparison.txt') / 'comparison_summary.json')
    require(comparison.get('preferred_cnn_by_prespecified_validation_rule') == 'Regularized CNN' and
            comparison.get('manifest_sha256') == bundle.manifest_hashes, 'Existing baseline selection differs.')
    same_numbers(comparison['models']['Regularized CNN']['patient_metrics'],
                 b.report['selected_validation_metrics']['patient'], 'retained CNN')
    # Reuse exactly the previous confidence implementation and fit constraints.
    old_confidence = read_json(pointer(private, 'latest_step9_confidence.txt') / 'confidence_summary.json')
    require(old_confidence.get('step9_completed') is True, 'Original confidence analysis is incomplete.')
    check_sources(project, old_confidence['source_code_sha256'])
    sources = {}
    for c in candidates.values():
        sources.update(c.report['source_code_sha256'])
    sources.update(old_confidence['source_code_sha256'])
    sources.update({name: sha256(project / name) for name in NEW_FILES})
    return {'private': private, 'project': project, 'bundle': bundle, 'candidates': candidates,
            'sources': sources, 'old_confidence': old_confidence,
            'pipeline_sha256': sha256(pipeline)}


def selection(ctx: dict) -> dict:
    return {
        'format_version': 1, 'status': 'DEVELOPMENT_SELECTION_NO_NEW_TEST_EVALUATION',
        'decision': 'Retain A1 after S1/S2 did not satisfy the prespecified promotion screen.',
        'models': {alias: {'name': c.name, 'architecture': 'ConvNeXtTiny' if alias == 'a1' else 'BaselineCNN-v1',
            'selected_epoch': c.report['selected_epoch'], 'configuration': c.report['configuration'],
            'input_hashes': c.hashes, 'validation_metrics': c.report['selected_validation_metrics'],
            'parameters': c.report['parameters']} for alias, c in ctx['candidates'].items()},
        'manifest_sha256': ctx['bundle'].manifest_hashes,
        'preprocessing': preprocessing_spec(0.0), 'model_label_map': {'NC': 0, 'AD': 1},
        'diagnostic_threshold': 0.5, 'patient_aggregation': 'mean of scan means of slice P(AD)',
        'verification': {'batch_size': 8, 'workers': 0, 'absolute_probability_tolerance': 2e-5,
                         'require_identical_class_decisions_at_all_three_levels': True},
        'calibration': {'temperature_bounds': list(TEMPERATURE_BOUNDS), 'n_folds': N_FOLDS,
            'fold_seed': FOLD_SEED, 'reliability_bins': N_BINS, 'confidence_grid': list(CONFIDENCE_GRID),
            'minimum_accepted': MIN_ACCEPTED, 'minimum_coverage': MIN_COVERAGE,
            'maximum_empirical_error': MAX_EMPIRICAL_ERROR,
            'maximum_ad_auto_cleared_fraction': MAX_AD_AUTOCLEARED_FRACTION},
        'source_code_sha256': ctx['sources'], 'pipeline_report_sha256': ctx['pipeline_sha256'],
        'test_previously_exposed_in_v1': True, 'test_evaluation_authorised_by_this_plan': False,
        'caution': CAUTION}


def prepare(ctx: dict) -> dict:
    proposed = selection(ctx)
    path = ctx['project'] / PLAN_FILE
    if path.exists():
        require(read_json(path) == proposed, 'Existing A1 selection differs. Preserve it; do not replace it.')
        print('A1_SELECTION_ALREADY_RECORDED')
    else:
        write_json(path, proposed)
        print('A1_SELECTION_RECORDED')
    print('Review and commit the three additions plus A1_SELECTED_MODEL.json before run.')
    return proposed


def load_selected_checkpoint(c: Candidate, manifests: dict, device: torch.device,
                             factories: dict | None = None) -> torch.nn.Module:
    """Load only the locally selected, hash-bound checkpoint; never download weights.

    This adapter is reusable by the later revised prediction/evaluation interface.
    Optional factories are a synthetic-test hook, not a CLI option.
    """
    require(sha256(c.checkpoint) == c.hashes['checkpoint_sha256'], 'Selected checkpoint bytes changed.')
    value = torch.load(c.checkpoint, map_location='cpu', weights_only=True)
    require(isinstance(value, dict) and value.get('epoch') == c.report['selected_epoch'] and
            value.get('manifest_sha256') == manifests and value.get('model_label_map') == {'NC': 0, 'AD': 1},
            'Checkpoint epoch, labels or patient manifests differ.')
    require(value.get('configuration') == c.report['configuration'] and
            value.get('source_code_sha256') == c.report['source_code_sha256'],
            'Checkpoint configuration/source provenance differs from the selected run.')
    if c.alias == 'a1':
        require(value.get('format_version') == 2 and value.get('architecture') == 'ConvNeXtTiny' and
                value.get('pretrained') == {'used': False}, 'Not the selected scratch-trained A1 checkpoint.')
    else:
        require(value.get('architecture') == 'BaselineCNN-v1' and value.get('model_name') == 'baseline' and
                value.get('preprocessing') == preprocessing_spec(0.0), 'Not the selected baseline checkpoint.')
    factory = (factories or {'a1': ConvNeXtTiny, 'cnn': BaselineCNN})[c.alias]
    model = factory()
    require(parameter_count(model) == c.report['parameters'], 'Constructed model parameter count differs.')
    # Check exact tensor shapes before PyTorch's legacy scalar conversion.
    # PyTorch 2.7.1 may take element zero of a 1-D checkpoint tensor when
    # the model expects a scalar, even with strict=True. These checkpoints
    # are from this exact project, so no legacy shape conversion is allowed.
    state = value.get('model_state')
    if not isinstance(state, dict):
        raise RuntimeError('Checkpoint model_state must be a tensor dictionary.')
    expected = model.state_dict()
    if set(state) != set(expected):
        raise RuntimeError('Checkpoint parameter/buffer keys do not match the selected model.')
    for name, target in expected.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f'Checkpoint entry {name} must be a torch.Tensor.')
        if tensor.shape != target.shape:
            raise RuntimeError(
                f'Checkpoint tensor shape mismatch for {name}: '
                f'expected {tuple(target.shape)}, received {tuple(tensor.shape)}.')
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def replay_checks(records, original: list[float], replayed: list[float], tolerance: float) -> dict:
    require(len(original) == len(replayed) == len(records) > 0, 'Validation replay is incomplete.')
    a, b = np.asarray(original), np.asarray(replayed)
    require(np.isfinite(b).all() and ((b >= 0) & (b <= 1)).all(), 'Replay contains invalid probabilities.')
    max_error = float(np.max(np.abs(a - b)))
    require(max_error <= tolerance, 'Checkpoint replay differs from saved probabilities beyond tolerance.')
    metrics_a, grouped_a = aggregate_predictions(records, original)
    metrics_b, grouped_b = aggregate_predictions(records, replayed)
    counts = {}
    for level, key in [('slice', 'slices'), ('scan', 'scans'), ('patient', 'patients')]:
        p = np.array([r['p_ad'] for r in grouped_a[key]])
        q = np.array([r['p_ad'] for r in grouped_b[key]])
        require(np.array_equal(p >= 0.5, q >= 0.5), 'Replay changed a stored class decision; preserve evidence for review.')
        counts[level] = len(p)
    return {'passed': True, 'maximum_absolute_probability_difference': max_error,
        'absolute_tolerance': tolerance, 'class_decisions_match_at_all_levels': True,
        'counts': counts, 'replayed_metrics': metrics_b,
        'note': 'Small floating-point differences are allowed; calibration uses the original verified saved probabilities.'}


def inference(model, dataset, device) -> tuple[list[float], float]:
    import time
    require(dataset.split == 'validation', 'This integration only accepts validation images.')
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(42))
    paths, probabilities = [], []
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            require(set(batch['split']) == {'validation'}, 'Unexpected split reached validation replay.')
            logits = model(batch['image'].to(device))
            require(bool(torch.isfinite(logits).all()), 'Nonfinite replay logits.')
            probabilities.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
            paths.extend(batch['image_path'])
            if (i + 1) % 200 == 0:
                print(f'  Replayed {len(probabilities)}/{len(dataset)} validation slices', flush=True)
    synchronize(device)
    require(paths == [r.image_path for r in dataset.records], 'Replay order/coverage differs from validation manifest.')
    return probabilities, time.perf_counter() - start


def verify_candidate(c: Candidate, bundle, device, folder: Path, binding: dict,
                     factories: dict | None = None, benchmark_repeats: int = 50) -> dict:
    receipt_path = folder / f'{c.alias}_verification.json'
    probs_path = folder / f'PRIVATE_{c.alias}_replay.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require(receipt.get('binding') == binding and receipt.get('passed') is True and
                probs_path.is_file() and sha256(probs_path) == receipt.get('replay_file_sha256'),
                'Saved replay receipt or private cache differs; preserve the output folder.')
        p = read_json(probs_path)['probabilities']
        replay_checks(bundle.records['validation'], c.slice_probabilities, p, 2e-5)
        print(f'{c.alias}: reusing previously verified validation replay', flush=True)
        return receipt
    reset_memory_peak(device)
    model = load_selected_checkpoint(c, bundle.manifest_hashes, device, factories)
    dataset = ADNISliceDataset(bundle, 'validation')
    probabilities, seconds = inference(model, dataset, device)
    check = replay_checks(dataset.records, c.slice_probabilities, probabilities, 2e-5)
    bench = inference_benchmark(model, dataset[0]['image'].unsqueeze(0), device, benchmark_repeats)
    receipt = {**check, 'binding': binding, 'model_name': c.name,
        'checkpoint_sha256': c.hashes['checkpoint_sha256'], 'selected_epoch': c.report['selected_epoch'],
        'parameters': parameter_count(model), 'validation_seconds': seconds,
        'model_only_inference': bench, 'gpu_memory': memory_info(device), 'hardware': hardware_info(device),
        'test_images_decoded': 0, 'optimizer_created': False, 'model_weights_updated': False}
    write_json(probs_path, {'probabilities': probabilities})
    receipt['replay_file_sha256'] = sha256(probs_path)
    write_json(receipt_path, receipt)
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(f'{c.alias}: CHECKPOINT_VALIDATION_REPLAY_PASSED', flush=True)
    return receipt


def analyse(ctx: dict, receipts: dict) -> tuple[dict, dict, dict]:
    report = {'format_version': 1, 'a1_confidence_completed': False, 'created_utc': now(),
        'scope': 'Selected A1 and retained CNN: validation replay, calibration and referral only',
        'new_training_performed': False, 'validation_inference_performed': True,
        'new_final_test_model_evaluation_performed': False, 'final_test_images_decoded': 0,
        'test_manifests_read_for_integrity_only': True, 'test_previously_exposed_in_v1': True,
        'old_run_files_and_pointers_modified': False, 'models': {},
        'manifest_sha256': ctx['bundle'].manifest_hashes, 'source_code_sha256': ctx['sources'],
        'candidate_selection_changed_by_calibration': False, 'caution': CAUTION}
    rules = {'format_version': 1, 'status': 'VALIDATION_FITTED_RULES_NOT_FINAL_TEST_RESULTS',
        'models': {}, 'manifest_sha256': ctx['bundle'].manifest_hashes,
        'preprocessing': preprocessing_spec(0.0), 'patient_probability_aggregation': 'mean of scan means of slice P(AD)',
        'input_probability_clip': [1e-7, 1 - 1e-7], 'clinical_validation': False, 'caution': CAUTION}
    private = {}
    for alias, c in ctx['candidates'].items():
        print(f'Fitting validation confidence procedure: {c.name}', flush=True)
        result, rows = analyse_patients(c.subjects, c.labels, c.p)
        result['caution'] = result['crossfit_diagnostic']['caution'] = CAUTION
        result.update(name=c.name, selected_epoch=c.report['selected_epoch'],
                      checkpoint_sha256=c.hashes['checkpoint_sha256'], input_hashes=c.hashes,
                      original_validation_metrics=c.report['selected_validation_metrics'],
                      checkpoint_replay=receipts[alias])
        if alias == 'cnn':
            old = ctx['old_confidence']['models']['cnn']
            same_numbers(result['full_validation_temperature_fit'], old['full_validation_temperature_fit'], 'unchanged CNN temperature')
            same_numbers(result['operating_rule'], old['operating_rule'], 'unchanged CNN referral rule')
            result['previous_cnn_rules_reproduced'] = True
        report['models'][alias] = result
        private[alias] = rows
        rules['models'][alias] = {'name': c.name, 'selected_epoch': c.report['selected_epoch'],
            'checkpoint_sha256': c.hashes['checkpoint_sha256'], 'configuration': c.report['configuration'],
            'temperature': result['full_validation_temperature_fit']['temperature'],
            'diagnostic_threshold': 0.5, 'confidence_threshold': result['operating_rule']['confidence_threshold'],
            'refer_all': result['operating_rule']['confidence_threshold'] is None,
            'fit_partition': 'validation', 'fit_patients': len(c.subjects),
            'architecture': 'ConvNeXtTiny' if alias == 'a1' else 'BaselineCNN-v1'}
    return report, rules, private


def plots(folder: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for alias, filename in [('cnn', PLOTS[0]), ('a1', PLOTS[1])]:
        report = summary['models'][alias]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
        for scores, name in [(report['raw_probability_scores'], 'Raw'),
                            (report['crossfit_diagnostic']['pooled_probability_scores'], 'Cross-fitted temperature')]:
            bins = [b for b in scores['reliability_bins'] if b['n']]
            ax.plot([b['mean_predicted_ad_probability'] for b in bins],
                    [b['observed_ad_fraction'] for b in bins], marker='o', label=name)
        ax.set(xlabel='Mean predicted P(AD)', ylabel='Observed AD fraction', xlim=(0, 1), ylim=(0, 1),
               title=report['name'] + '\nSelected validation development results')
        ax.legend(); fig.tight_layout(); fig.savefig(folder / filename, dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in summary['models'].values():
        rows = sorted([r for r in m['raw_risk_coverage_curve'] if r['n_accepted']], key=lambda r:r['coverage'])
        ax.plot([r['coverage'] for r in rows], [r['accepted_error_rate'] for r in rows], label=m['name'])
    ax.set(xlabel='Accepted fraction (coverage)', ylabel='Accepted error fraction', xlim=(0, 1), ylim=(0, 1),
           title='Validation confidence ranking; not independent test performance')
    ax.legend(); fig.tight_layout(); fig.savefig(folder / PLOTS[2], dpi=160); plt.close(fig)


def fmt(value, percent=False):
    return 'Not estimable' if value is None else (f'{value:.2%}' if percent else f'{value:.4f}')


def markdown(summary: dict) -> str:
    lines = ['## Revised selected-model confidence analysis', '', CAUTION, '',
        'The selected main model is A1 (scratch_augmented, seed 42, epoch 13). S1 and S2 did not '
        'meet the declared promotion criteria; they remain documented negative experiments. '
        'No pretrained weights, threshold change, new model training or test inference were used here.', '',
        'Both saved checkpoints were reconstructed and replayed on all finalized validation slices. '
        'Per-slice probabilities matched within the declared 2e-5 absolute tolerance; class decisions '
        'at slice, scan and patient levels were unchanged. Confidence fitting used the original '
        'verified saved patient probabilities. Patient probabilities are means of scan means.', '',
        '| Model | Patient accuracy | Macro-F1 | AUROC | AD recall |',
        '|---|---:|---:|---:|---:|']
    for m in summary['models'].values():
        a = m['original_validation_metrics']['patient']
        lines.append(f"| {m['name']} | {a['accuracy']:.2%} | {a['macro_f1']:.4f} | {a['auroc']:.4f} | {a['per_class']['AD']['recall']:.2%} |")
    lines += ['', '### Probability quality and confident errors', '',
        'Temperature scaling is applied to clipped patient log-odds after aggregation, with T in '
        '[0.5,10]. It does not change the 0.5 class decision. Five folds use exactly the previous '
        'subject-hash/stratification rule. Cross-fitting does not undo repeated checkpoint/model selection.', '',
        '| Model | Raw NLL | Cross-fit NLL | Raw Brier | Cross-fit Brier | Errors at raw confidence ≥90% | Full-fit T |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m in summary['models'].values():
        r=m['raw_probability_scores']; c=m['crossfit_diagnostic']['pooled_probability_scores']
        lines.append(f"| {m['name']} | {r['nll']:.4f} | {c['nll']:.4f} | {r['brier']:.4f} | {c['brier']:.4f} | {m['raw_confidence_diagnostics']['n_wrong_high_confidence']} | {m['full_validation_temperature_fit']['temperature']:.4f} |")
    lines += ['', '### Empirical referral proposal', '',
        'The unchanged finite grid maximizes coverage subject to accepting at least max(20,ceil(0.25*N)) '
        'fitting patients, at most 10% observed accepted error, and at most 10% of all AD patients '
        'automatically cleared as NC. If no cutoff qualifies, refer all. These are research constraints, '
        'not clinically validated safety limits. Referrals are not correct diagnoses.', '',
        '| Model | Full-fit cutoff | Full-fit coverage | Full-fit accepted error | Cross-fit coverage | Cross-fit accepted error |',
        '|---|---:|---:|---:|---:|---:|']
    for m in summary['models'].values():
        rule=m['operating_rule']; a=rule['development_fit_triage']; b=m['crossfit_diagnostic']['pooled_triage']
        tau='Refer all' if rule['confidence_threshold'] is None else str(rule['confidence_threshold'])
        lines.append(f"| {m['name']} | {tau} | {a['coverage']:.2%} | {fmt(a['accepted_error_rate'],True)} | {b['coverage']:.2%} | {fmt(b['accepted_error_rate'],True)} |")
    lines += ['', 'Full-fit figures are in-sample. Cross-fitted figures remain development diagnostics. '
        'Accepted-subset accuracy is not all-patient accuracy and cannot satisfy a test-accuracy target.', '',
        '### Prediction integration and resources', '',
        'selected_a1.load_selected_checkpoint supports the selected V2 ConvNeXt and retained CNN '
        'formats and verifies checkpoint/manifest/configuration provenance before strict loading. '
        'The historical predict.py and final evaluation plan are intentionally unchanged in this stage. '
        'They must not be rerun expecting automatic selection of A1.', '',
        '| Model | Parameters | Replay seconds | Peak allocated GiB | Batch-1 median ms/slice |',
        '|---|---:|---:|---:|---:|']
    for m in summary['models'].values():
        r=m['checkpoint_replay'];mem=r['gpu_memory']['peak_allocated_gib']
        lines.append(f"| {m['name']} | {r['parameters']:,} | {r['validation_seconds']:.2f} | {fmt(mem)} | {r['model_only_inference']['median_ms_per_slice']:.3f} |")
    lines += ['', 'Replay timing includes validation loading/preprocessing/forward inference, not model loading '
        'or report generation. Batch-1 timing uses a preloaded tensor, 10 warmup and 50 measured forwards; '
        'it is not patient/end-to-end latency. GPU allocator peaks are not total Windows VRAM.', '',
        '### Status and assistance', '',
        'A1 has no revised test result yet. The Version 1 test cohort has already been exposed; any '
        'later reuse must be disclosed. The all-patient 0.80 test requirement remains unresolved. '
        'The new temperatures/rules are tied to the exact selected checkpoint hashes for later review.', '',
        'ChatGPT assisted with this integration, tests and report draft. Record the code review, local '
        'checks and interpretation personally performed in Artificial Intelligence Usage Disclosure. '
        'Temperature-scaling reference: Guo et al., On Calibration of Modern Neural Networks, ICML 2017 '
        '(https://arxiv.org/abs/1706.04599).']
    for name in PLOTS:
        lines.extend(['', f'![Selected-model validation diagnostic](figures/{name})'])
    return '\n'.join(lines)+'\n'


def finish_bundle(folder: Path) -> None:
    temporary = folder / (BUNDLE + '.tmp')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as z:
        for name in PUBLIC:
            require((folder / name).is_file(), 'A required aggregate result is missing.')
            z.write(folder / name, name)
    temporary.replace(folder / BUNDLE)


def validate_completed(folder: Path, plan: dict) -> dict:
    report = read_json(folder / SUMMARY)
    require(report.get('a1_confidence_completed') is True and report.get('new_final_test_model_evaluation_performed') is False,
            'No completed validation-only analysis.')
    require(read_json(folder / PLAN_FILE) == plan, 'Completed result uses a different selection.')
    for name, digest in report.get('aggregate_file_sha256', {}).items():
        require(name in PUBLIC and (folder / name).is_file() and sha256(folder / name) == digest,
                'Completed aggregate result has changed; preserve it for review.')
    require(set(report.get('aggregate_file_sha256',{})) == set(PUBLIC)-{SUMMARY},
            'Completed output manifest is incomplete.')
    return report


def run(ctx: dict, device_name: str, *, factories: dict | None = None,
        benchmark_repeats: int = 50, enforce_git: bool = True) -> Path:
    plan = selection(ctx)
    require((ctx['project']/PLAN_FILE).is_file() and read_json(ctx['project']/PLAN_FILE) == plan,
            'Run prepare and review/commit the exact selected-model plan first.')
    if enforce_git:
        require_committed(ctx['project'])
    seed_everything(42)
    device = choose_device(device_name)
    folder = ctx['private'] / FOLDER
    folder.mkdir(exist_ok=True)
    binding = {'plan_sha256': sha256(ctx['project']/PLAN_FILE), 'environment': environment(device)}
    with run_lock(folder):
        if (folder / SUMMARY).is_file():
            validate_completed(folder, plan)
            finish_bundle(folder)
            (ctx['private']/POINTER).write_text(str(folder)+'\n',encoding='utf-8')
            print('A1_CONFIDENCE_ALREADY_COMPLETED'); return folder
        state = folder / 'run_binding.json'
        if state.exists():
            require(read_json(state) == binding, 'Replay environment or plan changed since the interrupted analysis.')
        else:
            write_json(state, binding)
        write_json(folder / PLAN_FILE, plan)
        receipts = {}
        for alias, c in ctx['candidates'].items():
            receipts[alias] = verify_candidate(c, ctx['bundle'], device, folder, binding, factories, benchmark_repeats)
        report, rules, individual = analyse(ctx, receipts)
        # Confirm bound source/evidence bytes have not changed during this run.
        check_sources(ctx['project'], ctx['sources'])
        for c in ctx['candidates'].values():
            require(sha256(c.checkpoint) == c.hashes['checkpoint_sha256'] and
                    sha256(c.prediction_file) == c.hashes['validation_predictions_sha256'] and
                    sha256(c.folder/c.report_name) == c.hashes['run_summary_sha256'],
                    'Selected run evidence changed during analysis.')
        rules['selection_plan_sha256'] = binding['plan_sha256']
        report['selection_plan_sha256'] = binding['plan_sha256']
        write_json(folder/'PRIVATE_patient_diagnostics.json',individual)
        write_json(folder/RULES,rules)
        plots(folder, report)
        (folder/'README_a1_confidence.md').write_text(markdown(report), encoding='utf-8')
        report['a1_confidence_completed'] = True
        report['status'] = 'COMPLETED_VALIDATION_ONLY'
        report['aggregate_file_sha256'] = {name:sha256(folder/name) for name in PUBLIC if name != SUMMARY}
        write_json(folder/SUMMARY,report)
        finish_bundle(folder)
        tmp = ctx['private']/(POINTER+'.tmp')
        tmp.write_text(str(folder)+'\n',encoding='utf-8'); tmp.replace(ctx['private']/POINTER)
    print('A1_CONFIDENCE_COMPLETED')
    print('VALIDATION_ONLY_NO_NEW_TEST_EVALUATION')
    print(folder / BUNDLE)
    return folder


def publish(ctx: dict) -> None:
    folder = ctx['private']/FOLDER
    validate_completed(folder, selection(ctx))
    project=ctx['project'];readme=project/'README.md'
    content=readme.read_bytes().decode('utf-8-sig')
    if MARKER in content:
        print('A1_CONFIDENCE_REPORT_ALREADY_PRESENT'); return
    backup=ctx['private']/('a1-confidence-readme-backup-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    backup.mkdir()
    (backup/'README.md').write_bytes(readme.read_bytes())
    figure_dir=project/'figures';figure_dir.mkdir(exist_ok=True)
    for name in PLOTS:
        target=figure_dir/name
        if target.exists():
            (backup/name).write_bytes(target.read_bytes())
        target.write_bytes((folder/name).read_bytes())
    section=(folder/'README_a1_confidence.md').read_text(encoding='utf-8')
    temporary=readme.with_suffix('.md.tmp')
    temporary.write_bytes((content+'\n\n'+MARKER+'\n'+section).encode('utf-8'))
    temporary.replace(readme)
    print('A1_CONFIDENCE_REPORT_APPENDED')


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','run','publish'))
    parser.add_argument('--private-root', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    args=parser.parse_args()
    try:
        ctx=context(args.private_root)
        if args.command=='prepare':prepare(ctx)
        elif args.command=='run':run(ctx,args.device)
        else:publish(ctx)
    except (Exception, KeyboardInterrupt) as exc:
        out=args.private_root.resolve()/FOLDER
        if out.is_dir():
            (out/'PRIVATE_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print(f'A1_CONFIDENCE_STOPPED: {type(exc).__name__}: {exc}',file=sys.stderr)
        print('Preserve results. Do not retrain, change old pointers, or run the historical final-test command.',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
