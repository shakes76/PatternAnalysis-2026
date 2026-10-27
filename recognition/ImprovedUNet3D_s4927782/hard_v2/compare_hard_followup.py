"""Three complete pipelines on unchanged v1 GT; no missing-new-2D fiction.

The original new-pair stopped at its TRAIN QA gate. A separately recorded,
Hard-only follow-up reuses the frozen original 2D/3D references. Pixel metrics
are remeasured on one common coarse grid; scope, training selection and runtime
receipts are audited before any TEST pixel is opened. No model selection occurs
here. The already-inspected TEST remains retrospective.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

import compare_common_grid as common
import summarize_v2 as audit

HERE = Path(__file__).resolve().parent
MODEL_KEYS = ('v1_unet2d', 'v1_improved3d', 'hard_improved3d')
CLASS_NAMES = ('背景', '身体', '骨骼', '膀胱', '直肠', '前列腺')
BASE_PROTOCOL_SHA256 = '9b415ccaf415b6299b5f86691e0a6576ea9028ebe52896d98bbf6ffdb0d20035'
V1_MANIFEST_SHA256 = 'c47fe880d353629348786b852e3f699291af1a4d5628b2980bf42a74635d0424'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return common.read_json(path)


def sha(path):
    return common.file_sha256(path)


def bound_file(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path)}


def verify_scope(scope_path, review_root):
    scope = load(scope_path)
    require(scope.get('base_protocol_sha256') == BASE_PROTOCOL_SHA256,
            'Hard scope must explicitly inherit the unchanged base scientific configuration.')
    require(scope.get('experiment_id') and scope['experiment_id'] != 'independent_resolution_sampling_v2',
            'The stopped pair and follow-up must retain different experiment IDs.')
    require(str(scope.get('status', '')).startswith('frozen'), 'Separate Hard scope was not frozen before execution.')
    base_path = Path(review_root) / 'protocol_v2.json'
    require(sha(base_path) == BASE_PROTOCOL_SHA256, 'Base protocol bytes changed.')
    base = load(base_path)
    for relative, expected in base['required_review_sha256'].items():
        require(sha(common.resolve_relative(review_root, relative)) == expected,
                'A previously frozen source changed: ' + relative)
    for relative, expected in scope.get('required_scope_sha256', {}).items():
        require(sha(common.resolve_relative(review_root, relative)) == expected,
                'Hard scope source changed: ' + relative)
    if 'configuration' in scope:
        require(scope['configuration'] == base['configuration'], 'Hard scientific configuration changed.')
    require(scope.get('formal_models') == ['improved3d']
            and scope.get('stopped_pair_status') == 'stopped_after_QA_failure'
            and scope.get('scope_change_is_scientific_plan_revision_not_engineering_fix') is True,
            'Do not disguise a changed scope as a successful original pair or engineering-only correction.')
    require(scope.get('configuration_exactly_inherited') is True and scope.get('seed') == 3711
            and scope.get('epochs') == 30 and scope.get('max_gpu_walltime_minutes') == 60
            and scope.get('reused_reference_models') == ['v1_unet2d', 'v1_improved3d']
            and scope.get('no_new_parameters_or_candidates') is True,
            'Only the bounded unchanged Hard candidate and two original references are allowed.')
    require(scope.get('adoption') == base['adoption'], 'Prespecified adoption conditions changed.')
    return scope, base


def verify_failed_pair_qa(artifacts, base):
    """The original failed pair remains failed; no threshold or budget is relaxed."""
    qa_path = Path(artifacts) / 'qa_v2/qa_receipt.json'
    fit_path = Path(artifacts) / 'qa_v2/fit/fit_receipt.json'
    qa, fit = load(qa_path), load(fit_path)
    require(qa['status'] == 'failed_fit_gate' and fit['status'] == 'failed_fit_gate',
            'Original pair failure must remain explicitly recorded.')
    require(qa['protocol_sha256'] == fit['protocol_sha256'] == BASE_PROTOCOL_SHA256,
            'Original QA protocol differs.')
    require(qa['fit_receipt_sha256'] == sha(fit_path), 'Original fit receipt bytes changed.')
    require(qa['models']['unet2d']['passed'] is False and fit['models']['unet2d']['passed'] is False,
            'Do not reinterpret the failed new 2D candidate as passed.')
    gates = base['qa']
    require(fit['steps'] == gates['fit_steps'] == 400 and fit['case'] == gates['fit_case']
            and fit['amp'] is False and fit['seed'] == gates['fit_seed'], 'Original fit case/budget/precision changed.')
    hard = fit['models']['improved3d']
    first, last = hard['history'][0], hard['history'][-1]
    require(first['step'] == 0 and last['step'] == 400 and first['loss'] > 0,
            'Hard fit must use the fixed final400 step, not its best observed step.')
    require(hard['passed'] is True and qa['models']['improved3d']['passed'] is True
            and last['prostate_dice'] >= gates['minimum_fit_prostate_dice']
            and last['foreground_dice'] >= gates['minimum_fit_foreground_dice']
            and last['loss'] / first['loss'] <= gates['maximum_final_initial_loss_ratio'],
            'Hard candidate did not pass the unchanged original TRAIN gate.')
    for flag in ('full_image_only_inference_passed', 'native_geometry_serialisation_passed',
                 'independent_metric_check_passed', 'sampler_batch_backward_passed'):
        require(qa['models']['improved3d'][flag] is True, 'Hard pipeline QA incomplete: ' + flag)
    return {'status': 'stopped_after_QA_failure', 'qa': bound_file(qa_path), 'fit': bound_file(fit_path),
            'unet2d': fit['models']['unet2d'], 'improved3d': hard,
            'fit_exposure_note': 'The fixed QA used16slices/update for2D and the whole64slice crop/update for3D:100versus400 full-crop exposures. This was an existing design choice, not a newly reproduced bug or proof of why2D missed its gate; these TRAIN scores do not compare generalization.',
            'interpretation': 'Only TRAIN QA led to scope reduction. No extended fit, lowered gate, retrained new2D, QA warm start or new-model VAL/TEST score was used.'}


def verify_selection(freeze, scope, scope_sha, manifest_sha, run_root):
    require(freeze.get('status') == 'frozen' and freeze.get('experiment_id') == scope['experiment_id'],
            'Selected-checkpoint record is not the separate Hard scope.')
    require(freeze.get('scope_plan_sha256') == scope_sha and freeze.get('manifest_sha256') == manifest_sha,
            'Hard selection record belongs to another scope/data manifest.')
    require(freeze.get('base_protocol_sha256') == BASE_PROTOCOL_SHA256,
            'Selected model changed the inherited scientific configuration.')
    require(set(freeze.get('models', {})) == {'improved3d'}, 'Do not fabricate a missing new2D checkpoint.')
    item = freeze['models']['improved3d']
    require(item['completed_epochs'] == 30 and item['checkpoint_sha256'] == sha(Path(run_root) / 'best.pt'),
            'Hard selection must bind the completed30-epoch chosen checkpoint.')


def verify_scope_execution(path, scope_sha, review_root, chosen_sha):
    receipt = load(path)
    require(receipt.get('status') == 'completed' and receipt.get('scope_plan_sha256') == scope_sha,
            'Actual Hard-only training execution is not complete/bound.')
    expected_executor = sha(Path(review_root) / 'train_hard_followup.py')
    require(receipt.get('training_executor_sha256') == expected_executor,
            'Actual Hard training executor differs from recorded source.')
    driver_sha = sha(Path(review_root) / 'hard_followup_gpu.slurm')
    require(receipt.get('scope_driver_sha256', receipt.get('driver_sha256')) == driver_sha,
            'Actual Hard driver differs from recorded source.')
    require(receipt.get('best_checkpoint_sha256') == chosen_sha, 'Scope receipt selected weight mismatch.')
    return receipt


def verify_runtime(path, evaluation_path, scope_sha, manifest_sha, chosen_sha, review_root, counters):
    runtime = load(path)
    evaluation = load(evaluation_path)
    require(runtime.get('status') == 'completed' and runtime.get('scope_plan_sha256') == scope_sha,
            'Inference runtime is not explicitly bound to the separate Hard scope.')
    require(runtime.get('base_protocol_sha256') == runtime.get('protocol_sha256') == BASE_PROTOCOL_SHA256,
            'Inference changed base protocol.')
    require(runtime.get('manifest_sha256') == manifest_sha and runtime.get('checkpoint_sha256') == chosen_sha,
            'Inference runtime data/checkpoint mismatch.')
    require(runtime.get('evaluation_sha256') == sha(evaluation_path), 'Evaluation runtime byte binding differs.')
    metrics_path = Path(evaluation_path).with_name('metrics.json')
    require(runtime.get('metrics_sha256') == sha(metrics_path) and load(metrics_path) == evaluation,
            'Metrics and evaluation must preserve the same original measurements.')
    require(runtime.get('adapter_file') == 'train_hard_followup.py'
            and runtime.get('adapter_sha256') == sha(Path(review_root) / 'train_hard_followup.py'),
            'Actual Hard evaluation executor differs.')
    require(runtime.get('scope_driver_sha256') == sha(Path(review_root) / 'hard_followup_gpu.slurm'),
            'Actual Hard evaluation driver differs.')
    require(runtime.get('device') == 'cuda' and runtime.get('gpu_name') and runtime.get('amp') is True
            and runtime.get('seed') == 3711 and runtime.get('cudnn_deterministic') is True
            and runtime.get('cudnn_benchmark') is False
            and runtime.get('deterministic_algorithms_enabled') is True
            and runtime.get('deterministic_algorithms_warn_only') is True,
            'Actual Hard inference runtime/precision/seed/settings differ.')
    for key in ('python_version', 'torch_version', 'cuda_version', 'numpy_version', 'scipy_version',
                'nibabel_version', 'slurm_job_id', 'slurm_node_list'):
        require(runtime.get(key), 'Missing actual inference runtime field: ' + key)
    counters['runtime_receipts'] += 1
    return runtime


def verify_stored_case(actual, stored, where):
    for a, b in zip(actual['per_class'], stored['per_class']):
        for field in ('class_index', 'target_voxels', 'predicted_voxels', 'intersection', 'union'):
            require(a[field] == b[field], f'{where}: recomputed preserved count differs at {field}')
        for field in ('dice', 'iou'):
            audit.equal(a[field], b[field], where + '/' + field)


def confusion_details(prediction, target):
    matrix = np.bincount(target.ravel().astype(np.int64) * 6 + prediction.ravel().astype(np.int64),
                         minlength=36).reshape(6, 6)
    return {'confusion_matrix_gt_rows_prediction_columns': matrix.tolist(),
            'prostate_fp_by_gt_class0to4': matrix[:5, 5].tolist(),
            'prostate_fn_by_predicted_class0to4': matrix[5, :5].tolist()}


def compare(args):
    start = time.perf_counter()
    root, artifacts, project = Path(args.review_root).resolve(), Path(args.artifacts_root).resolve(), Path(args.v1_project).resolve()
    scope_path = Path(args.scope_plan).resolve()
    scope, base = verify_scope(scope_path, root)
    scope_sha = sha(scope_path)
    analysis_freeze_path = root / 'analysis_source_freeze.json'
    analysis_freeze = None
    if analysis_freeze_path.exists():
        analysis_freeze = load(analysis_freeze_path)
        for relative, expected in analysis_freeze.get('sha256', {}).items():
            require(sha(common.resolve_relative(root, relative)) == expected,
                    'Analysis-only source changed after its separate freeze: ' + relative)
    if args.split == 'test':
        require(args.retrospective_test_ack, 'Known TEST requires explicit retrospective acknowledgement before pixels.')
    qa = verify_failed_pair_qa(artifacts, base)
    require(scope.get('qa_receipt_sha256') == qa['qa']['sha256']
            and scope.get('fit_receipt_sha256') == qa['fit']['sha256'],
            'Scope must bind the preserved failed-pair QA, not newly edited gates/results.')
    old_path, new_path = project / 'artifacts/cache64/manifest.json', artifacts / 'cache_v2/manifest.json'
    require(sha(old_path) == V1_MANIFEST_SHA256, 'Original common-GT manifest changed.')
    old, new = load(old_path), load(new_path)
    require(scope.get('manifest_sha256') == scope.get('base_manifest_sha256') == sha(new_path),
            'Separate scope must retain precisely the inherited data cache manifest.')
    require(old['patient_splits'] == new['patient_splits'] and old['label_ids'] == new['label_ids'] == list(range(6)),
            'Patient split and label semantics must stay unchanged.')
    require(new['protocol_sha256'] == BASE_PROTOCOL_SHA256 and new['original_manifest_sha256'] == sha(old_path),
            'Inherited cache identity differs.')
    for manifest in (old, new):
        require({s: len(manifest['patient_splits'][s]) for s in ('train', 'validation', 'test')} ==
                {'train': 26, 'validation': 6, 'test': 6}, 'Original patient populations changed.')
        require({s: sum(r['split'] == s for r in manifest['records']) for s in ('train', 'validation', 'test')} ==
                {'train': 143, 'validation': 38, 'test': 30}, 'Original scan populations changed.')
    old_records, new_records = common.index_records(old), common.index_records(new)
    keys = sorted(key for key, record in old_records.items() if record['split'] == args.split)
    require(set(keys) == {key for key, record in new_records.items() if record['split'] == args.split},
            'All complete pipelines must cover the same patients/visits.')
    for relative, expected in base['required_original_sha256'].items():
        require(sha(common.resolve_relative(project, relative)) == expected, 'A frozen original input changed: ' + relative)
    identity_path = old_path.parent / 'cache_identity.json'
    require(sha(identity_path) == common.FROZEN_V1_CACHE_IDENTITY_SHA256, 'Original GT identity book changed.')
    gt_book = load(identity_path)
    require(gt_book['manifest_sha256'] == sha(old_path), 'Original identity book/manifest differ.')
    counters = Counter()
    freeze_path = artifacts / 'hard_selection.json'
    freeze = load(freeze_path)
    run_root = artifacts / 'hard_followup/improved3d'
    # This checks completion before reading unfinished history or pixel data.
    require(load(run_root / 'run.json').get('status') == 'completed', 'Hard run is not complete; do not inspect partial results.')
    verify_selection(freeze, scope, scope_sha, sha(new_path), run_root)
    execution_path = artifacts / 'hard_followup/scope_execution.json'
    execution = verify_scope_execution(execution_path, scope_sha, root, freeze['models']['improved3d']['checkpoint_sha256'])
    require(freeze.get('scope_execution_sha256') == sha(execution_path),
            'Selected-checkpoint record is not bound to actual Hard-only execution.')
    runs = {
        'v1_unet2d': audit.audit_run(project / 'artifacts/full_seed42/unet2d', 'v1', 'unet2d', base, old, counters),
        'v1_improved3d': audit.audit_run(project / 'artifacts/full_seed42/improved3d', 'v1', 'improved3d', base, old, counters),
        'hard_improved3d': audit.audit_run(run_root, 'v2', 'improved3d', base, new, counters, freeze),
    }
    hard_identity = runs['hard_improved3d']['run']['identity']
    require(hard_identity.get('model') == 'improved3d'
            and hard_identity.get('manifest_sha256') == sha(new_path)
            and hard_identity.get('protocol_sha256') == BASE_PROTOCOL_SHA256
            and hard_identity.get('configuration') == base['configuration']
            and hard_identity.get('source_hashes') == base['required_original_sha256']
            and hard_identity.get('v2_pipeline_sha256') == base['required_review_sha256']['v2_pipeline.py'],
            'Hard training must bind all inherited model/data/algorithm/source identities.')
    paths = {
        'v1_unet2d': project / ('artifacts/test_final' if args.split == 'test' else 'artifacts/validation_seed42') / 'unet2d/evaluation.json',
        'v1_improved3d': project / ('artifacts/test_final' if args.split == 'test' else 'artifacts/validation_seed42') / 'improved3d/evaluation.json',
        'hard_improved3d': artifacts / ('hard_test' if args.split == 'test' else 'hard_validation') / 'improved3d/evaluation.json',
    }
    evals, indexed = {}, {}
    for name, path in paths.items():
        family = 'unet2d' if name == 'v1_unet2d' else 'improved3d'
        manifest_path, manifest = (new_path, new) if name == 'hard_improved3d' else (old_path, old)
        evaluation, cases = common.validate_evaluation(path, manifest_path, manifest, args.split, keys, family)
        expected = freeze['models']['improved3d']['checkpoint_sha256'] if name == 'hard_improved3d' else common.FROZEN_V1_CHECKPOINTS[family]
        require(evaluation['checkpoint_sha256'] == expected, 'Wrong selected checkpoint: ' + name)
        if name == 'hard_improved3d':
            require(evaluation['identity'] == hard_identity,
                    'Hard evaluation inherited configuration differs.')
            require(evaluation.get('scope_plan_sha256') == scope_sha
                    and evaluation.get('actual_scope_id') == scope['experiment_id'],
                    'Hard measurements must explicitly identify the separate scope, not the stopped pair.')
            if args.split == 'test':
                require(evaluation.get('selection_freeze_sha256') == sha(freeze_path),
                        'Known TEST measurement is not bound to the chosen Hard checkpoint freeze.')
        evals[name], indexed[name] = evaluation, cases
    runtime_path = paths['hard_improved3d'].with_name('runtime_receipt.json')
    runtime = verify_runtime(runtime_path, paths['hard_improved3d'], scope_sha, sha(new_path),
                             freeze['models']['improved3d']['checkpoint_sha256'], root, counters)
    # Audit the standalone new-grid measurements before deriving common-grid ones.
    new_split_records = {key: dict(new_records[key]) for key in keys}
    completion_path = new_path.resolve().parent / 'test_completion_receipt.json'
    if args.split == 'test':
        completion = load(completion_path)
        require(completion['status'] == 'completed' and completion['manifest_sha256'] == sha(new_path),
                'TEST cache was not completed for the immutable inherited manifest.')
        require(completion.get('scope_plan_sha256') == scope_sha
                and completion.get('selection_freeze_sha256') == sha(freeze_path)
                and completion.get('protocol_sha256') == BASE_PROTOCOL_SHA256,
                'TEST preparation must follow this actual Hard-only checkpoint freeze.')
        completion_rows = {(r['patient'], r['visit']): r for r in completion['records']}
        require(set(completion_rows) == set(keys), 'TEST completion must cover all30 known scans.')
        for key in keys:
            new_split_records[key]['class_voxel_counts'] = completion_rows[key]['class_voxel_counts']
    audit.audit_evaluation(evals['hard_improved3d'], new_split_records, 'Hard standalone new-grid', counters)
    if args.split == 'validation':
        selected = next(r for r in runs['hard_improved3d']['history'] if r['epoch'] == freeze['models']['improved3d']['checkpoint_epoch'])
        stored = {(r['patient'], r['visit']): r for r in selected['validation']['cases']}
        for key in keys:
            verify_stored_case(indexed['hard_improved3d'][key], stored[key], 'Selected validation checkpoint reload')
    results = {name: [] for name in MODEL_KEYS}
    gt_inputs = []
    for key in keys:
        before, after = old_records[key], new_records[key]
        cache_path = common.resolve_relative(old_path.parent, before['cache_path'])
        with np.load(cache_path, allow_pickle=False) as saved:
            target, spacing = saved['mask'], saved['spacing']
        expected_arrays = gt_book['records'][before['cache_path']]['arrays']
        require(common.decoded_array_sha256(target) == expected_arrays['mask']['sha256']
                and common.decoded_array_sha256(spacing) == expected_arrays['spacing']['sha256'],
                'Actual original GT/spacing differs from the frozen decoded identity.')
        require(target.shape == tuple(old['target_shape_dhw'])
                and np.allclose(spacing, before['spacing_dhw_mm'], rtol=0, atol=1e-6), 'Original common grid/spacing changed.')
        a_old, a_new = common.affine(before), common.affine(after)
        require(np.allclose(np.linalg.norm(a_old[:3, :3], axis=0)[::-1], spacing, rtol=1e-6, atol=1e-6), 'Physical metric spacing mismatch.')
        gram = a_old[:3, :3].T @ a_old[:3, :3]
        require(np.allclose(gram, np.diag(np.diag(gram)), rtol=0, atol=1e-6), 'EDT distance requires an orthogonal common grid.')
        gt_inputs.append({'patient': key[0], 'visit': key[1], **bound_file(cache_path),
                          'decoded_mask_sha256': common.decoded_array_sha256(target),
                          'decoded_spacing_sha256': common.decoded_array_sha256(spacing)})
        for name in MODEL_KEYS:
            source_record, shape = (after, new['target_shape_dhw']) if name == 'hard_improved3d' else (before, old['target_shape_dhw'])
            prediction, evidence = common.load_prediction(paths[name], indexed[name][key], shape, source_record,
                                                          require_affine=name == 'hard_improved3d')
            mapping = None
            if name == 'hard_improved3d':
                prediction, mapping = common.categorical_to_grid(prediction, a_new, a_old, target.shape)
            measured = common.class_metrics(prediction, target)
            if name != 'hard_improved3d':
                verify_stored_case(measured, indexed[name][key], 'Original preserved ' + name)
            case = {'patient': key[0], 'visit': key[1], **measured,
                    'prostate_surface': common.prostate_surface_metrics(prediction, target, spacing),
                    'shape_dhw': list(target.shape), 'spacing_dhw': spacing.tolist(),
                    'prediction_evidence': evidence, **confusion_details(prediction, target)}
            if mapping is not None:
                case['target_common_dhw_to_source_new_dhw'] = mapping.tolist()
            results[name].append(case)
    models = {}
    old_split_records = {key: old_records[key] for key in keys}
    for name in MODEL_KEYS:
        aggregate = common.aggregate(results[name])
        audit.audit_evaluation({'cases': results[name], 'aggregate': aggregate}, old_split_records,
                               name + ' independently remeasured common grid', counters, surface=True)
        resource = dict(runs[name]['resources'])
        resource.update(mean_whole_case_inference_ms=evals[name]['mean_inference_ms'],
                        peak_inference_allocated_cuda_bytes=evals[name]['peak_allocated_cuda_bytes'],
                        latency_protocol=evals[name]['latency_protocol'],
                        inference_runtime=runtime if name == 'hard_improved3d' else {k: evals[name].get(k) for k in
                            ('gpu_name', 'device', 'torch_version', 'python_version', 'cuda_version')})
        models[name] = {'cases': results[name], 'aggregate': aggregate, 'resources': resource,
                        'checkpoint_sha256': evals[name]['checkpoint_sha256'],
                        'selected_epoch': runs[name]['selected_epoch'],
                        'selection_metric': runs[name]['selection_metric'],
                        'prostate_dice_point7_met': aggregate['per_class'][5]['dice'] is not None and aggregate['per_class'][5]['dice'] >= .7,
                        'pooled_prostate_counts_descriptive_only': {
                            'tp': sum(c['per_class'][5]['intersection'] for c in results[name]),
                            'fp': sum(c['per_class'][5]['predicted_voxels'] - c['per_class'][5]['intersection'] for c in results[name]),
                            'fn': sum(c['per_class'][5]['target_voxels'] - c['per_class'][5]['intersection'] for c in results[name])}}
    paired = {name: common.paired_summary(results[name], results['hard_improved3d']) for name in ('v1_unet2d', 'v1_improved3d')}
    for name in paired:
        audit.audit_paired(paired[name], results[name], results['hard_improved3d'], name + ' paired newHard')
    gate = common.comparison_gate(models['v1_improved3d']['aggregate'], models['hard_improved3d']['aggregate'], args.split)
    report = {'status': 'completed', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'split': args.split,
              'scientific_status': 'Separate, scope-reduced exploratory Hard-only follow-up after failed-pair TRAIN QA and previously inspected TEST. No independent blind-test claim.',
              'scope_plan_sha256': scope_sha, 'experiment_id': scope['experiment_id'],
              'original_pair': qa, 'scope_execution': execution,
              'metric_grid': 'Unchanged original64x128x128 GT and per-case physical spacing for all three pipelines; not native-resolution accuracy.',
              'hd95_protocol': 'Pooled bidirectional nearest six-connected voxel-surface distances,95th percentile in mm. Empty surfaces null with explicit statuses; pair comparisons use identical both-present cases.',
              'comparison_scope': 'Three complete pipelines. New resolution/sampling/exposure/seed3711/selector/patch-window context differ from originalseed42; no isolated3D causal attribution.',
              'adoption_gate': gate, 'paired': paired, 'models': models, 'audit_counts': dict(counters),
              'input_files': {'scope_plan': bound_file(scope_path), 'base_protocol': bound_file(root / 'protocol_v2.json'),
                              'v1_manifest': bound_file(old_path), 'new_manifest': bound_file(new_path),
                              'v1_cache_identity': bound_file(identity_path), 'selection': bound_file(freeze_path),
                              'scope_execution': bound_file(execution_path), 'runtime': bound_file(runtime_path),
                              'evaluations': {name: bound_file(path) for name, path in paths.items()}},
              'common_gt_inputs': gt_inputs, 'analysis_seconds_including_pixel_IO_metrics': time.perf_counter() - start,
              'analysis_source_sha256': sha(__file__), 'common_math_source_sha256': sha(common.__file__),
              'independent_audit_source_sha256': sha(audit.__file__),
              'numpy_version': np.__version__, 'scipy_version': common.scipy.__version__,
              'analysis_source_freeze': bound_file(analysis_freeze_path) if analysis_freeze is not None else None,
              'limitations': ['TRAIN crop fitting does not establish generalization. The failed new2D is not trained formally or labelled as passed.',
                              'The all-or-none original pair stopped; changing to a Hard-only study is an acknowledged adaptive scope revision, not its natural continuation or a pure engineering fix.',
                              'The same six VAL patients have been reused; all six TEST patients were already inspected before this study. New scores cannot restore blindness.',
                              'Common grid is coarse. HD95 is conditional on available surfaces; Dice0/FN and excluded surface cases remain explicit.',
                              'P5 Dice>=.70 is a numerical project objective under declared aggregation, not a grade guarantee or validated clinical acceptance criterion.']}
    return report


def n(value, digits=4):
    return '未定义' if value is None else f'{value:.{digits}f}'


def render_markdown(report):
    title = '共同VAL：采用门槛' if report['split'] == 'validation' else '已接触TEST：回顾性比较'
    lines = ['# Hard-only有限跟进：' + title, '', report['scientific_status'], '',
             '原双模型方案因新2D的400步TRAIN拟合未过门槛而终止，记录不改。仅已通过原门槛的新Hard3D进入另立的30轮计划；原2D、原3D保留作冻结对照。没有重训失败的新2D或增加超参组合。', '',
             '原QA中2D每步16切片、3D每步整个64切片crop，400步分别相当于100/400次全crop暴露。这是既定检查设计，不能据此证明2D失败根因或宣称3D泛化更好。', '',
             '| 指标（共同原64网格） | 原2D | 原3D | 新Hard3D |', '|---|---:|---:|---:|']
    models = report['models']
    for ci, name in enumerate(CLASS_NAMES):
        for metric in ('dice', 'iou'):
            lines.append('| ' + name + ' ' + metric + ' | ' + ' | '.join(n(models[k]['aggregate']['per_class'][ci][metric]) for k in MODEL_KEYS) + ' |')
    lines.append('| 前景Dice | ' + ' | '.join(n(models[k]['aggregate']['foreground_mean_dice']) for k in MODEL_KEYS) + ' |')
    lines += ['', '| 患者 | visits | 原2D前列腺Dice | 原3D | 新Hard3D |', '|---|---:|---:|---:|---:|']
    indexed = {k: {p['patient']: p for p in models[k]['aggregate']['patients']} for k in MODEL_KEYS}
    for patient in sorted(indexed['v1_unet2d']):
        lines.append('| ' + patient + f' | {indexed["v1_unet2d"][patient]["num_cases"]} | ' + ' | '.join(n(indexed[k][patient]['per_class'][5]['dice']) for k in MODEL_KEYS) + ' |')
    lines += ['', '| 方案 | 选择epoch | 成功更新/AMP跳步 | 训练循环秒 | 参数 | 训练峰值GiB | 整例推理ms | P5 TP/FP/FN | 空预测状态 |', '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for key in MODEL_KEYS:
        m, r = models[key], models[key]['resources']; counts = m['pooled_prostate_counts_descriptive_only']
        lines.append(f'| {key} | {m["selected_epoch"]} | {r["successful_updates"]}/{r["amp_skipped_updates"]} | {n(r["training_loop_seconds"],2)} | {r["parameters"]} | {n(r["peak_training_allocated_cuda_bytes"]/2**30,3)} | {n(r["mean_whole_case_inference_ms"],2)} | {counts["tp"]}/{counts["fp"]}/{counts["fn"]} | {json.dumps(m["aggregate"]["prostate_empty_status_counts"],ensure_ascii=False)} |')
    lines += ['', '池化TP/FP/FN仅描述错误负担，不替代患者均衡Dice。训练循环排除验证/保存/绘图/准备/排队；新3D滑窗CPU概率累加计入推理时间。GPU allocated bytes不等于设备总占用。', '',
              '| 同病例HD95对照 | 原方案mm | 新Hardmm | 合格病例/排除 | 患者 |', '|---|---:|---:|---:|---:|']
    for key, pair in report['paired'].items():
        h = pair['paired_hd95']
        lines.append(f'| {key} | {n(h["v1_hd95_mm"],2)} | {n(h["v2_hd95_mm"],2)} | {h["eligible_case_pairs"]}/{h["excluded_case_pairs"]} | {h["num_patients"]} |')
    g = report['adoption_gate']
    lines += ['', '预定采用门槛只以共同VAL的新Hard相对原3D前列腺Dice至少+.01、前景Dice至少−.01判断；TEST不选择模型或轮次。',
              f'本划分P5变化 {n(g["prostate_dice_delta"])}，前景变化 {n(g["foreground_dice_delta"])}。' +
              (f'共同VAL门槛：{"通过" if g["replacement_supported_by_validation_gate"] else "未通过"}。' if report['split'] == 'validation' else '此处仅为回顾性描述，不生成采用判断。'), '',
              '## 已见失败', '', '| 方案 | 病例（完整保留P5<.70或未定义） | Dice | FP | FN |', '|---|---|---:|---:|---:|']
    for key in MODEL_KEYS:
        for c in sorted(models[key]['cases'], key=lambda x: (x['per_class'][5]['dice'] is None, x['per_class'][5]['dice'] or 0, x['patient'], x['visit'])):
            p = c['per_class'][5]
            if p['dice'] is None or p['dice'] < .7:
                lines.append(f'| {key} | {c["patient"]}/{c["visit"]} | {n(p["dice"])} | {p["predicted_voxels"]-p["intersection"]} | {p["target_voxels"]-p["intersection"]} |')
    lines += ['', '这些已见失败不能被平均分掩盖；本实验没有建立在无GT的新患者中自动检测失败的可靠阈值。HD95与标签末端可视化不等于真实器官壁距离或独立解剖apex/base标注。', '',
              '## 方法限制', '', *['- ' + item for item in report['limitations']], '',
              'AI最终声明仍由用户最后如实整理；实际协助与本轮停止/范围修订记录均保留。', '']
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-root', type=Path, default=HERE)
    p.add_argument('--v1-project', type=Path, required=True)
    p.add_argument('--artifacts-root', type=Path, required=True)
    p.add_argument('--scope-plan', type=Path, required=True)
    p.add_argument('--split', choices=('validation', 'test'), required=True)
    p.add_argument('--retrospective-test-ack', action='store_true')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(argv)
    require(not a.output_dir.exists(), 'Use a fresh output directory; preserve every earlier result.')
    if a.split == 'test':
        require(a.retrospective_test_ack, 'Known TEST requires explicit retrospective acknowledgement.')
    report = compare(a)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (a.output_dir / 'RESULTS.md').write_text(render_markdown(report))
    print(json.dumps({'status': report['status'], 'split': a.split, 'output': str(a.output_dir),
                      'gate': report['adoption_gate'], 'audit_counts': report['audit_counts']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
