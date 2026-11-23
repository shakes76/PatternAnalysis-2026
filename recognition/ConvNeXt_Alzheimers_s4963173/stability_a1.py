"""Bounded A1 replication: two new seeds, unchanged V2 trainer, no test inference.

Runs live under the private root. The seed-42 reference and old V2 pointers are
read-only. New runs call train_convnext_v2.train_run with only Config.seed changed.
This is a descriptive stability check on an already-used validation cohort, not
an independent test or an automatic model-selection/ensembling procedure.

Development assistance: ChatGPT. Review, test and disclose this contribution.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import traceback
import zipfile

import train_convnext_v2 as v2
from training_utils import read_json, sha256_file

NEW_SEEDS = (43, 44)
ALL_SEEDS = (42, 43, 44)
EXTENSION_FILES = ('stability_a1.py', 'A1_STABILITY_PLAN.md',
                   'tests_stability/test_stability.py')
REFERENCE_FILES = ('v2_summary.json', 'history.csv', 'PRIVATE_best_model.pt',
                   'PRIVATE_validation_predictions.json')
STUDY_NAME = 'a1-stability'
MARKER = '<!-- COMP3710-A1-STABILITY -->'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def extension_hashes():
    base = Path(__file__).resolve().parent
    return {name: sha256_file(base / name) for name in EXTENSION_FILES}


def git(project, *args):
    result = subprocess.run(['git', '-C', str(project), *args], capture_output=True,
                            text=True, encoding='utf-8', errors='replace')
    require(result.returncode == 0, 'Git check failed: ' + result.stderr.strip())
    return result.stdout.strip()


def private_path(path, private):
    path, private = Path(path).resolve(), Path(private).resolve()
    require(path != private and private in path.parents,
            'Results must stay below the existing private root.')
    require(not any((parent / '.git').exists() for parent in (path, *path.parents)),
            'Private results/checkpoints must stay outside Git.')
    return path


def file_hashes(folder, names):
    result = {}
    for name in names:
        path = Path(folder) / name
        require(path.is_file(), 'Required saved evidence is missing: ' + name)
        result[name] = sha256_file(path)
    return result


def load_history(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), 'Training history is empty.')
    needed = ('epoch', 'patient_accuracy', 'patient_macro_f1', 'patient_log_loss',
              'patient_auroc', 'ad_recall', 'nc_recall')
    for index, row in enumerate(rows, 1):
        require(all(key in row for key in needed), 'History columns are incomplete.')
        require(int(row['epoch']) == index, 'History must contain consecutive epochs from 1.')
        require(all(math.isfinite(float(row[key])) for key in needed),
                'History contains non-finite values.')
    return rows


def verify_completed(folder, expected_config=None):
    report = read_json(Path(folder) / 'v2_summary.json')
    require(report.get('experiment_completed') is True and report.get('status') == 'COMPLETED',
            'Run has not completed; resume it instead of comparing incomplete results.')
    require(report.get('recipe') == 'scratch_augmented', 'Expected A1, not A2 or transfer learning.')
    require(report.get('pretrained') == {'used': False}, 'This study uses scratch initialization only.')
    require(report.get('new_final_test_model_evaluation_performed') is False and
            report.get('new_test_images_decoded') == 0, 'Unexpected final-test access.')
    require(report.get('training_patient_weighting') is False, 'A1 does not use patient weighting.')
    require(report.get('operating_threshold') == 0.5, 'Decision threshold changed.')
    config = v2.Config(**report['configuration'])
    config.validate()
    if expected_config is not None:
        require(asdict(config) == asdict(expected_config), 'Configuration differs from the locked A1 recipe.')
    rows = load_history(Path(folder) / 'history.csv')
    require(len(rows) == report.get('epochs_completed'), 'History and summary epoch counts differ.')
    chosen = max(rows, key=lambda r: (float(r['patient_accuracy']),
                                     float(r['patient_macro_f1']), -float(r['patient_log_loss'])))
    require(int(chosen['epoch']) == report.get('selected_epoch'),
            'Selected epoch does not follow accuracy/F1/log-loss selection and earliest exact tie.')
    m = report['selected_validation_metrics']['patient']
    for field, column in (('accuracy', 'patient_accuracy'), ('macro_f1', 'patient_macro_f1'),
                          ('auroc', 'patient_auroc'), ('log_loss', 'patient_log_loss')):
        require(math.isclose(float(m[field]), float(chosen[column]), rel_tol=0, abs_tol=1e-10),
                'Selected metrics disagree with history: ' + field)
    for label, column in (('AD', 'ad_recall'), ('NC', 'nc_recall')):
        require(math.isclose(float(m['per_class'][label]['recall']), float(chosen[column]),
                             rel_tol=0, abs_tol=1e-10), 'Recall disagrees with history.')
    cm = m['confusion_matrix_rows_true_columns_predicted_NC_AD']
    require(len(cm) == 2 and all(len(row) == 2 for row in cm), 'Expected a binary confusion matrix.')
    require(sum(map(sum, cm)) == m['n'] and m['n'] > 0, 'Confusion-matrix sample count differs.')
    require(math.isclose((cm[0][0] + cm[1][1]) / m['n'], m['accuracy'], abs_tol=1e-10),
            'Confusion matrix and accuracy differ.')
    hashes = file_hashes(folder, REFERENCE_FILES)
    return report, rows, hashes


def context(private):
    private, project, bundle, pipeline = v2.local_context(private)
    private_path(private / STUDY_NAME, private)
    reference = private_path(v2.pointer(private, 'latest_v2_scratch_augmented.txt'), private)
    report, rows, hashes = verify_completed(reference)
    require(report['configuration']['seed'] == 42, 'The preserved reference must be A1 seed 42.')
    require(report['source_code_sha256'] == v2.source_hashes(),
            'Historical V2 source changed. Do not bypass the source check.')
    require(report['manifest_sha256'] == bundle.manifest_hashes, 'Patient manifests changed.')
    require(asdict(v2.Config(**report['configuration'])) == report['configuration'],
            'Reference configuration is not fully specified by the installed trainer.')
    smoke = read_json(v2.pointer(private, 'latest_v2_smoke.txt') / 'v2_smoke_summary.json')
    require(smoke.get('passed') is True and smoke.get('source_hashes') == v2.source_hashes()
            and smoke.get('manifest_sha256') == bundle.manifest_hashes,
            'The previously passed V2 GPU check must match the unchanged source and manifests.')
    return private, project, bundle, pipeline, reference, report, hashes


def study_payload(reference, report, reference_hashes):
    return {
        'format_version': 1, 'created_utc': utc_now(),
        'purpose': 'Two predeclared A1 replications; development-only descriptive stability.',
        'reference_folder_PRIVATE': str(reference), 'reference_files_sha256': reference_hashes,
        'reference_configuration': report['configuration'],
        'core_source_sha256': report['source_code_sha256'],
        'extension_sha256': extension_hashes(),
        'manifest_sha256': report['manifest_sha256'], 'reference_environment': report['hardware'],
        'existing_seed': 42, 'additional_seeds': list(NEW_SEEDS),
        'provisional_candidate_seed': 42,
        'selection_policy': 'Retain seed 42 provisionally; do not switch to the luckiest replicate automatically.',
        'no_ensembling': True, 'test_previously_exposed': True,
        'new_test_inference_permitted': False,
        'reference_selected_patient_metrics': report['selected_validation_metrics']['patient'],
    }


def check_study(study, reference, report, reference_hashes):
    require(study.get('format_version') == 1 and study.get('additional_seeds') == list(NEW_SEEDS)
            and study.get('existing_seed') == 42 and study.get('provisional_candidate_seed') == 42,
            'Unexpected stability plan; preserve it and report the mismatch.')
    require(Path(study['reference_folder_PRIVATE']).resolve() == reference,
            'Original A1 pointer changed. Do not silently change the reference.')
    require(study['reference_files_sha256'] == reference_hashes, 'Preserved A1 evidence changed.')
    require(study['reference_configuration'] == report['configuration'] and
            study['core_source_sha256'] == v2.source_hashes() and
            study['extension_sha256'] == extension_hashes(), 'Code or recipe changed after study preparation.')
    require(study['manifest_sha256'] == report['manifest_sha256'] and
            study['reference_environment'] == report['hardware'], 'Study inputs changed.')
    require(study.get('new_test_inference_permitted') is False and study.get('no_ensembling') is True,
            'The study scope changed.')


def get_study(ctx):
    private, project, bundle, pipeline, reference, report, hashes = ctx
    folder = private / STUDY_NAME
    require((folder / 'study.json').is_file(), 'Run the check command first.')
    study = read_json(folder / 'study.json')
    check_study(study, reference, report, hashes)
    return folder, study


def prepare(private):
    ctx = context(private)
    private, project, bundle, pipeline, reference, report, hashes = ctx
    device = v2.choose_device(report['configuration']['device'])
    require(v2.environment(device) == report['hardware'],
            'Software/GPU differs from seed 42; stop before claiming a matched replication.')
    folder = private / STUDY_NAME
    folder.mkdir(exist_ok=True)
    with v2.run_lock(folder):
        if (folder / 'study.json').exists():
            study = read_json(folder / 'study.json')
            check_study(study, reference, report, hashes)
        else:
            require(not any(folder.glob('seed-*')), 'Seed folders exist without a plan; preserve and report.')
            study = study_payload(reference, report, hashes)
            v2.atomic_json(folder / 'study.json', study)
    print('A1_STABILITY_READY')
    print('Original A1/A2 preserved. New seeds: 43 and 44. Final-test inference: disabled.')
    print('Available private-drive space: %.2f GiB' % (shutil.disk_usage(private).free / 2**30))


def require_committed(project):
    """Do not mutate Git. Require this small plan/extension to have been committed."""
    prefix = git(project, 'rev-parse', '--show-prefix')
    require(prefix.startswith('recognition/') and prefix != 'recognition/', 'Wrong assignment folder.')
    require(git(project, 'branch', '--show-current') == 'topic-recognition', 'Wrong Git branch.')
    for name in EXTENSION_FILES:
        git(project, 'ls-files', '--error-unmatch', '--', name)
    require(not git(project, 'diff', 'HEAD', '--', *EXTENSION_FILES),
            'Commit the stability extension and plan before training (without changing existing V2 code).')


def seeded_config(study, seed):
    require(seed in NEW_SEEDS, 'Only the two declared replication seeds 43 and 44 may be run.')
    return replace(v2.Config(**study['reference_configuration']), seed=seed)


def ensure_run_metadata(run, study, config):
    expected = {'format_version': 1, 'seed': config.seed,
                'configuration': asdict(config), 'study_sha256': digest_json(study),
                'extension_sha256': extension_hashes(), 'core_source_sha256': v2.source_hashes()}
    p = run / 'run_metadata.json'
    if p.exists():
        require(read_json(p) == expected, 'Run settings/source differ; do not overwrite the run.')
    else:
        require(not any(run.iterdir()), 'Existing run lacks metadata; preserve it and report.')
        v2.atomic_json(p, expected)
    cfg = run / 'config.json'
    if cfg.exists():
        require(read_json(cfg) == asdict(config), 'Saved run configuration differs.')
    else:
        v2.atomic_json(cfg, asdict(config))


def verify_run_binding(run, study, config):
    expected = {'format_version': 1, 'seed': config.seed,
                'configuration': asdict(config), 'study_sha256': digest_json(study),
                'extension_sha256': extension_hashes(), 'core_source_sha256': v2.source_hashes()}
    require(read_json(run / 'run_metadata.json') == expected, 'Run is not bound to this study.')
    require(read_json(run / 'config.json') == asdict(config), 'Run configuration changed.')


def run_seed(private, seed):
    ctx = context(private)
    private, project, bundle, pipeline, reference, report, hashes = ctx
    folder, study = get_study(ctx)
    require_committed(project)
    config = seeded_config(study, seed)
    device = v2.choose_device(config.device)
    require(v2.environment(device) == study['reference_environment'],
            'Software/GPU differs from the reference; stop.')
    # A single OS lock prevents concurrent seeds and is released on process death.
    with v2.run_lock(folder):
        run = folder / ('seed-' + str(seed))
        run.mkdir(exist_ok=True)
        ensure_run_metadata(run, study, config)
        try:
            completed = run / 'stability_run_complete.json'
            if completed.exists():
                r, rows, current_hashes = verify_completed(run, config)
                receipt = read_json(completed)
                require(receipt['file_sha256'] == current_hashes and
                        receipt['study_sha256'] == digest_json(study), 'Completed evidence changed.')
                print('A1_STABILITY_SEED_ALREADY_COMPLETED:', seed)
                return
            require(shutil.disk_usage(private).free >= 4 * 2**30,
                    'Keep at least 4 GiB free for checkpoints; do not delete earlier evidence blindly.')
            resumable = any((run / name).exists() for name in ('PRIVATE_resume.pt', 'PRIVATE_previous.pt'))
            if not resumable:
                require(not (run / 'history.csv').exists() and not (run / 'v2_summary.json').exists(),
                        'Training evidence exists but no resumable checkpoint; preserve and report.')
            print(('Resuming' if resumable else 'Starting') + ' A1 seed ' + str(seed), flush=True)
            with v2.run_lock(run):
                state = v2.train_run(run, bundle, config, pipeline, private, resume=resumable)
            require(state['completed'] is True, 'Training returned without completion; rerun this seed to resume.')
            r, rows, current_hashes = verify_completed(run, config)
            require(r['manifest_sha256'] == study['manifest_sha256'] and
                    r['source_code_sha256'] == study['core_source_sha256'] and
                    r['hardware'] == study['reference_environment'], 'Completed run differs from the study.')
            v2.atomic_json(completed, {'seed': seed, 'study_sha256': digest_json(study),
                                      'file_sha256': current_hashes})
            print('A1_STABILITY_SEED_COMPLETED:', seed)
        except BaseException:
            atomic_text(run / 'PRIVATE_stability_error.txt', traceback.format_exc())
            print('A1_STABILITY_PAUSED: preserve this folder and rerun the SAME seed command to resume.',
                  file=sys.stderr, flush=True)
            raise


def score_row(seed, report, history):
    m = report['selected_validation_metrics']['patient']
    cm = m['confusion_matrix_rows_true_columns_predicted_NC_AD']
    selected = next(row for row in history if int(row['epoch']) == report['selected_epoch'])
    return {'seed': seed, 'selected_epoch': report['selected_epoch'],
            'epochs_completed': report['epochs_completed'], 'correct_patients': cm[0][0] + cm[1][1],
            'patients': m['n'], 'patient_accuracy': m['accuracy'], 'macro_f1': m['macro_f1'],
            'auroc': m['auroc'], 'patient_log_loss': m['log_loss'],
            'ad_recall': m['per_class']['AD']['recall'], 'nc_recall': m['per_class']['NC']['recall'],
            'ad_false_negatives': cm[1][0], 'nc_false_positives': cm[0][1],
            'epochs_at_or_above_80_percent_validation': sum(float(r['patient_accuracy']) >= .8 for r in history),
            'selected_train_slice_accuracy': float(selected['train_slice_accuracy']),
            'selected_validation_slice_accuracy': float(report['selected_validation_metrics']['slice']['accuracy']),
            'stop_reason': report['stop_reason'], 'resume_count': report['resume_count'],
            'training_seconds_committed_epochs': report['training_loop_seconds_committed_epochs']}


def describe(rows):
    result = {}
    for key in ('patient_accuracy', 'macro_f1', 'auroc', 'ad_recall', 'nc_recall'):
        values = [row[key] for row in rows]
        result[key] = {'mean': statistics.mean(values),
                       'sample_sd': statistics.stdev(values) if len(values) > 1 else None,
                       'min': min(values), 'max': max(values), 'number_of_runs': len(values)}
    return result


def make_comparison(rows, study):
    require([row['seed'] for row in rows] == list(ALL_SEEDS), 'All three distinct seeds are required.')
    require(len({row['patients'] for row in rows}) == 1, 'Validation patient counts differ.')
    return {
        'format_version': 1, 'stability_check_completed': True, 'seeds': list(ALL_SEEDS),
        'new_replication_seeds': list(NEW_SEEDS), 'runs': rows,
        'all_three_selected_checkpoint_descriptive_statistics': describe(rows),
        'two_new_replications_descriptive_statistics': describe(rows[1:]),
        'unique_validation_patients_per_run': rows[0]['patients'],
        'same_patients_reused_across_seeds': True,
        'replication_seeds_reaching_80_percent_validation': [r['seed'] for r in rows[1:] if r['patient_accuracy'] >= .8],
        'provisional_candidate_seed': 42, 'candidate_changed_automatically': False,
        'ensembling_performed': False, 'new_final_test_model_evaluation_performed': False,
        'test_accuracy_requirement_demonstrated_by_this_analysis': False,
        'source_code_sha256': study['core_source_sha256'],
        'extension_sha256': study['extension_sha256'], 'manifest_sha256': study['manifest_sha256'],
        'reference_configuration': study['reference_configuration'],
        'limitations': [
            'Same validation cohort used for recipe choice and checkpoint selection; these are selected development scores.',
            'Seed 42 already helped choose A1. The two new runs are replications, not independent patient cohorts.',
            'Sample standard deviations describe these runs; they are not confidence intervals or clinical validation.',
            'No automatic replacement of seed 42 by the highest-scoring seed, no ensemble, no threshold fitting.',
            'No untouched-test claim: Version 1 test was exposed; revised evaluation protocol remains to be agreed.'
        ]}


def comparison_markdown(summary):
    lines = ['## A1 replication stability check', '',
             '**Development results only. No final-test inference or threshold fitting was performed.**', '',
             'The existing seed-42 A1 recipe was repeated at the predeclared seeds 43 and 44. Only the random seed changed. '
             'This changes initialization, stochastic layers, augmentation draws and training order, not just initialization. '
             'Each run selected its checkpoint by validation patient accuracy, then macro-F1, then lower log loss; '
             'the earliest exact tie was retained.', '',
             '| Seed | Selected epoch / completed | Correct patients | Accuracy | Macro-F1 | AUROC | AD recall | NC recall | Epochs ≥80% |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['runs']:
        lines.append(f'| {r["seed"]} | {r["selected_epoch"]} / {r["epochs_completed"]} | '
                     f'{r["correct_patients"]}/{r["patients"]} | {r["patient_accuracy"]:.2%} | '
                     f'{r["macro_f1"]:.4f} | {r["auroc"]:.4f} | {r["ad_recall"]:.2%} | '
                     f'{r["nc_recall"]:.2%} | {r["epochs_at_or_above_80_percent_validation"]} |')
    lines += ['', '| Cohort of runs | Mean accuracy | Sample SD (percentage points) | Accuracy range |',
              '|---|---:|---:|---:|']
    for title, key in [('All three (includes recipe-selection seed 42)', 'all_three_selected_checkpoint_descriptive_statistics'),
                       ('New replications only (43 and 44)', 'two_new_replications_descriptive_statistics')]:
        d = summary[key]['patient_accuracy']
        lines.append(f'| {title} | {d["mean"]:.2%} | {100*d["sample_sd"]:.2f} | {d["min"]:.2%}–{d["max"]:.2%} |')
    lines += ['', 'These are repeated measurements on the same patients, not a larger independent test cohort. '
              'Selected-checkpoint means and sample standard deviations are descriptive; they do not establish an 80% test result.', '',
              'Seed 42 remains the provisional candidate. No seed has been automatically substituted, no models have been '
              'ensembled, and no calibration or referral rule has been transferred from Version 1. The results require '
              'review before another model-development decision or revised evaluation.', '',
              'Existing A1/A2 histories and private checkpoints were preserved. Resumption repeats only work after the '
              'last usable completed-epoch checkpoint. Runtime excludes interrupted partial work and checkpoint/report I/O.', '',
              'ChatGPT assisted with this additive runner, tests and draft. Document the code review and checks actually '
              'performed in the Artificial Intelligence Usage Disclosure; do not claim clinical verification.', '']
    return '\n'.join(lines)


def compare(private):
    ctx = context(private)
    private, project, bundle, pipeline, reference, ref_report, ref_hashes = ctx
    folder, study = get_study(ctx)
    with v2.run_lock(folder):
        reports, histories, hashes, folders = {}, {}, {}, {42: reference}
        for seed in ALL_SEEDS:
            config = replace(v2.Config(**study['reference_configuration']), seed=seed)
            run = reference if seed == 42 else folder / ('seed-' + str(seed))
            if seed != 42:
                require((run / 'stability_run_complete.json').is_file(),
                        'Seed %s has not completed. Run that seed command; do not compare early.' % seed)
                verify_run_binding(run, study, config)
            report, history, files = verify_completed(run, config)
            require(report['source_code_sha256'] == study['core_source_sha256'] and
                    report['manifest_sha256'] == study['manifest_sha256'] and
                    report['hardware'] == study['reference_environment'], 'Runs are not a matched comparison.')
            if seed != 42:
                receipt = read_json(run / 'stability_run_complete.json')
                require(receipt['file_sha256'] == files and receipt['study_sha256'] == digest_json(study),
                        'Completed run evidence changed.')
            reports[seed], histories[seed], hashes[str(seed)], folders[seed] = report, history, files, run
        result = make_comparison([score_row(s, reports[s], histories[s]) for s in ALL_SEEDS], study)
        result['input_file_sha256'] = hashes
        out = folder / 'comparison'
        out.mkdir(exist_ok=True)
        v2.atomic_json(out / 'stability_summary.json', result)
        atomic_text(out / 'README_a1_stability.md', comparison_markdown(result))
        dest = private / 'COMP3710_A1_Stability_Results.zip'
        temp = dest.with_name(dest.name + '.tmp')
        with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(out / 'stability_summary.json', 'stability_summary.json')
            archive.write(out / 'README_a1_stability.md', 'README_a1_stability.md')
            archive.write(project / 'A1_STABILITY_PLAN.md', 'A1_STABILITY_PLAN.md')
            for seed in ALL_SEEDS:
                # Explicit whitelist: never package checkpoints, predictions, study.json or errors.
                for name in ('v2_summary.json', 'history.csv', 'V2_RESULTS.md'):
                    archive.write(folders[seed] / name, 'seed_%s/%s' % (seed, name))
        os.replace(temp, dest)
        v2.atomic_json(out / 'publication_receipt.json', {
            'report_sha256': sha256_file(out / 'README_a1_stability.md'),
            'summary_sha256': sha256_file(out / 'stability_summary.json'),
            'bundle_sha256': sha256_file(dest)})
    print('A1_STABILITY_COMPARISON_READY')
    print(dest)


def publish(private):
    ctx = context(private)
    private, project, bundle, pipeline, reference, report, hashes = ctx
    folder, study = get_study(ctx)
    with v2.run_lock(folder):
        out = folder / 'comparison'
        receipt = read_json(out / 'publication_receipt.json')
        require(sha256_file(out / 'README_a1_stability.md') == receipt['report_sha256'] and
                sha256_file(out / 'stability_summary.json') == receipt['summary_sha256'],
                'Comparison output changed; preserve and report.')
        summary = read_json(out / 'stability_summary.json')
        require(summary.get('stability_check_completed') is True and summary.get('seeds') == list(ALL_SEEDS),
                'A completed three-seed comparison is required.')
        require(summary['extension_sha256'] == extension_hashes() and
                summary['source_code_sha256'] == v2.source_hashes(), 'Compared source changed.')
        # Preserve the original README bytes. Do not duplicate a completed append.
        readme = project / 'README.md'
        require(readme.is_file(), 'Assignment README is missing.')
        original = readme.read_bytes()
        text = original.decode('utf-8-sig')
        if MARKER in text:
            print('A1_STABILITY_REPORT_ALREADY_PRESENT')
            return
        git(project, 'rev-parse', '--show-toplevel')
        require(git(project, 'branch', '--show-current') == 'topic-recognition', 'Wrong branch.')
        backup = folder / ('README-before-stability-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.md')
        backup.write_bytes(original)
        addition = ('\n\n' + MARKER + '\n' + (out / 'README_a1_stability.md').read_text(encoding='utf-8')).encode('utf-8')
        temp = readme.with_name('README.md.stability-tmp')
        temp.write_bytes(original + addition)
        os.replace(temp, readme)
    print('A1_STABILITY_REPORT_APPENDED')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'run', 'compare', 'publish'))
    parser.add_argument('--private-root', required=True, type=Path)
    parser.add_argument('--seed', type=int, choices=NEW_SEEDS)
    args = parser.parse_args(argv)
    if args.command == 'run':
        require(args.seed is not None, 'Run requires --seed 43 or --seed 44.')
        run_seed(args.private_root, args.seed)
    else:
        require(args.seed is None, '--seed is only used with run.')
        {'check': prepare, 'compare': compare, 'publish': publish}[args.command](args.private_root)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print('A1_STABILITY_STOPPED: %s: %s' % (type(error).__name__, error), file=sys.stderr)
        sys.exit(1)
