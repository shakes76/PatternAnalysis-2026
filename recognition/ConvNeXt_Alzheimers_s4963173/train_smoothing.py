"""Two fixed ConvNeXt scratch-training experiments: S1 smoothing, S2 schedule.

Additive development runner: historical V2/stability code and artifacts are read
only. No pretrained loading, test inference, threshold search or split changes.
Model implementation remains in modules.py; validation uses ordinary hard labels.

AI assistance: ChatGPT drafted this extension/tests. Review, verify and disclose.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile

import torch
from torch import nn
from torch.utils.data import DataLoader

import train_convnext_v2 as v2
import stability_a1 as prior
from dataset import ADNISliceDataset, preprocessing_spec
from training_utils import (aggregate_predictions, predict_validation, read_json,
                            sha256_file, synchronize, seed_everything, choose_device,
                            memory_info, reset_memory_peak, binary_metrics)

EXPERIMENTS = ('S1', 'S2')
STUDY = 'convnext-smoothing'
NEW_FILES = ('train_smoothing.py', 'SMOOTHING_PLAN.md',
             'tests_smoothing/test_smoothing.py')
DEPENDENCIES = ('stability_a1.py',)
PUBLIC_RUN_FILES = ('smoothing_summary.json', 'history.csv', 'SMOOTHING_RESULTS.md',
                    'smoothing_loss.png', 'smoothing_validation.png')
PRIVATE_RESULT_FILES = ('PRIVATE_best_model.pt', 'PRIVATE_validation_predictions.json')
MARKER = '<!-- COMP3710-S1-S2-SMOOTHING -->'
require = prior.require
atomic_text = prior.atomic_text


@dataclass(frozen=True)
class Config(v2.Config):
    """The inherited recipe stays scratch_augmented: no transfer or weighting."""
    experiment: str = 'S1'
    label_smoothing: float = 0.10
    decay_end_epoch: int = 30

    def validate(self):
        super().validate()
        require(self.experiment in EXPERIMENTS, 'Choose only S1 or S2.')
        require(self.recipe == 'scratch_augmented', 'Only the unweighted scratch A1 recipe is permitted.')
        require(0 <= self.label_smoothing < 1, 'Invalid smoothing coefficient.')
        require(1 <= self.warmup_epochs < self.decay_end_epoch <= self.epochs,
                'Require warmup < decay endpoint <= maximum epochs.')
        require(0 < self.min_lr <= self.learning_rate, 'Invalid learning-rate bounds.')
        require(self.seed == 42, 'This fixed comparison uses seed 42 only.')


def configuration(reference: dict, experiment: str) -> Config:
    """Copy the recorded A1 configuration, altering only the declared settings."""
    require(experiment in EXPERIMENTS, 'Only the two declared experiments S1 and S2 are supported.')
    require(reference == asdict(v2.Config('scratch_augmented')),
            'A1 settings differ from the agreed recipe. Preserve them and report the mismatch.')
    fields = dict(reference)
    fields.update(experiment=experiment, label_smoothing=.10,
                  decay_end_epoch=30 if experiment == 'S1' else 12)
    if experiment == 'S2':
        fields['learning_rate'] = 1e-4
    result = Config(**fields)
    result.validate()
    return result


def learning_rate(epoch: int, config: Config) -> float:
    require(1 <= epoch <= config.epochs, 'Epoch outside the configured schedule.')
    if epoch <= config.warmup_epochs:
        return config.learning_rate * epoch / config.warmup_epochs
    fraction = min(1., (epoch-config.warmup_epochs-1) /
                   max(1, config.decay_end_epoch-config.warmup_epochs-1))
    return config.min_lr + .5*(config.learning_rate-config.min_lr)*(1+math.cos(math.pi*fraction))


def set_epoch(model, optimizer, config, epoch):
    """All scratch-model parameters are trainable; all groups use the same LR."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    lr = learning_rate(epoch, config)
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr


def per_example_losses(logits: torch.Tensor, labels: torch.Tensor,
                       epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (uniform-smoothed CE, ordinary hard-label CE), without reduction.

    (1-eps)*one_hot + eps/K gives (.95, .05) for K=2 and eps=.10.
    Explicit log_softmax avoids depending on a CUDA NLLLoss reduction in strict
    deterministic mode. No labels or evaluation probabilities are smoothed.
    """
    require(logits.ndim == 2 and logits.shape[1] == 2 and
            labels.shape == (len(logits),) and labels.dtype == torch.long,
            'Expected two-class logits [B,2] and int64 labels [B].')
    require(0 <= epsilon < 1, 'Invalid label smoothing.')
    require(bool(torch.isfinite(logits).all()) and bool(((labels == 0) | (labels == 1)).all()),
            'Invalid logits or class labels.')
    logp = torch.log_softmax(logits, dim=1)
    hard = -logp.gather(1, labels[:, None]).squeeze(1)
    smooth = (1-epsilon)*hard - epsilon*logp.mean(dim=1)
    return smooth, hard


def train_epoch(model, loader, optimizer, device, config):
    require(getattr(loader.dataset, 'split', None) == 'train', 'Only training data may update weights.')
    require(loader.batch_size == config.batch_size and not loader.drop_last,
            'Unexpected training batch configuration.')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = len(loader.dataset)
    effective = config.batch_size*config.accumulation_steps
    count = correct = updates = 0
    loss_sum = hard_sum = 0.
    synchronize(device)
    started = time.perf_counter()
    for index, batch in enumerate(loader):
        require(set(batch['split']) == {'train'}, 'Non-training batch supplied to optimiser.')
        if index % config.accumulation_steps == 0:
            window = min(effective, total-count)
        x, y = batch['image'].to(device), batch['label'].to(device)
        require(bool((batch['loss_weight'] == 1).all()), 'Patient/class weighting is disabled in S1/S2.')
        logits = model(x)
        smooth, hard = per_example_losses(logits, y, config.label_smoothing)
        loss = smooth.sum()/window
        require(bool(torch.isfinite(loss)), 'Non-finite training objective.')
        loss.backward()
        count += len(y)
        correct += int((logits.detach().argmax(1) == y).sum())
        loss_sum += float(smooth.detach().sum())
        hard_sum += float(hard.detach().sum())
        if (index+1) % config.accumulation_steps == 0 or count == total:
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        if (index+1) % 400 == 0:
            print(f'  {count}/{total} training images', flush=True)
    synchronize(device)
    require(count == total and updates == math.ceil(total/effective), 'Incomplete training epoch.')
    return {'loss': loss_sum/count, 'hard_log_loss': hard_sum/count,
            'slice_accuracy': correct/count, 'updates': updates,
            'seconds': time.perf_counter()-started}


def source_hashes():
    base = Path(__file__).resolve().parent
    return {**v2.source_hashes(),
            **{name: sha256_file(base/name) for name in (*NEW_FILES, *DEPENDENCIES)}}


def confidence_counts(predictions):
    """Aggregate only. Raw confidence at .5; no calibration or cutoff selection."""
    rows = predictions['patients']
    wrong = [r for r in rows if int(r['p_ad'] >= .5) != r['label']]
    return {'patients': len(rows), 'incorrect_patients': len(wrong),
            'incorrect_at_raw_confidence_ge_0_90': sum(max(r['p_ad'], 1-r['p_ad']) >= .9 for r in wrong),
            'patient_brier_score': math.fsum((r['p_ad']-r['label'])**2 for r in rows)/len(rows)}


def plot_history(folder, history):
    if not history:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for filename, fields, ylabel in (
        ('smoothing_loss.png', ('train_smoothed_loss', 'train_hard_log_loss',
                                'validation_hard_log_loss'), 'Loss'),
        ('smoothing_validation.png', ('patient_accuracy', 'patient_macro_f1', 'ad_recall'), 'Score')):
        fig, ax = plt.subplots(figsize=(8, 5))
        for field in fields:
            ax.plot([r['epoch'] for r in history], [r[field] for r in history], label=field)
        ax.set(xlabel='Epoch', ylabel=ylabel,
               title='Development only; validation metrics use hard labels')
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(folder)/filename, dpi=150)
        plt.close(fig)


def summary_for(state):
    best = state['best']
    return {
        'format_version': 1, 'status': 'COMPLETED' if state['completed'] else 'IN_PROGRESS',
        'experiment_completed': state['completed'], 'experiment': state['configuration']['experiment'],
        'configuration': state['configuration'], 'architecture': 'ConvNeXtTiny',
        'parameters': state['parameters'], 'pretrained': {'used': False},
        'label_smoothing_target_for_correct_class': 1-state['configuration']['label_smoothing']/2,
        'label_smoothing_target_for_other_class': state['configuration']['label_smoothing']/2,
        'validation_targets_smoothed': False, 'training_patient_weighting': False,
        'operating_threshold': .5, 'preprocessing': preprocessing_spec(0.),
        'selected_epoch': best['epoch'] if best else None,
        'selected_validation_metrics': best['metrics'] if best else None,
        'selected_confidence_diagnostics': confidence_counts(best['private_predictions']) if best else None,
        'selection_rule': 'validation patient accuracy, then macro-F1, then lower ordinary log loss; earliest exact tie',
        'epochs_completed': state['epoch'], 'stop_reason': state.get('stop_reason'),
        'resume_count': state.get('resume_count', 0),
        'hardware': state['environment'], 'gpu_memory': state.get('memory', {}),
        'training_loop_seconds_committed_epochs': sum(r['train_seconds']+r['validation_seconds']
                                                       for r in state['history']),
        'runtime_scope': 'Completed training and validation only; excludes interrupted work, checkpoints/report I/O and downtime.',
        'source_code_sha256': state['source_hashes'], 'manifest_sha256': state['manifest_hashes'],
        'checkpoint_format': 'Full model/optimizer/RNG/best/history at complete epoch boundaries.',
        'test_previously_exposed_in_v1': True, 'new_test_images_decoded': 0,
        'new_final_test_model_evaluation_performed': False,
        'interpretation': 'Selected development evidence, not proof of 0.80 test accuracy or clinical safety.'}


def result_markdown(summary):
    s = summary
    lines = [f'## {s["experiment"]}: controlled scratch-training experiment', '',
             '**Selected validation results; no new test inference.**', '',
             'Configuration: `' + json.dumps(s['configuration'], sort_keys=True) + '`.', '',
             'Uniform smoothing 0.10 gives binary training targets 0.95/0.05. Labels, validation loss and '
             'metrics remain hard-label quantities. Smoothed training loss is not directly comparable to '
             'the historical unsmoothed training objective; ordinary training log loss is recorded separately.', '',
             f'Completed epochs: {s["epochs_completed"]}; selected epoch: {s["selected_epoch"]}; '
             f'stopping reason: {s["stop_reason"]}.', '',
             '| Validation unit | Accuracy | Macro-F1 | AUROC | AD recall |',
             '|---|---:|---:|---:|---:|']
    if s['selected_validation_metrics']:
        for unit, m in s['selected_validation_metrics'].items():
            auc = 'Not estimable' if m['auroc'] is None else f'{m["auroc"]:.4f}'
            lines.append(f'| {unit} | {m["accuracy"]:.2%} | {m["macro_f1"]:.4f} | '
                         f'{auc} | {m["per_class"]["AD"]["recall"]:.2%} |')
    lines += ['', 'S1 changes the training loss relative to A1. S2 additionally changes peak learning rate '
              'and decay timing; these two schedule components are not isolated from each other. '
              'Both use seed 42, A1 augmentation, no patient weighting and a 0.5 class cutoff.', '',
              'Checkpoint selection repeatedly uses the same validation cohort. A higher score is not '
              'independent evidence of generalisation. Neither run automatically replaces A1 or changes '
              'the calibration/referral policy. Prior test exposure must remain disclosed.', '',
              f'Parameters: {s["parameters"]:,}. Committed-epoch training/validation time: '
              f'{s["training_loop_seconds_committed_epochs"]:.2f} seconds. GPU allocator measurements: '
              f'`{s["gpu_memory"]}`. {s["runtime_scope"]}', '',
              'ChatGPT assisted with code/tests and this report draft. Record actual personal code review '
              'and empirical checks in the Artificial Intelligence Usage Disclosure.', '']
    return '\n'.join(lines)


def artifacts(folder, state, complete=False):
    summary = summary_for(state)
    v2.atomic_json(folder/'smoothing_summary.json', summary)
    history = state['history']
    temporary = folder/'history.csv.tmp'
    with temporary.open('w', encoding='utf-8', newline='') as f:
        if history:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader(); writer.writerows(history)
    os.replace(temporary, folder/'history.csv')
    atomic_text(folder/'SMOOTHING_RESULTS.md', result_markdown(summary))
    if complete:
        best = state['best']
        require(best is not None, 'A completed run must contain a selected model.')
        v2.atomic_json(folder/'PRIVATE_validation_predictions.json', best['private_predictions'])
        checkpoint = {'format_version': 3, 'architecture': 'ConvNeXtTiny',
                      'epoch': best['epoch'], 'configuration': state['configuration'],
                      'model_state': best['model_state'], 'model_label_map': {'NC': 0, 'AD': 1},
                      'manifest_sha256': state['manifest_hashes'],
                      'source_code_sha256': state['source_hashes'],
                      'preprocessing': preprocessing_spec(0.), 'pretrained': {'used': False}}
        temporary = folder/'PRIVATE_best_model.tmp'
        with temporary.open('wb') as f:
            torch.save(checkpoint, f); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, folder/'PRIVATE_best_model.pt')
        plot_history(folder, history)


def train_run(folder, bundle, config, *, resume=False, model_factory=v2.make_model,
              max_new_epochs=None):
    """No CLI testing hooks: only synthetic tests use smaller model/epoch limits."""
    folder = Path(folder)
    config.validate()
    device = choose_device(config.device)
    seed_everything(config.seed)
    model = model_factory(config).to(device)
    optimizer = v2.make_optimizer(model, config)
    hashes, environment = source_hashes(), v2.environment(device)
    reset_memory_peak(device)
    state_path = folder/'PRIVATE_resume.pt'
    state = None
    if resume:
        recovered = None
        for candidate in (state_path, folder/'PRIVATE_previous.pt'):
            if not candidate.is_file():
                continue
            try:
                state = torch.load(candidate, map_location='cpu', weights_only=True)
                recovered = candidate
                break
            except Exception:
                continue
        require(state is not None, 'No readable complete checkpoint. Preserve the run and report the error.')
        require(state['format_version'] == 3 and state['configuration'] == asdict(config),
                'Checkpoint configuration differs; do not silently start a different experiment.')
        require(state['source_hashes'] == hashes and state['manifest_hashes'] == bundle.manifest_hashes,
                'Code or manifests changed since the checkpoint.')
        require(state['environment'] == environment, 'Software/GPU changed since this run.')
        if recovered != state_path and state_path.exists():
            os.replace(state_path, folder/('PRIVATE_unreadable-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.pt'))
        model.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        v2.restore_rng(state['rng'])
        if not state['completed']:
            state['resume_count'] += 1
    else:
        state = {'format_version': 3, 'configuration': asdict(config), 'source_hashes': hashes,
                 'manifest_hashes': bundle.manifest_hashes, 'environment': environment,
                 'epoch': 0, 'history': [], 'best': None, 'bad_epochs': 0, 'completed': False,
                 'parameters': v2.parameter_count(model), 'resume_count': 0,
                 'model_state': v2.tensor_state(model), 'optimizer_state': optimizer.state_dict(),
                 'rng': v2.rng_state(), 'schedule_state': {'last_completed_epoch': 0},
                 'loader_state': {'seed_rule': 'seed+epoch, workers=0'}}
        v2.save_state(state_path, state)
    if state['completed']:
        artifacts(folder, state, complete=True)
        return state
    artifacts(folder, state)
    train_data = v2.WeightedDataset(bundle, config)
    val_data = ADNISliceDataset(bundle, 'validation')
    finished = 0
    for epoch in range(state['epoch']+1, config.epochs+1):
        generator = torch.Generator().manual_seed(config.seed+epoch)
        loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=0,
                            generator=generator, pin_memory=device.type == 'cuda', drop_last=False)
        val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=0,
                               generator=torch.Generator().manual_seed(config.seed+100000+epoch))
        lr = set_epoch(model, optimizer, config, epoch)
        print(f'{config.experiment}: epoch {epoch}/{config.epochs}; learning rate {lr:.8g}', flush=True)
        tr = train_epoch(model, loader, optimizer, device, config)
        probabilities, seconds = predict_validation(model, val_loader, device)
        metrics, private_predictions = aggregate_predictions(val_data.records, probabilities)
        better = state['best'] is None or v2.selection_key(metrics) > v2.selection_key(state['best']['metrics'])
        if better:
            state['best'] = {'epoch': epoch, 'metrics': metrics, 'model_state': v2.tensor_state(model),
                             'private_predictions': private_predictions}
            state['bad_epochs'] = 0
        else:
            state['bad_epochs'] += 1
        m = metrics['patient']
        state['history'].append({'epoch': epoch, 'learning_rate': lr,
            'train_smoothed_loss': tr['loss'], 'train_hard_log_loss': tr['hard_log_loss'],
            'train_slice_accuracy': tr['slice_accuracy'], 'validation_hard_log_loss': metrics['slice']['log_loss'],
            'validation_slice_accuracy': metrics['slice']['accuracy'], 'patient_accuracy': m['accuracy'],
            'patient_macro_f1': m['macro_f1'], 'patient_auroc': m['auroc'], 'patient_log_loss': m['log_loss'],
            'ad_recall': m['per_class']['AD']['recall'], 'nc_recall': m['per_class']['NC']['recall'],
            'train_seconds': tr['seconds'], 'validation_seconds': seconds,
            'optimizer_updates': tr['updates'], 'selected_as_best': better})
        state.update(epoch=epoch, model_state=v2.tensor_state(model), optimizer_state=optimizer.state_dict(),
                     rng=v2.rng_state(), schedule_state={'last_completed_epoch': epoch, 'learning_rate': lr},
                     loader_state={'seed_rule': 'seed+epoch, workers=0',
                                   'last_generator_state': generator.get_state()})
        memory = memory_info(device)
        for key in ('peak_allocated_gib', 'peak_reserved_gib'):
            if memory.get(key) is not None:
                memory[key] = max(memory[key], state.get('memory', {}).get(key) or 0.)
        state['memory'] = memory
        if epoch == config.epochs:
            state.update(completed=True, stop_reason='maximum_epochs')
        elif epoch >= config.min_epochs and state['bad_epochs'] >= config.patience:
            state.update(completed=True, stop_reason='patience')
        v2.save_state(state_path, state)
        artifacts(folder, state)
        print(f'  validation patient accuracy {m["accuracy"]:.2%}, '
              f'AD recall {m["per_class"]["AD"]["recall"]:.2%}; checkpoint saved', flush=True)
        finished += 1
        if state['completed'] or (max_new_epochs is not None and finished >= max_new_epochs):
            break
    if state['completed']:
        artifacts(folder, state, complete=True)
    return state


def context(private):
    # Existing helper reads all manifest headers for integrity; no test images or
    # test predictions are opened. Historical A1 result hashes are checked here.
    ctx = prior.context(private)
    private, project, bundle, pipeline, reference, report, hashes = ctx
    prior.private_path(private/STUDY, private)
    return ctx


def study_payload(ctx):
    private, project, bundle, pipeline, reference, report, hashes = ctx
    return {'format_version': 1, 'created_utc': prior.utc_now(),
            'reference_folder_PRIVATE': str(reference), 'reference_files_sha256': hashes,
            'reference_configuration': report['configuration'], 'reference_environment': report['hardware'],
            'reference_patient_metrics': report['selected_validation_metrics']['patient'],
            'reference_epoch': report['selected_epoch'], 'manifest_sha256': bundle.manifest_hashes,
            'source_sha256': source_hashes(),
            'configurations': {name: asdict(configuration(report['configuration'], name)) for name in EXPERIMENTS},
            'scope': 'Two fixed scratch-training runs. No automatic confirmation run or test inference.',
            'classification_cutoff': .5, 'candidate_changed_automatically': False,
            'promotion_screen': 'Match/exceed A1 patient accuracy and AD recall; compare accuracy/F1/log loss thereafter.',
            'promotion_screen_is_clinical_safety_standard': False,
            'test_previously_exposed': True, 'new_test_inference_permitted': False}


def load_study(ctx):
    private, project, bundle, pipeline, reference, report, hashes = ctx
    folder = private/STUDY
    study = read_json(folder/'study.json')
    expected = study_payload(ctx)
    expected['created_utc'] = study.get('created_utc')
    require(study == expected, 'Study evidence, sources or settings changed. Preserve all files and report.')
    return folder, study


def check(private):
    ctx = context(private)
    private, project, bundle, pipeline, reference, report, hashes = ctx
    device = choose_device(report['configuration']['device'])
    require(v2.environment(device) == report['hardware'], 'Environment differs from A1. Do not upgrade packages.')
    folder = private/STUDY
    folder.mkdir(exist_ok=True)
    with v2.run_lock(folder):
        if (folder/'study.json').exists():
            load_study(ctx)
        else:
            require(not any((folder/name).exists() for name in EXPERIMENTS), 'Runs already exist without a study plan.')
            v2.atomic_json(folder/'study.json', study_payload(ctx))
    print('SMOOTHING_STUDY_READY')


def require_committed(project):
    prefix = prior.git(project, 'rev-parse', '--show-prefix')
    require(prefix.startswith('recognition/') and prefix != 'recognition/' and
            prior.git(project, 'branch', '--show-current') == 'topic-recognition', 'Wrong assignment location or branch.')
    for name in NEW_FILES:
        prior.git(project, 'ls-files', '--error-unmatch', '--', name)
    require(not prior.git(project, 'diff', 'HEAD', '--', *NEW_FILES),
            'Commit the three extension/plan/test files before training.')


def smoke(private):
    ctx = context(private)
    private, project, bundle, pipeline, reference, report, hashes = ctx
    folder, study = load_study(ctx)
    with v2.run_lock(folder):
        config = Config(**study['configurations']['S1'])
        device = choose_device(config.device)
        require(v2.environment(device) == study['reference_environment'], 'Environment differs from reference.')
        seed_everything(42)
        data = v2.WeightedDataset(bundle, config)
        batch = next(iter(DataLoader(data, batch_size=config.batch_size, shuffle=False, num_workers=0,
                                    generator=torch.Generator().manual_seed(42))))
        model = v2.make_model(config).to(device)
        optimizer = v2.make_optimizer(model, config)
        set_epoch(model, optimizer, config, 1)
        before = model.head.weight.detach().clone()
        reset_memory_peak(device)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.accumulation_steps):
            objective, hard = per_example_losses(model(batch['image'].to(device)),
                                                batch['label'].to(device), config.label_smoothing)
            (objective.mean()/config.accumulation_steps).backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        require(not torch.equal(before, model.head.weight.detach()), 'GPU smoke parameters did not update.')
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Non-finite model parameter.')
        model.eval(); x = batch['image'].to(device)
        with torch.inference_mode(): expected = model(x).cpu()
        path = folder/'PRIVATE_smoke.pt'
        torch.save({'model_state': v2.tensor_state(model), 'optimizer': optimizer.state_dict()}, path)
        value = torch.load(path, map_location='cpu', weights_only=True)
        clone = v2.make_model(config).to(device).eval()
        clone.load_state_dict(value['model_state'], strict=True)
        with torch.inference_mode(): actual = clone(x).cpu()
        require(torch.equal(expected, actual), 'Smoke checkpoint reload differs.')
        path.unlink()
        v2.atomic_json(folder/'smoke_summary.json', {
            'passed': True, 'study_sha256': prior.digest_json(study), 'source_sha256': source_hashes(),
            'environment': v2.environment(device), 'model_parameter_count': v2.parameter_count(model),
            'smoothed_gradient_accumulation_and_update_passed': True, 'checkpoint_reload_passed': True,
            'new_test_images_decoded': 0, 'gpu_memory': memory_info(device)})
    print('SMOOTHING_GPU_SMOKE_PASSED')


def verify_completed(run, config):
    report = read_json(run/'smoothing_summary.json')
    require(report.get('experiment_completed') is True and report.get('status') == 'COMPLETED',
            'Experiment incomplete. Rerun the SAME run command to resume.')
    require(report.get('configuration') == asdict(config) and report.get('pretrained') == {'used': False}
            and report.get('validation_targets_smoothed') is False
            and report.get('new_final_test_model_evaluation_performed') is False
            and report.get('new_test_images_decoded') == 0, 'Unexpected experiment settings or scope.')
    rows = prior.load_history(run/'history.csv')
    require(len(rows) == report['epochs_completed'], 'Epoch count differs.')
    chosen = max(rows, key=lambda r: (float(r['patient_accuracy']), float(r['patient_macro_f1']),
                                      -float(r['patient_log_loss'])))
    require(int(chosen['epoch']) == report['selected_epoch'], 'Selected checkpoint differs from declared rule.')
    m = report['selected_validation_metrics']['patient']
    for key, col in (('accuracy', 'patient_accuracy'), ('macro_f1', 'patient_macro_f1'),
                     ('log_loss', 'patient_log_loss'), ('auroc', 'patient_auroc')):
        require(math.isclose(m[key], float(chosen[col]), abs_tol=1e-10), 'History metric differs: '+key)
    for row in rows:
        require(math.isclose(float(row['learning_rate']), learning_rate(int(row['epoch']), config), abs_tol=1e-12),
                'Recorded learning-rate schedule differs.')
    predictions = read_json(run/'PRIVATE_validation_predictions.json')
    calculated = binary_metrics([r['label'] for r in predictions['patients']],
                                [r['p_ad'] for r in predictions['patients']])
    require(calculated == m, 'Saved patient metrics do not reproduce from saved probabilities.')
    hashes = prior.file_hashes(run, (*PUBLIC_RUN_FILES, *PRIVATE_RESULT_FILES))
    return report, rows, hashes


def run_experiment(private, experiment):
    require(experiment in EXPERIMENTS, 'Specify S1 or S2.')
    ctx = context(private)
    private, project, bundle, pipeline, reference, report, hashes = ctx
    folder, study = load_study(ctx)
    require_committed(project)
    config = Config(**study['configurations'][experiment])
    require(v2.environment(choose_device(config.device)) == study['reference_environment'], 'Environment changed.')
    with v2.run_lock(folder):
        smoke_report = read_json(folder/'smoke_summary.json')
        require(smoke_report.get('passed') is True and smoke_report.get('study_sha256') == prior.digest_json(study)
                and smoke_report.get('source_sha256') == source_hashes()
                and smoke_report.get('environment') == study['reference_environment'], 'Matching passed GPU smoke required.')
        run = folder/experiment
        run.mkdir(exist_ok=True)
        binding = {'configuration': asdict(config), 'study_sha256': prior.digest_json(study)}
        if (run/'run_metadata.json').exists():
            require(read_json(run/'run_metadata.json') == binding, 'Run binding changed.')
        else:
            require(not any(run.iterdir()), 'Existing run lacks binding; preserve it and report.')
            v2.atomic_json(run/'run_metadata.json', binding)
        receipt_path = run/'completion.json'
        if receipt_path.exists():
            r, rows, current = verify_completed(run, config)
            require(read_json(receipt_path) == {'study_sha256': prior.digest_json(study), 'files_sha256': current},
                    'Completed outputs changed.')
            print('SMOOTHING_ALREADY_COMPLETED:', experiment)
            return
        require(shutil.disk_usage(private).free >= 4*2**30, 'Keep at least 4 GiB free for recovery checkpoints.')
        resumable = any((run/name).exists() for name in ('PRIVATE_resume.pt', 'PRIVATE_previous.pt'))
        if not resumable:
            require(not (run/'history.csv').exists() and not (run/'smoothing_summary.json').exists(),
                    'Outputs exist but no resume state: preserve the run and report.')
        try:
            state = train_run(run, bundle, config, resume=resumable)
            require(state['completed'] is True, 'Run stopped before completion; rerun the SAME command.')
            r, rows, current = verify_completed(run, config)
            require(r['source_code_sha256'] == study['source_sha256']
                    and r['manifest_sha256'] == study['manifest_sha256']
                    and r['hardware'] == study['reference_environment'], 'Completed run provenance differs.')
            v2.atomic_json(receipt_path, {'study_sha256': prior.digest_json(study), 'files_sha256': current})
            print('SMOOTHING_EXPERIMENT_COMPLETED:', experiment)
        except BaseException:
            atomic_text(run/'PRIVATE_error.txt', traceback.format_exc())
            print('SMOOTHING_PAUSED: preserve the folder and rerun the SAME S1/S2 command.', file=sys.stderr)
            raise


def score_row(name, report):
    m = report['selected_validation_metrics']['patient']
    cm = m['confusion_matrix_rows_true_columns_predicted_NC_AD']
    return {'name': name, 'selected_epoch': report['selected_epoch'], 'epochs_completed': report['epochs_completed'],
            'patient_accuracy': m['accuracy'], 'correct_patients': cm[0][0]+cm[1][1], 'patients': m['n'],
            'macro_f1': m['macro_f1'], 'auroc': m['auroc'], 'patient_log_loss': m['log_loss'],
            'AD_recall': m['per_class']['AD']['recall'], 'NC_recall': m['per_class']['NC']['recall'],
            'AD_missed': cm[1][0], 'NC_false_positives': cm[0][1],
            'scan_accuracy': report['selected_validation_metrics']['scan']['accuracy'],
            'slice_accuracy': report['selected_validation_metrics']['slice']['accuracy'],
            'source_model_changed_automatically': False}


def promotion_screen(reference, candidate):
    same = candidate['patients'] == reference['patients']
    return same and candidate['correct_patients'] >= reference['correct_patients'] and candidate['AD_missed'] <= reference['AD_missed']


def comparison_markdown(summary):
    rows = summary['runs']
    text = ['## Controlled smoothing and learning-rate experiments', '',
            '**Validation/development evidence only; no new final-test inference.**', '',
            'A1 seed 42 was preserved. S1 adds uniform training-label smoothing 0.10. S2 additionally lowers '
            'the peak learning rate to 0.0001 and reaches the cosine floor at epoch 12, then holds it. '
            'All runs use the same patient partitions, scratch ConvNeXt-Tiny, A1 rotations, batch/accumulation, '
            'seed 42, 0.5 class cutoff and accuracy/F1/log-loss checkpoint rule. No patient weighting or pretraining.', '',
            '| Run | Selected epoch | Correct patients | Accuracy | Macro-F1 | AUROC | AD recall | Hard-label log loss |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f'| {r["name"]} | {r["selected_epoch"]} | {r["correct_patients"]}/{r["patients"]} | '
                    f'{r["patient_accuracy"]:.2%} | {r["macro_f1"]:.4f} | {r["auroc"]:.4f} | '
                    f'{r["AD_recall"]:.2%} | {r["patient_log_loss"]:.4f} |')
    text += ['', 'The predeclared promotion screen requires matching/exceeding A1 patient accuracy without '
             'detecting fewer AD patients. It is a development safeguard, not a course or clinical safety standard. '
             'Passing the screen is not an automatic replacement or a guarantee of generalisation.', '',
             'Candidates passing the screen: **'+(', '.join(summary['screen_passes']) or 'none')+'**.', '',
             'Smoothed training objectives differ from old unsmoothed losses. Validation metrics always use '
             'ordinary hard labels. S1 is a loss intervention; S2 changes two schedule components together. '
             'Repeated use of the small validation cohort and seed 42 limits claims; the test exposed in Version 1 '
             'has not been re-evaluated. A fresh/test score cannot be inferred from any validation peak.', '',
             'Previous reports, model checkpoints and data remain unchanged. Revised calibration, an explicitly '
             'disclosed evaluation procedure and selected-model integration still follow candidate review.', '',
             'ChatGPT assisted with implementation, tests and this draft. Record the code review and tests actually '
             'performed in the Artificial Intelligence Usage Disclosure.', '']
    return '\n'.join(text)


def compare(private):
    ctx = context(private)
    private, project, bundle, pipeline, reference, ref, hashes = ctx
    folder, study = load_study(ctx)
    with v2.run_lock(folder):
        reports = {'A1': ref}
        file_hashes = {}
        for name in EXPERIMENTS:
            run = folder/name
            report, history, current = verify_completed(run, Config(**study['configurations'][name]))
            require(read_json(run/'completion.json') == {'study_sha256': prior.digest_json(study), 'files_sha256': current},
                    'Completed run is not bound to this study.')
            reports[name] = report
            file_hashes[name] = current
        rows = [score_row(name, reports[name]) for name in ('A1', *EXPERIMENTS)]
        summary = {'comparison_completed': True, 'runs': rows,
                   'screen_passes': [r['name'] for r in rows[1:] if promotion_screen(rows[0], r)],
                   'candidate_changed_automatically': False, 'new_final_test_model_evaluation_performed': False,
                   'new_test_images_decoded': 0, 'test_accuracy_requirement_demonstrated': False,
                   'manifest_sha256': study['manifest_sha256'], 'reference_files_sha256': study['reference_files_sha256'],
                   'source_code_sha256': study['source_sha256'], 'configurations': study['configurations'],
                   'completed_file_sha256': file_hashes,
                   'limitation': 'Repeated selected validation results on the same patients; old test already exposed.'}
        out = folder/'comparison'; out.mkdir(exist_ok=True)
        v2.atomic_json(out/'smoothing_comparison.json', summary)
        atomic_text(out/'README_smoothing_comparison.md', comparison_markdown(summary))
        for name in EXPERIMENTS:
            for filename in ('smoothing_loss.png', 'smoothing_validation.png'):
                shutil.copyfile(folder/name/filename, out/f'{name}_{filename}')
        dest = private/'COMP3710_Smoothing_Results.zip'; temp = dest.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
            for filename in ('smoothing_comparison.json', 'README_smoothing_comparison.md'):
                z.write(out/filename, filename)
            for name in EXPERIMENTS:
                for filename in PUBLIC_RUN_FILES:
                    z.write(folder/name/filename, f'{name}/{filename}')
            for filename in ('v2_summary.json', 'history.csv'):
                z.write(reference/filename, f'A1/{filename}')
            z.write(folder/'smoke_summary.json', 'smoke_summary.json')
            z.write(project/'SMOOTHING_PLAN.md', 'SMOOTHING_PLAN.md')
        os.replace(temp, dest)
    print('SMOOTHING_COMPARISON_READY')
    print(dest)


def publish(private):
    ctx = context(private)
    private, project, bundle, pipeline, reference, ref, hashes = ctx
    folder, study = load_study(ctx)
    with v2.run_lock(folder):
        comparison = folder/'comparison'
        summary = read_json(comparison/'smoothing_comparison.json')
        require(summary.get('comparison_completed') is True and summary['source_code_sha256'] == source_hashes(),
                'A matching completed comparison is required.')
        for name in EXPERIMENTS:
            r, rows, current = verify_completed(folder/name, Config(**study['configurations'][name]))
            require(current == summary['completed_file_sha256'][name], 'Comparison is stale.')
        readme = project/'README.md'; original = readme.read_bytes()
        if MARKER.encode() in original:
            print('SMOOTHING_REPORT_ALREADY_PRESENT'); return
        backup = folder/('README_before_smoothing-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.md')
        backup.write_bytes(original)
        section = (comparison/'README_smoothing_comparison.md').read_text(encoding='utf-8')
        figures = []
        for name in EXPERIMENTS:
            for filename in ('smoothing_loss.png', 'smoothing_validation.png'):
                fname = f'{name}_{filename}'
                source = comparison/fname; dest = project/'figures'/fname
                require(not dest.exists() or sha256_file(dest) == sha256_file(source),
                        'A different report figure already exists: '+fname)
                figures.append((source, dest, fname))
        for source, dest, fname in figures:
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, dest)
            section += f'\n![{fname}](figures/{fname})\n'
        tmp = readme.with_name('README.md.smoothing-tmp')
        tmp.write_bytes(original + ('\n\n'+MARKER+'\n'+section+'\n').encode('utf-8'))
        os.replace(tmp, readme)
    print('SMOOTHING_REPORT_APPENDED')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'smoke', 'run', 'compare', 'publish'))
    parser.add_argument('--private-root', required=True, type=Path)
    parser.add_argument('--experiment', choices=EXPERIMENTS)
    args = parser.parse_args(argv)
    require((args.command == 'run') == (args.experiment is not None),
            'Use --experiment only with run; run requires S1 or S2.')
    actions = {'check': check, 'smoke': smoke, 'compare': compare, 'publish': publish}
    if args.command == 'run':
        run_experiment(args.private_root, args.experiment)
    else:
        actions[args.command](args.private_root)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f'SMOOTHING_STOPPED: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('Preserve previous work. No test inference is permitted here.', file=sys.stderr)
        sys.exit(1)
