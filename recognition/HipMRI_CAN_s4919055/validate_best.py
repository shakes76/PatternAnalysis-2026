"""Recheck a fixed best checkpoint on validation only; never train or open test.

Save checkpoint provenance, all six class metrics and three low-scoring
validation examples from different patient groups. These are development
diagnostics, not a final held-out benchmark or a controlled timing study.
"""

import argparse
import math
import os
from pathlib import Path
import socket

import torch

from dataset import DEFAULT_DATA_ROOT, HipMRIVolumeDataset
from experiment import (evaluate_cases, file_sha256, get_device, load_checkpoint,
                        predict_volume, save_json, seed_everything)
from predict import save_panels


def low_scoring_patients(cases, count=3):
    """One example per patient; select by existing whole-volume prostate Dice."""
    selected, seen = [], set()
    ordered = sorted((case for case in cases if case['dice'][5] is not None),
                     key=lambda case: (case['dice'][5], case['case_id']))
    for case in ordered:
        if case['patient_id'] not in seen:
            selected.append(case)
            seen.add(case['patient_id'])
            if len(selected) == count:
                break
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('runs/split_seed42.json'))
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if file_sha256(args.checkpoint) != args.expected_sha256:
        raise ValueError('Checkpoint differs from the inspected preview checkpoint.')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    device = get_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint['config']
    if config['mode'] != 'fit' or config['manifest_sha256'] != file_sha256(args.manifest):
        raise ValueError('Validation requires a fitted model and its exact saved manifest.')
    selected_score = checkpoint['history'][-1]['prostate_dice']
    if not math.isclose(selected_score, checkpoint['best_validation_prostate'], abs_tol=1e-12):
        raise ValueError('Checkpoint is not a validation-selected best model.')
    seed_everything(config['seed'])
    data = HipMRIVolumeDataset(args.manifest, 'validation', args.data_root,
                              tuple(config['shape_dhw']))
    args.output.mkdir(parents=True, exist_ok=False)
    report = evaluate_cases(model, data, device, config['batch_size'],
                            loss_kind=config.get('loss_kind', 'soft-dice'))
    score = report['patient_macro_dice'][5]
    matched = score is not None and math.isclose(score, selected_score, rel_tol=0, abs_tol=1e-6)
    examples = low_scoring_patients(report['cases'])
    report.update(split='validation', checkpoint_epoch=checkpoint['epoch'],
                  checkpoint_sha256=args.expected_sha256,
                  manifest_sha256=config['manifest_sha256'],
                  expected_validation_prostate_dice=selected_score,
                  matches_recorded_validation_score=matched,
                  device=str(device), hostname=socket.gethostname(),
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  gpu_name=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                  torch_version=str(torch.__version__),
                  validation_source_sha256={name: file_sha256(Path(__file__).with_name(name))
                      for name in ['validate_best.py','experiment.py','dataset.py','modules.py','predict.py']},
                  training_config=config, selected_examples=examples,
                  example_selection='Lowest prostate Dice, one scan per distinct validation patient; not test data')
    save_json(args.output/'validation_best.json', report)
    for case in examples:
        index = next(i for i, pair in enumerate(data.pairs) if pair['case_id'] == case['case_id'])
        sample = data[index]
        prediction = predict_volume(model, sample['image'], device, config['batch_size']).argmax(0)
        save_panels(sample, prediction, args.output/f'{case["case_id"]}_prostate.png')
    print(f'Best checkpoint epoch={checkpoint["epoch"]}, validation prostate Dice={score:.6f}', flush=True)
    print('BEST_VALIDATION_CHECK_PASSED' if matched else 'BEST_VALIDATION_SCORE_MISMATCH', flush=True)
    print('Validation only; no training updates or final test cases were used.', flush=True)
    return 0 if matched else 1


if __name__ == '__main__':
    raise SystemExit(main())
