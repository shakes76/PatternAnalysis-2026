"""R1 opt-in selection for future experiments; V1 runners remain unchanged.

Consume the existing validation forward's rows, never load images or run a
model. Primary: patient mean-probability NLL. Auxiliary: unweighted slice CE.
Strict improvement wins, so ties retain the earliest epoch. Call with an eval
model after each contiguous epoch. Output directories must be new.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from patient_evaluation import POLICY, evaluate_patients


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copied(value):
    return json.loads(json.dumps(value, allow_nan=False))


def write_json(path, value):
    temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class DualCheckpointSelector:
    def __init__(self, output_dir, config, provenance, expected_slices=20):
        self.folder = Path(output_dir)
        self.config, self.provenance = copied(config), copied(provenance)
        if not isinstance(expected_slices, int) or expected_slices < 1:
            raise ValueError('expected_slices must be positive')
        self.expected_slices = expected_slices
        self.folder.mkdir(parents=True, exist_ok=False)
        self.summary = dict(status='ready', config=self.config, provenance=self.provenance,
                            policy=copied(POLICY), expected_slices=expected_slices,
                            best={}, history=[], additional_image_reads=0)
        self._save_summary()

    def _save_summary(self):
        write_json(self.folder / 'selection_summary.json', self.summary)

    def consider(self, epoch, model, validation_rows, expected_records, slice_ce):
        try:
            if self.summary['status'] == 'failed':
                raise ValueError('failed selection cannot be resumed')
            if type(epoch) is not int or epoch != len(self.summary['history']) + 1:
                raise ValueError('epochs must be contiguous starting at 1')
            if not math.isfinite(slice_ce) or slice_ce < 0:
                raise ValueError('slice CE must be finite and nonnegative')
            if model.training:
                raise ValueError('selection requires an eval model')
            expected_records = list(expected_records)
            if not expected_records or any(r.get('split') != 'validation' for r in expected_records):
                raise ValueError('only validation records may select weights')
            evaluation = evaluate_patients(validation_rows, expected_records, self.expected_slices)
            cohort = sorted((r['file_path'], r['patient_id'], r['image_id'], str(r['label'])) for r in expected_records)
            cohort_hash = hashlib.sha256(json.dumps(cohort).encode()).hexdigest()
            if self.summary.get('validation_records_sha256', cohort_hash) != cohort_hash:
                raise ValueError('validation cohort changed between epochs')
            self.summary['validation_records_sha256'] = cohort_hash
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if any(not torch.isfinite(v).all().item() for v in state.values()):
                raise ValueError('model state must be finite')
            values = dict(patient_nll=evaluation['metrics']['nll'], slice_ce=float(slice_ce))
            for rule, value in values.items():
                old = self.summary['best'].get(rule)
                if old is None or value < old['value']:
                    path = self.folder / ('best_' + rule + '.pt')
                    checkpoint = dict(format_version=1, epoch=epoch, selection_rule=rule,
                                      selection_value=value, config=self.config,
                                      provenance=self.provenance, policy=copied(POLICY),
                                      expected_slices=self.expected_slices, validation_records_sha256=cohort_hash, model_state_dict=state)
                    temporary = path.with_suffix('.pt.partial')
                    torch.save(checkpoint, temporary)
                    temporary.replace(path)
                    self.summary['best'][rule] = dict(epoch=epoch, value=value,
                                                      file=path.name, sha256=digest(path))
            self.summary['history'].append(dict(epoch=epoch, **values,
                                                 patient_metrics=evaluation['metrics']))
            self.summary['status'] = 'passed'
            self._save_summary()
            return evaluation
        except Exception as exc:
            self.summary['status'] = 'failed'
            self.summary['error'] = f'{type(exc).__name__}: {exc}'
            self._save_summary()
            raise


def load_selected(output_dir, rule, model_factory, expected_config, expected_provenance):
    """Verify file hash and selection metadata, strictly load without consuming RNG."""
    folder = Path(output_dir)
    summary = json.loads((folder / 'selection_summary.json').read_text())
    if summary.get('status') != 'passed':
        raise ValueError('selection run has not passed; failed artifacts are for diagnosis only')
    if rule not in ('patient_nll', 'slice_ce'):
        raise ValueError('unknown selection rule')
    if summary['config'] != expected_config or summary['provenance'] != expected_provenance:
        raise ValueError('configuration or provenance mismatch')
    best = summary['best'][rule]
    path = folder / ('best_' + rule + '.pt')
    if best['file'] != path.name or digest(path) != best['sha256']:
        raise ValueError('checkpoint hash/path mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    expected = dict(format_version=1, epoch=best['epoch'], selection_rule=rule,
                    selection_value=best['value'], config=expected_config,
                    provenance=expected_provenance, policy=POLICY,
                    expected_slices=summary['expected_slices'],
                    validation_records_sha256=summary['validation_records_sha256'])
    if any(checkpoint.get(k) != v for k, v in expected.items()):
        raise ValueError('checkpoint metadata mismatch')
    history_best = min(summary['history'], key=lambda r: (r[rule], r['epoch']))
    if history_best['epoch'] != best['epoch'] or history_best[rule] != best['value']:
        raise ValueError('checkpoint does not match earliest minimum')
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        model = model_factory()
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.eval()
    return model, checkpoint
