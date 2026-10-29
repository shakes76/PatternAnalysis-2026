"""One final, fixed retrospective evaluation of the retained geometry candidate.

This separate closeout command does not relax the train/val-only study loaders.
It cannot train, scan thresholds or select a replacement. Prior image exposure
is retained in the output status. Run only after restoring the exact images.
"""
import argparse
import csv
import math
from pathlib import Path

from classification_metrics import summarize_predictions
from evaluation import collect_predictions, gt_record, inference_args, validator_class
from modules import load_trained
from runtime import check_runtime, digest, load_config, read_json, write_json


def evaluate(root, image_root, output, device, verify_only=False):
    code = Path(__file__).resolve().parent
    plan_path = code/'course_closeout_plan.json'
    plan = read_json(plan_path)
    for name, expected in plan['reporting_code_sha256'].items():
        if digest(code/name) != expected:
            raise ValueError('Reporting implementation changed after closeout plan: '+name)
    chosen = plan['selected_model']
    config = root/chosen['config_path']
    checkpoint = root/chosen['checkpoint_path']
    lock_path = root/chosen['evaluation_path']/'selection_lock.json'
    cfg = load_config(config)
    lock = read_json(lock_path)
    for path, expected in plan['selection_records_sha256'].items():
        if digest(root/path) != expected:
            raise ValueError('Selection changed after closeout plan: '+path)
    if (digest(checkpoint) != chosen['checkpoint_sha256'] or
            lock['checkpoint_sha256'] != digest(checkpoint) or
            lock['config_sha256'] != digest(config) or lock['threshold'] != .4):
        raise ValueError('Use only the fixed geometry checkpoint, config and threshold')
    if cfg['train']['imgsz'] != 640 or chosen['confidence'] != .4:
        raise ValueError('Final protocol differs from the selected candidate')
    manifest = root/'data/known_lesion_v1/reassessment.csv'
    if digest(manifest) != plan['manifest_sha256']['reassessment']:
        raise ValueError('Reassessment membership changed')
    rows = list(csv.DictReader(manifest.open()))
    if len(rows) != 165 or len({r['image_id'] for r in rows}) != 165:
        raise ValueError('Unexpected cohort')
    source_audit = read_json(root/'outputs/provisional_20261017_669143/inputs/staged/audit.json')
    files = {}
    for row in rows:
        image = image_root/(row['image_id']+'.jpg')
        expected = source_audit['files'][row['image_id']]['file_sha256']
        if digest(image) != expected:
            raise ValueError('Photo is not the audited school input: '+row['image_id'])
        g = gt_record(row)
        if not all(math.isfinite(v) for v in g['box']):
            raise ValueError('Invalid GT')
        files[row['image_id']] = {'file_sha256': expected}
    if output.exists():
        raise ValueError('Choose a fresh output directory; results must not be overwritten')
    if verify_only:
        print('Verified fixed selection, reporting code and all 165 original photos; no inference started.')
        return
    runtime = check_runtime()
    import yaml
    output.mkdir(parents=True)
    stage = output/'inputs'
    images = stage/'reassessment/images'; labels = stage/'reassessment/labels'
    images.mkdir(parents=True); labels.mkdir(parents=True)
    for row in rows:
        identifier = row['image_id']
        (images/(identifier+'.jpg')).symlink_to((image_root/(identifier+'.jpg')).resolve())
        label = ' '.join([row['class_id'], *[row[k] for k in ('center_x','center_y','box_width','box_height')]])
        (labels/(identifier+'.txt')).write_text(label+'\n')
    dataset = stage/'reassessment.yaml'
    dataset.write_text(yaml.safe_dump({'path': str(stage.resolve()), 'train': 'reassessment/images',
        'val': 'reassessment/images', 'names': {0: 'Benign', 1: 'Malignant'}, 'nc': 2}))
    write_json(output/'started.json', {'plan_sha256': digest(plan_path), 'device': device,
        'runtime': runtime, 'threshold': .4, 'checkpoint_sha256': digest(checkpoint),
        'manifest_sha256': digest(manifest), 'files': files,
        'source_sha256': {n: digest(code/n) for n in ['evaluate_locked_candidate.py','classification_metrics.py','evaluation.py','preprocessing.py']},
        'fresh_blind_test': False, 'new_training': False, 'threshold_search': False})
    model = load_trained(checkpoint)
    ap = model.val(validator=validator_class(), data=str(dataset.resolve()), batch=16,
        workers=0, project=str(output.resolve()), name='ap', plots=True, **inference_args(cfg, device))
    records, timing = collect_predictions(model, rows, stage/'reassessment', cfg, device)
    write_json(output/'predictions.json', records)
    report = summarize_predictions(records, .4, .5)
    write_json(output/'evaluation.json', {'status': 'completed', 'split': 'reassessment',
        'fresh_blind_test': False, 'reporting_plan_sha256': digest(plan_path),
        'checkpoint_sha256': digest(checkpoint), 'config_sha256': digest(config),
        'selection_lock_sha256': digest(lock_path), 'manifest_sha256': digest(manifest),
        'predictions_sha256': digest(output/'predictions.json'), 'metrics': report,
        'operating_point': report['detection'], 'threshold_search': False, 'selection_changed': False,
        'runtime': runtime, 'device': device,
        'ap': {str(int(c)): {'name': model.names[int(c)], 'AP50': float(ap.box.ap50[i]),
                           'AP50_95': float(ap.box.ap[i])} for i,c in enumerate(ap.box.ap_class_index)},
        'mAP50': float(ap.box.map50), 'mAP50_95': float(ap.box.map),
        'library_speed_ms': timing, 'timing_scope': 'Library stage timings only; not full application latency'})
    print('Completed one fixed retrospective evaluation; selection remains unchanged.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--verify-only', action='store_true', help='Check inputs without creating outputs or running inference')
    a = p.parse_args()
    evaluate(a.root.resolve(), a.image_root.resolve(), a.output.resolve(), a.device, a.verify_only)
