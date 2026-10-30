"""Inspect saved validation predictions; no training or threshold optimization."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from evaluate import box_iou, operating_metrics
from modules import CLASS_NAMES
from utils import read_config, save_json, sha256


def classify_case(predictions, target, threshold):
    boxes = sorted((p for p in predictions if p['confidence'] >= threshold),
                   key=lambda p: p['confidence'], reverse=True)
    same = [(box_iou(p['xyxy'], target['xyxy']), p) for p in boxes
            if p['class_id'] == target['class_id']]
    valid = [(iou, p) for iou, p in same if iou >= 0.5]
    wrong = [box_iou(p['xyxy'], target['xyxy']) for p in boxes
             if p['class_id'] != target['class_id']]
    if valid:
        # One target: the highest-confidence qualifying prediction matches it.
        matched_iou = valid[0][0]
        status = 'correct' if matched_iou >= 0.8 else 'imprecise_localization'
        if status == 'correct' and len(boxes) > 1:
            status = 'correct_with_false_positives'
    else:
        matched_iou = None
        if not boxes:
            status = 'no_detection'
        elif max(wrong, default=0.0) >= 0.5:
            status = 'classification_error'
        elif max((item[0] for item in same), default=0.0) > 0:
            status = 'localization_error'
        else:
            status = 'missed_with_false_positives'
    return {'status': status, 'matched_iou': matched_iou, 'target': target,
            'detections': boxes, 'false_positives_at_iou05': len(boxes) - int(bool(valid)),
            'false_negative_at_iou05': not bool(valid)}


def draw_case(image_path, case, output):
    with Image.open(image_path) as original:
        image = original.convert('RGB')
    scale = min(900/image.width, 600/image.height)
    canvas_image = image.resize((round(image.width*scale), round(image.height*scale)))
    canvas = Image.new('RGB', (max(canvas_image.width, 600), canvas_image.height+95), 'white')
    canvas.paste(canvas_image, (0, 95))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{case['image_id']} | {case['status']}", fill='black')
    draw.text((8, 27), 'Green = ground truth; red = prediction; labels include confidence', fill='black')
    draw.text((8, 46), f"GT: {CLASS_NAMES[case['target']['class_id']]} | "
              f"Shown {min(6,len(case['detections']))}/{len(case['detections'])} predictions", fill='black')
    draw.text((8, 65), 'Geometric error category only; visual cause requires human review.', fill='black')

    def rectangle(box, color):
        x1, y1, x2, y2 = box
        coords = (x1*scale, y1*scale+95, x2*scale, y2*scale+95)
        draw.rectangle(coords, outline=color, width=3)
        return coords

    rectangle(case['target']['xyxy'], 'lime')
    for p in case['detections'][:6]:
        coords = rectangle(p['xyxy'], 'red')
        draw.text((coords[0]+2, max(95, coords[1]-12)),
                  f"{CLASS_NAMES[p['class_id']]} {p['confidence']:.2f}", fill='red', stroke_width=1, stroke_fill='white')
    canvas.save(output, quality=90)


def analyze(data_yaml, predictions_path, output, threshold=0.15, limit=5):
    if not 0 <= threshold <= 1 or limit < 1:
        raise ValueError('Invalid threshold or case limit')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    data_yaml = Path(data_yaml).resolve()
    data = read_config(data_yaml)
    names = data['names']
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if tuple(names) != CLASS_NAMES:
        raise ValueError('Dataset class order mismatch')
    root = Path(data.get('path', data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    images = {p.stem: p for p in (root / data['val']).iterdir()
              if p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
    predictions = json.loads(Path(predictions_path).read_text())
    if len(predictions) != len(images) or {p['image_id'] for p in predictions} != set(images):
        raise ValueError('Saved predictions must cover exactly the validation image inventory')
    truth, cases = [], []
    for row in predictions:
        image_id = row['image_id']
        with Image.open(images[image_id]) as image:
            width, height = image.size
        lines = (root / 'labels/val' / f'{image_id}.txt').read_text().splitlines()
        if len(lines) != 1:
            raise ValueError('This case analyzer expects exactly one annotated lesion per image')
        cls, x, y, w, h = map(float, lines[0].split())
        target = {'class_id': int(cls), 'xyxy': [(x-w/2)*width, (y-h/2)*height,
                                               (x+w/2)*width, (y+h/2)*height]}
        truth.append({'image_id': image_id, 'detections': [target]})
        case = classify_case(row['detections'], target, threshold)
        case['image_id'] = image_id
        cases.append(case)
    metrics05 = operating_metrics(predictions, truth, threshold, 0.5)
    metrics08 = operating_metrics(predictions, truth, threshold, 0.8)
    candidates = sorted((c for c in cases if c['status'] != 'correct'),
                        key=lambda c: (c['target']['class_id'] != 0, c['image_id']))
    selected, statuses = [], set()
    for case in candidates:
        if case['status'] not in statuses and len(selected) < limit:
            selected.append(case)
            statuses.add(case['status'])
    chosen_ids = {c['image_id'] for c in selected}
    selected.extend(c for c in candidates if c['image_id'] not in chosen_ids)
    selected = selected[:limit]
    output.mkdir(parents=True)
    for case in selected:
        draw_case(images[case['image_id']], case, output / f"{case['image_id']}.jpg")
    report = {'split': 'val', 'confidence': threshold,
              'predictions_sha256': sha256(predictions_path), 'data_yaml_sha256': sha256(data_yaml),
              'operating_iou05': metrics05, 'operating_iou08': metrics08,
              'case_counts': dict(Counter(c['status'] for c in cases)),
              'candidate_selection': 'one per available error category, then fill; melanoma first, sorted image IDs',
              'notes': ['Exploratory validation analysis, not a test benchmark.',
                        'Categories are geometric symptoms, not verified causal explanations.',
                        'At most six highest-confidence boxes are drawn; JSON retains all.'],
              'selected_cases': selected}
    save_json(output / 'failure_summary.json', report)
    save_json(output / 'all_cases.json', cases)
    text = ['# Validation failure candidates', '', f'Confidence: {threshold}. Green: GT; red: predictions.', '',
            'These are automatically selected candidates. Review images before explaining causes.', '']
    for case in selected:
        text.extend([f"## {case['image_id']}: {case['status']}", '',
                     f"![{case['image_id']}]({case['image_id']}.jpg)", '',
                     'Visual observation: [complete after inspecting the image]', '',
                     'Possible cause and evidence: [complete; do not assert an untested cause]', ''])
    (output / 'case_review.md').write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps({'images': len(cases), 'case_counts': report['case_counts'],
                      'melanoma_iou05': metrics05['per_class']['melanoma'],
                      'melanoma_iou08': metrics08['per_class']['melanoma'],
                      'selected': [c['image_id'] for c in selected], 'output': str(output)}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conf', type=float, default=0.15)
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()
    analyze(args.data, args.predictions, args.output, args.conf, args.limit)
