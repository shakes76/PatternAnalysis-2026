"""Summarize localization from fixed saved predictions; no inference or tuning."""

import argparse
import json
from pathlib import Path
from PIL import Image

from evaluate import box_iou
from modules import CLASS_NAMES
from utils import read_config, save_json, sha256


def summarize(data_yaml, predictions, output, confidence=0.15):
    data_yaml = Path(data_yaml).resolve()
    data = read_config(data_yaml)
    root = Path(data['path'])
    if not root.is_absolute():
        root = data_yaml.parent/root
    images = {p.stem:p for p in (root/data['test']).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')}
    rows = json.loads(Path(predictions).read_text(encoding='utf-8'))
    if len(rows)!=len(images) or {r['image_id'] for r in rows}!=set(images):
        raise ValueError('Test inventory differs from saved predictions')
    stats = {name:{'ground_truth':0,'matched':0,'iou_sum':0.0} for name in CLASS_NAMES}
    for row in rows:
        image_id=row['image_id']
        with Image.open(images[image_id]) as image:
            width,height=image.size
        targets=[]
        for line in (root/'labels/test'/f'{image_id}.txt').read_text().splitlines():
            cls,x,y,w,h=map(float,line.split())
            targets.append({'class_id':int(cls),'xyxy':[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]})
        for gt in targets:
            stats[CLASS_NAMES[gt['class_id']]]['ground_truth']+=1
        used=set()
        for p in sorted(row['detections'],key=lambda x:x['confidence'],reverse=True):
            if p['confidence']<confidence:
                continue
            options=[(box_iou(p['xyxy'],g['xyxy']),i) for i,g in enumerate(targets)
                     if i not in used and g['class_id']==p['class_id']]
            overlap,index=max(options,default=(0.0,-1))
            if index<0 or overlap<0.5:
                continue
            used.add(index)
            item=stats[CLASS_NAMES[p['class_id']]]
            item['matched']+=1
            item['iou_sum']+=overlap
    total={key:sum(r[key] for r in stats.values()) for key in ['ground_truth','matched','iou_sum']}
    def finish(row):
        return {**row,'mean_iou_matched':row['iou_sum']/row['matched'] if row['matched'] else None,
                'mean_iou_all_gt_zero_for_misses':row['iou_sum']/row['ground_truth'] if row['ground_truth'] else None,
                'recall_iou05':row['matched']/row['ground_truth'] if row['ground_truth'] else None}
    result={'confidence':confidence,'source_predictions_sha256':sha256(predictions),
            'matching':'class-aware, confidence-ordered one-to-one; minimum matching IoU=0.5',
            'mean_definition':'mean_iou_matched is conditional on successful detections; all-GT version assigns zero to misses',
            'per_class':{name:finish(r) for name,r in stats.items()},'overall':finish(total)}
    if Path(output).exists():
        raise FileExistsError(output)
    save_json(output,result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--conf',type=float,default=0.15)
    args=parser.parse_args()
    if not 0<=args.conf<=1:
        parser.error('Confidence must be between 0 and 1')
    summarize(args.data,args.predictions,args.output,args.conf)
