"""Independent NumPy audit of added cached metrics (no inference/selection)."""
import json
from pathlib import Path
import numpy as np


def close(x, y):
    if x is None or y is None:
        assert x is y, (x, y)
    else:
        assert np.isclose(x, y, rtol=0, atol=1e-12), (x, y)


def audit(root):
    code = Path(__file__).resolve().parent
    result = json.loads((code/'course_metrics_20261022.json').read_text())
    verified = []
    for key, r in result['reports'].items():
        rows = json.loads((root/r['evaluation_path']/'predictions.json').read_text())
        truth = np.array([x['class_id'] for x in rows])
        predicted = []
        overlaps = {k: [] for k in ['all_gt_class_correct','all_gt_class_agnostic',
                                    'primary_box_ignore_class','primary_box_correct_class']}
        for row in rows:
            ps = sorted([p for p in row['predictions'] if p['confidence'] > r['threshold']],
                        key=lambda p: (-p['confidence'],p['class_id'],*p['box']))
            predicted.append(ps[0]['class_id'] if ps else 2)
            if ps:
                boxes = np.array([p['box'] for p in ps]); gt = np.array(row['box'])
                intersection = np.maximum(0, np.minimum(boxes[:,2:],gt[2:]) - np.maximum(boxes[:,:2],gt[:2])).prod(axis=1)
                union = (boxes[:,2:]-boxes[:,:2]).prod(axis=1)+(gt[2:]-gt[:2]).prod()-intersection
                values = np.divide(intersection,union,out=np.zeros_like(union),where=union>0)
                same = np.array([p['class_id']==row['class_id'] for p in ps])
                actual = [values[same].max(initial=0),values.max(initial=0),values[0],values[0] if same[0] else 0]
            else:
                actual = [0,0,0,0]
            for k, value in zip(overlaps,actual):overlaps[k].append(value)
        predicted = np.array(predicted)
        c = r['metrics']['classification']
        close((truth==predicted).mean(), c['accuracy'])
        close((predicted!=2).mean(), c['coverage'])
        matrix = [[int(((truth==g)&(predicted==p)).sum()) for p in (0,1,2)] for g in (0,1)]
        assert matrix == c['confusion_matrix']
        f1s=[]
        for i, name in enumerate(('Benign','Malignant')):
            tp=int(((truth==i)&(predicted==i)).sum());fp=int(((truth!=i)&(predicted==i)).sum());fn=int(((truth==i)&(predicted!=i)).sum())
            f1=2*tp/(2*tp+fp+fn);f1s.append(f1)
            close(f1,c['classes'][name]['f1']);close(tp/(tp+fn),c['classes'][name]['recall'])
        close(np.mean(f1s),c['macro_f1'])
        for kind, values in overlaps.items():
            values=np.array(values);loc=r['metrics']['localization'][kind]
            close(float(values.mean()),loc['overall']['mean'])
            assert loc['overall']['denominator']==len(values)
            for i,name in enumerate(('Benign','Malignant')):
                close(float(values[truth==i].mean()),loc['per_class'][name]['mean'])
        verified.append(key)
    return {'status':'passed','independent_implementation':'NumPy boolean classification and vectorized box overlap',
            'runs':verified,'new_inference':False,'new_selection':False}


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.root.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
