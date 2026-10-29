"""Validation raw-score/NMS autopsy, not a GT-assisted prediction method."""
import csv,hashlib,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
S=Path(__file__).resolve().parent; R=S.parents[1]
sys.path.insert(0,str(S/'code'))
from evaluation import gt_record,score_records,inference_args
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops
from runtime import check_runtime
torch.set_num_threads(2)
OUT=R/'outputs/known_followup_diagnostics_20261022'
IMAGE=OUT/'images'
old=R/'studies/known_lesion_optimization_20261021'
decision=json.loads((old/'decision.json').read_text())
cfg=json.loads((R/decision['config_path']).read_text())
weights=R/decision['checkpoint_path']
assert hashlib.sha256(weights.read_bytes()).hexdigest()==decision['checkpoint_sha256']
manifest=R/'data/known_lesion_v1/val.csv'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==cfg['manifest_sha256']['val']
rows=list(csv.DictReader(manifest.open()));assert len(rows)==165
assert {p.stem for p in IMAGE.iterdir()}=={r['image_id'] for r in rows}
audit=json.loads((R/decision['evaluation_path']).parent.joinpath('inputs/staged/audit.json').read_text())
for row in rows:
    path=IMAGE/(row['image_id']+'.jpg')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['files'][row['image_id']]['file_sha256']

class RawPredictor(DetectionPredictor):
    def postprocess(self,preds,img,orig_imgs,**kwargs):
        output=preds[0] if isinstance(preds,(tuple,list)) else preds
        assert output.ndim==3 and output.shape[0]==1 and output.shape[1]==6
        raw=output[0].detach().clone().T
        assert raw.shape[1]==6
        raw=raw[raw[:,4:6].amax(1)>self.args.conf]
        boxes=ops.xywh2xyxy(raw[:,:4])
        shape=orig_imgs[0].shape
        boxes=ops.scale_boxes(img.shape[2:],boxes,shape)
        boxes/=torch.tensor([shape[1],shape[0],shape[1],shape[0]],device=boxes.device)
        self.raw_current={'boxes':boxes.cpu().numpy(),'scores':raw[:,4:6].cpu().numpy()}
        return super().postprocess(preds,img,orig_imgs,**kwargs)

def overlaps(boxes,gt):
    boxes=np.asarray(boxes).reshape(-1,4);gt=np.asarray(gt)
    inter=np.maximum(0,np.minimum(boxes[:,2:],gt[2:])-np.maximum(boxes[:,:2],gt[:2])).prod(1)
    area=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(1)
    return inter/np.maximum(area+np.maximum(0,gt[2:]-gt[:2]).prod()-inter,1e-15)

model=YOLO(str(weights)); records=[];details=[];arrays={}
for idx,row in enumerate(rows):
    gt=gt_record(row)
    result=model.predict(str(IMAGE/(row['image_id']+'.jpg')),predictor=RawPredictor,
        **inference_args(cfg,'cpu'))[0]
    preds=[{'class_id':int(c),'confidence':float(s),'box':b} for c,s,b in
           zip(result.boxes.cls.tolist(),result.boxes.conf.tolist(),result.boxes.xyxyn.tolist())]
    records.append({**gt,'predictions':preds})
    raw=model.predictor.raw_current; boxes=raw['boxes'];scores=raw['scores']
    arrays[row['image_id']]=np.c_[boxes,scores]
    ov=overlaps(boxes,gt['box']);loc=ov>=.5
    pre_class=scores.argmax(1)
    postov=overlaps([p['box'] for p in preds],gt['box'])
    d={**gt,'oracle_diagnostic_only':True,'raw_candidates_above_score_floor':len(boxes),
       'best_raw_IoU_ignoring_class':float(ov.max(initial=0)),
       'maximum_malignant_score_on_well_localized_raw_box':float(scores[loc,1].max(initial=0)),
       'maximum_malignant_score_on_well_localized_argmax_malignant_box':float(scores[loc&(pre_class==1),1].max(initial=0)),
       'maximum_malignant_score_on_well_localized_post_NMS_box':max((p['confidence'] for p,o in zip(preds,postov) if p['class_id']==1 and o>=.5),default=0.)}
    for t in (.001,.4):
        raw_m=d['maximum_malignant_score_on_well_localized_raw_box']>t
        argmax_m=d['maximum_malignant_score_on_well_localized_argmax_malignant_box']>t
        post_m=d['maximum_malignant_score_on_well_localized_post_NMS_box']>t
        d[f'malignant_chain_{t}']=('post_NMS_TP_available' if post_m else
            'lost_between_argmax_and_NMS' if argmax_m else 'loses_class_argmax' if raw_m else
            'no_well_localized_raw_box' if not loc.any() else 'malignant_score_below_threshold')
    details.append(d)
    if (idx+1)%25==0:print(f'validation {idx+1}/165',flush=True)
np.savez_compressed(OUT/'raw_two_class_scores.npz',**arrays)
cached=json.loads((R/decision['evaluation_path']/'predictions.json').read_text())
cpu=score_records(records,.4);gpu=score_records(cached,.4)
same_counts=all(cpu['classes'][c][k]==gpu['classes'][c][k]
               for c in ('Benign','Malignant') for k in ('tp','fp','fn'))
summary={'scope':'validation only','checkpoint_sha256':decision['checkpoint_sha256'],
    'runtime':check_runtime(),'device':'CPU','image_count':165,'new_reassessment_inference':False,
    'cpu_GPU_cached_operating_counts_match':same_counts,'CPU_operating_point':cpu,
    'malignant_raw_chain':{str(t):dict(Counter(d[f'malignant_chain_{t}'] for d in details if d['class_id']==1)) for t in (.001,.4)},
    'GT_oracle_warning':'Raw best-IoU and well-localized candidate statistics use GT solely for autopsy; not deployable predictions or model performance',
    'details':details}
for name,data in [('raw_diagnosis.json',summary),('cpu_predictions.json',records)]:
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('details','CPU_operating_point')},indent=2))
