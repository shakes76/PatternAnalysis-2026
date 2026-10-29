"""A GT-free, single-lesion operating rule for this one-GT-per-image cohort.

This ratio is a relative model score, not a calibrated malignancy probability.
No NMS is needed because exactly one raw candidate is retained.
"""
import numpy as np

def one_prediction(raw, ratio_threshold):
    a=np.asarray(raw)
    if a.size==0:return []
    if a.ndim!=2 or a.shape[1]!=6 or not np.isfinite(a).all():
        raise ValueError('Expected finite normalized xyxy plus two scores')
    if not 0<=ratio_threshold<=1 or (a[:,4:]<0).any() or (a[:,4:]>1).any():
        raise ValueError('Invalid score/threshold')
    # Stable tie: first upstream raw candidate index.
    p=a[int(np.argmax(np.max(a[:,4:6],axis=1)))]
    total=float(p[4]+p[5])
    if total<=0:return []
    ratio=float(p[5])/total
    c=int(ratio>ratio_threshold)
    return [{'class_id':c,'confidence':ratio if c else 1-ratio,'box':p[:4].tolist()}]
