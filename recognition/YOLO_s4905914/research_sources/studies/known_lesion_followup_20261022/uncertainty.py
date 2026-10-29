"""Conditional paired component bootstrap; does not undo validation tuning."""
import csv,importlib.util,json
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[1]
spec=importlib.util.spec_from_file_location('previous_comparison',R/'studies/known_lesion_optimization_20261021/compare_validation.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
read=lambda p:json.loads(p.read_text())
results=read(S/'results.json');rows=list(csv.DictReader((R/'data/known_lesion_v1/val.csv').open()))
base=results['reports']['reference_geometry']
bp=read(R/base['evaluation_path']/'predictions.json')
b=previous.row_metrics(bp,base['selected']['threshold']);out={}
for name,r in results['reports'].items():
    if name=='reference_geometry' or not r['selected']:continue
    records=read(R/r['evaluation_path']/'predictions.json')
    out[name]=previous.cluster_ci(b,previous.row_metrics(records,r['selected']['threshold']),rows)
policy=read(S/'policy_results_seed3710.json')
out['single_lesion_policy']=previous.cluster_ci(b,previous.row_metrics(read(S/'policy_predictions_seed3710.json'),0.),rows)
(S/'conditional_uncertainty.json').write_text(json.dumps(out,indent=2)+'\n')
print('Conditional intervals for',list(out))
