"""Read only this new study's run records and epoch CSVs."""
import csv,json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
status={}
for run in sorted((R/'outputs').glob('known_lesion_followup_20261022_*')):
    found=False
    for file in run.glob('*/locked_config.json'):
        found=True;p=file.parent;name=p.name
        model='s' if name.startswith('s_') else 'n'
        item={'run':str(run.relative_to(R)),'phase':'input_audit','model':model}
        result=p/f'train_{model}/fit/results.csv'
        if result.exists():
            rows=list(csv.DictReader(result.open()))
            if rows:
                row={k.strip():v.strip() for k,v in rows[-1].items()}
                item.update(phase='training',epoch=int(row['epoch']),epoch_budget=60)
        record=p/f'train_{model}/run.json'
        if record.exists():
            x=json.loads(record.read_text());item['training_status']=x['status']
            if x['status']=='completed':item['phase']='validation'
            if x['status']=='failed':item.update(phase='failed',error=x.get('error'))
        if (p/f'val_{model}/evaluation.json').exists():item['phase']='completed'
        status[name]=item
    if not found:status[run.name]={'run':str(run.relative_to(R)),'phase':'environment_setup'}
print(json.dumps(status,indent=2))
