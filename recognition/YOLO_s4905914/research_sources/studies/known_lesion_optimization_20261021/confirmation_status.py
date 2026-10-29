"""Compact progress; reads only this study's logs and training/val records."""
from pathlib import Path
import csv,json
R=Path(__file__).resolve().parents[2]
status={}
for run in sorted((R/'outputs').glob('known_lesion_confirmation_20261021_*')):
    for cfg in run.glob('n_*/locked_config.json'):
        p=cfg.parent;n=p.name
        item={'run':str(run.relative_to(R)),'phase':'input_audit'}
        result=p/'train_n/fit/results.csv'
        if result.exists():
            with result.open() as f: rows=list(csv.DictReader(f))
            if rows:
                last={k.strip():v.strip() for k,v in rows[-1].items()}
                item.update(phase='training',epoch=int(last['epoch']),epoch_budget=json.loads(cfg.read_text())['train']['epochs'])
        trained=p/'train_n/run.json'
        if trained.exists():
            rec=json.loads(trained.read_text());item['training_status']=rec['status']
            if rec['status']=='completed': item['phase']='validation'
            if rec['status']=='failed':item.update(phase='failed',error=rec.get('error'))
        if (p/'val_n/evaluation.json').exists():item['phase']='completed'
        status[n]=item
print(json.dumps(status,indent=2))
