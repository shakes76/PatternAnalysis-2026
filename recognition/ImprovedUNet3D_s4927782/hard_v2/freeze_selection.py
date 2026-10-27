"""Freeze BOTH completed v2 checkpoints before any new TEST pixels are opened."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import torch
import v2_pipeline as v2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    plan=v2.load_protocol(a.protocol)
    v2.verify_review_sources(plan,a.protocol.parent)
    result=dict(status='frozen',created_at_utc=datetime.now(timezone.utc).isoformat(),
                manifest_sha256=v2.sha256(a.manifest),protocol_sha256=v2.sha256(a.protocol),
                interpretation='Validation-selected fixed checkpoints; later TEST is previously observed and retrospective.',models={})
    for name in v2.MODELS:
        root=a.runs/name
        run=v2.read_json(root/'run.json');history=v2.read_json(root/'history.json')
        if run['status']!='completed' or run['completed_epochs']!=30 or [r['epoch'] for r in history]!=list(range(1,31)):
            raise ValueError('Both exact 30-epoch runs must finish before TEST')
        scheduled=[r for r in history if r['validation_prostate_dice'] is not None]
        if [r['epoch'] for r in scheduled]!=plan['configuration']['validation_epochs']:
            raise ValueError('Unexpected validation schedule')
        best=max(scheduled,key=lambda r:(r['validation_prostate_dice'],-r['epoch']))
        if run['best_epoch']!=best['epoch']:raise ValueError('Run did not select the predeclared earliest strict maximum')
        checkpoint=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
        if checkpoint['epoch']!=best['epoch'] or checkpoint['best_epoch']!=best['epoch']:
            raise ValueError('Actual selected checkpoint metadata differs from validation selection')
        identity=checkpoint['identity']
        if identity['model']!=name or identity['protocol_sha256']!=result['protocol_sha256'] or identity['manifest_sha256']!=result['manifest_sha256'] or identity['configuration']!=plan['configuration']:
            raise ValueError('Selected checkpoint identity differs')
        digest=v2.sha256(root/'best.pt')
        if run['best_checkpoint_sha256']!=digest:raise ValueError('Best checkpoint changed')
        result['models'][name]=dict(completed_epochs=30,checkpoint_sha256=digest,
                                    checkpoint_epoch=best['epoch'],validation_prostate_dice=best['validation_prostate_dice'],
                                    run_sha256=v2.sha256(root/'run.json'),history_sha256=v2.sha256(root/'history.json'))
    v2.write_json(a.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
