"""Export nine fixed test-case slice montages for private visual review."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import tempfile
from PIL import Image,ImageOps,ImageDraw


def select_cases(rows):
    rows=list(rows)
    groups=[('confident_AD_miss',[r for r in rows if r['label']==1 and r['prediction']==0],lambda r:(-r['confidence'],r['patient_id'])),
            ('confident_NC_false_alarm',[r for r in rows if r['label']==0 and r['prediction']==1],lambda r:(-r['confidence'],r['patient_id'])),
            ('uncertain_correct',[r for r in rows if r['label']==r['prediction']],lambda r:(r['confidence'],r['patient_id']))]
    result=[]
    for name,candidates,key in groups:
        if len(candidates)<3:raise ValueError('Fewer than three cases in '+name)
        for i,row in enumerate(sorted(candidates,key=key)[:3],1):result.append(dict(row,alias=f'{name}_{i:02d}',group=name))
    if len({r['patient_id'] for r in result})!=9:raise ValueError('Case overlap')
    return result


def export_cases(chosen,manifest,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    chosen=list(chosen);manifest=list(manifest);sources=[]
    for case in chosen:
        matched=sorted((r for r in manifest if r['patient_id']==case['patient_id']),key=lambda r:r['file_path'])
        if len(matched)!=20 or len({r['file_path'] for r in matched})!=20:raise ValueError('Wrong slice count')
        if any(r['split']!='test' or r['image_id']!=case['image_id'] or (1 if r['label']=='AD' else 0)!=case['label'] for r in matched):raise ValueError('Metadata mismatch')
        if any(not Path(r['file_path']).is_file() for r in matched):raise FileNotFoundError('Missing selected slice')
        sources.append(matched)
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='.r6case-',dir=out.parent))
    try:
        for case,items in zip(chosen,sources):
            canvas=Image.new('RGB',(5*160,4*162),'white');draw=ImageDraw.Draw(canvas)
            for i,r in enumerate(items):
                with Image.open(r['file_path']) as img:
                    img.load();tile=ImageOps.contain(img.convert('L'),(156,140))
                x=(i%5)*160;y=(i//5)*162
                canvas.paste(tile.convert('RGB'),(x+(160-tile.width)//2,y))
                draw.text((x+4,y+142),str(i+1),fill='black')
            canvas.save(tmp/(case['alias']+'.png'))
        fields=['alias','group','patient_id','image_id','label','p_ad_raw','p_ad','confidence','prediction','prediction_ad80','refer_10_percent']
        with (tmp/'case_index.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in chosen)
        (tmp/'REVIEW_TEMPLATE.md').write_text('# R6b visual review\n\nFor each alias, inspect all 20 slices and record only observed features.\n\n|Alias|Empty/edge slices|Crop or artifact|Visual observation|Possible effect on prediction|Reviewed by/date|\n|---|---|---|---|---|---|\n'+''.join(f"|{r['alias']}||||||\n" for r in chosen))
        tmp.rename(out)
    finally:
        if tmp.exists():shutil.rmtree(tmp)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    evaluation=json.loads(a.evaluation.read_text())
    with a.manifest.open() as f:manifest=list(csv.DictReader(f))
    chosen=select_cases(evaluation['patient_tables']['ConvNeXt-F/ensemble'])
    export_cases(chosen,manifest,a.output)
    print('Exported nine private case montages:',a.output)
if __name__=='__main__':main()
