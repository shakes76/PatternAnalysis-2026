"""Download three training examples and official masks for pipeline inspection.

Archive images are fetched by ISIC ID; alignment is checked against 2017 masks.
This is a development audit, not a patient-safe training split or performance test.
"""
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageDraw

from dataset import mask_to_box

ROOT = Path(__file__).resolve().parent
MASK_URL = ('https://isic-archive.s3.amazonaws.com/challenges/2017/'
            'ISIC-2017_Training_Part1_GroundTruth.zip')
NAMES = ['melanoma', 'seborrheic_keratosis', 'nevus']


def cached_download(url, path, limit):
    if path.exists():
        payload = path.read_bytes()
    else:
        with urlopen(url, timeout=90) as response:
            payload = response.read(limit + 1)
        if len(payload) > limit:
            raise ValueError(f'Download exceeds size limit: {url}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    if len(payload) > limit:
        raise ValueError('Cached file exceeds size limit')
    return payload


def main():
    data = ROOT / 'data'
    labels = data / 'metadata' / 'ISIC-2017_Training_Part3_GroundTruth.csv'
    if not labels.exists():
        raise ValueError('Run audit_labels.py first')
    chosen = {}
    with labels.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            name = 'melanoma' if float(row['melanoma']) else 'seborrheic_keratosis' if float(row['seborrheic_keratosis']) else 'nevus'
            chosen.setdefault(name, row['image_id'])
    if set(chosen) != set(NAMES):
        raise ValueError('Expected all three classes')
    archive_bytes = cached_download(MASK_URL, data / 'archives' / 'training_masks.zip', 20*1024*1024)
    archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    panels, findings = [], []
    for name in NAMES:
        image_id = chosen[name]
        api_url = f'https://api.isic-archive.com/api/v2/images/{image_id}/'
        meta_bytes = cached_download(api_url, data / 'metadata' / f'{image_id}_api.json', 1024*1024)
        meta = json.loads(meta_bytes)
        if meta['isic_id'] != image_id:
            raise ValueError('API image ID mismatch')
        source = meta['files']['full']['url']
        raw = cached_download(source, data / 'samples' / f'{image_id}.jpg', 25*1024*1024)
        matches = [p for p in archive.namelist() if Path(p).name == image_id + '_segmentation.png']
        if len(matches) != 1:
            raise ValueError(f'Mask missing/ambiguous for {image_id}')
        info = archive.getinfo(matches[0])
        if info.file_size > 20*1024*1024:
            raise ValueError('Mask unexpectedly large')
        mask_bytes = archive.read(matches[0])
        image = Image.open(io.BytesIO(raw)).convert('RGB')
        mask = Image.open(io.BytesIO(mask_bytes))
        if image.size != mask.size:
            raise ValueError(f'Archive image / 2017 mask dimensions disagree: {image_id}')
        box = mask_to_box(mask)
        (data / 'samples' / f'{image_id}_segmentation.png').write_bytes(mask_bytes)
        a = np.asarray(mask)
        ys, xs = np.nonzero(a > 0)
        pixel_box = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
        overlay = image.copy()
        tint = Image.new('RGB', image.size, (255, 70, 70))
        alpha = Image.fromarray(((a > 0)*85).astype('uint8'))
        overlay.paste(tint, mask=alpha)
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((pixel_box[0],pixel_box[1],pixel_box[2]-1,pixel_box[3]-1), outline=(0,255,70), width=max(2,image.width//250))
        panel = Image.new('RGB',(540,450),'white')
        overlay.thumbnail((520,390))
        panel.paste(overlay,((540-overlay.width)//2,50))
        draw = ImageDraw.Draw(panel)
        draw.text((12,10),f'{image_id} | {name}',fill='black')
        draw.text((12,28),'Red: foreground mask; Green: bounding box',fill='black')
        panels.append(panel)
        clinical = meta.get('metadata',{}).get('clinical',{})
        findings.append({'image_id':image_id,'class_name':name,'class_id':NAMES.index(name),
                         'image_size':list(image.size),'yolo_box':list(box),'pixel_box_xyxy_exclusive':pixel_box,
                         'api_url':api_url,'image_url':source,'license':meta.get('copyright_license'),
                         'attribution':meta.get('attribution'), 'api_clinical_fields':sorted(clinical),
                         'image_sha256':hashlib.sha256(raw).hexdigest(),
                         'mask_sha256':hashlib.sha256(mask_bytes).hexdigest()})
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    sheet = Image.new('RGB',(540*len(panels),450),'white')
    for i,panel in enumerate(panels):
        sheet.paste(panel,(540*i,0))
    sheet.save(reports / 'sample_boxes.jpg',quality=90)
    report = {'selection':'first training CSV entry for each class, not performance sampling',
              'mask_archive_url':MASK_URL,'mask_archive_sha256':hashlib.sha256(archive_bytes).hexdigest(),
              'patient_independence_verified':False,
              'notes':['Current archive JPEGs, matched to 2017 masks by image ID and dimensions.',
                       'Matching dimensions does not prove pixel identity; inspect overlays.',
                       'No patient mapping established; samples are for development inspection only.'],
              'examples':findings}
    (reports / 'sample_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
