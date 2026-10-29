"""Example inference from a saved two-class checkpoint and validation-selected lock."""
import argparse
from pathlib import Path
from evaluation import inference_args
from modules import load_trained
from runtime import digest, load_config, read_json, write_json
from classification_metrics import primary_prediction


def predict(config, checkpoint, lock, image, output, device='cpu'):
    cfg = load_config(config); selected = read_json(lock)
    if selected['selection_split'] != 'val' or selected['checkpoint_sha256'] != digest(checkpoint) or selected['config_sha256'] != digest(config):
        raise ValueError('Use the checkpoint/config that produced the validation selection')
    if selected['threshold'] not in cfg['evaluation']['threshold_grid']:
        raise ValueError('Unrecognized selected threshold')
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new prediction output directory')
    output.mkdir(parents=True)
    model = load_trained(checkpoint)
    # Keep the same low-score NMS pass, then apply the selected operating threshold.
    result = model.predict(str(image), **inference_args(cfg, device))[0]
    result = result[result.boxes.conf > selected['threshold']]
    result.save(filename=str(output / 'prediction.jpg'))
    primary = primary_prediction([
        {'class_id': int(c), 'confidence': float(s), 'box': b}
        for c, s, b in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxyn.tolist())
    ], selected['threshold'])
    data = {'image': str(image), 'threshold': selected['threshold'], 'confidence_is_cancer_probability': False,
            'image_classification': {'class': model.names[primary['class_id']] if primary else None,
                'status': 'classified' if primary else 'no_prediction',
                'rule': 'Highest-confidence retained box; no GT or IoU used for this image decision'},
            'checkpoint_sha256': digest(checkpoint), 'predictions': [
                {'class': model.names[int(c)], 'confidence': float(s), 'box_xyxy_pixels': b}
                for c, s, b in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())]}
    write_json(output / 'prediction.json', data)
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'lock', 'image', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    print(predict(a.config, a.checkpoint, a.lock, a.image, a.output, a.device))
