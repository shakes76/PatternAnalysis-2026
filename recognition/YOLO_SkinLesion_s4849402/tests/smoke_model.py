"""Synthetic CPU integration check; never a lesion-performance experiment."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_pipeline import fixture
from dataset import export
from evaluate import evaluate_detector, profile_inference
from modules import CLASS_NAMES, DetectorConfig, build_detector, load_trained_detector, model_summary
from predict import predict_images
from train import TrainConfig, run_training
from utils import save_json


def main():
    import torch
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    config = DetectorConfig(weights=str(root / 'weights/yolov8n.pt'))
    model = build_detector(config)
    model.model.eval()
    with torch.inference_mode():
        result = model.model(torch.zeros(1, 3, 64, 64))
    raw = result[0] if isinstance(result, tuple) else result
    assert raw.shape[1] == 4 + len(CLASS_NAMES), raw.shape
    (root / 'runs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / 'runs') as temp:
        folder = Path(temp)
        manifest = fixture(folder)
        output = folder / 'dataset'
        export(manifest, output, list(CLASS_NAMES), 'official-isic', 'Synthetic integration test only', folder)
        best = run_training(config, TrainConfig(output / 'data.yaml', folder / 'training',
                                               image_size=64, epochs=1, batch_size=1, device='cpu', workers=0))
        loaded = load_trained_detector(best)
        rows = predict_images(best, output / 'images/val', folder / 'predictions', 64, 1.0, 'cpu')
        assert len(rows) == 1
        assert rows[0]['detections'] == [], 'Expected empty predictions at confidence=1'
        evaluate_detector(best, output / 'data.yaml', 'val', 64, folder / 'evaluation', 'cpu', True, 0.25)
        speed = profile_inference(loaded, torch.zeros(1, 3, 64, 64), 1, 2)
        save_json(root / 'reports/software_smoke.json', {
            'scope': 'software integration only; synthetic images, not model performance',
            'model': model_summary(loaded), 'forward_shape': list(raw.shape),
            'checks': ['three-class head', 'forward', 'one-epoch CPU training', 'checkpoint reload',
                       'empty prediction JSON and overlay', 'validation AP API', 'threshold sweep', 'forward profiler'],
            'profiling_exercised': bool(speed['images_per_second'] > 0),
        })
    print('PASS: synthetic CPU integration; no clinical or test-set performance claims')


if __name__ == '__main__':
    main()
