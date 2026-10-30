"""Check local preprocessing or GPU training readiness without installing packages."""
import argparse
import importlib.util
import json
import platform
import sys


def inspect(training=False):
    report = {'python': sys.version, 'executable': sys.executable,
              'platform': platform.platform(), 'packages': {}}
    for name in ['numpy', 'PIL', 'torch', 'ultralytics']:
        report['packages'][name] = bool(importlib.util.find_spec(name))
    report['data_preprocessing_ready'] = all(report['packages'][n] for n in ['numpy', 'PIL'])
    if training:
        if report['packages']['torch']:
            import torch
            report.update(torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                          cuda_available=torch.cuda.is_available())
            if report['cuda_available']:
                report['gpu'] = torch.cuda.get_device_name(0)
                report['gpu_memory_bytes'] = torch.cuda.get_device_properties(0).total_memory
                report['gpu_test'] = (torch.ones(3, device='cuda') * 2).tolist()
        if report['packages']['ultralytics']:
            import ultralytics
            report['ultralytics_version'] = ultralytics.__version__
        report['gpu_training_ready'] = bool(report.get('cuda_available') and report['packages']['ultralytics'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', action='store_true', help='Also import training libraries and test CUDA')
    args = parser.parse_args()
    result = inspect(args.training)
    print(json.dumps(result, indent=2))
    ready = result.get('gpu_training_ready') if args.training else result['data_preprocessing_ready']
    raise SystemExit(0 if ready else 1)
