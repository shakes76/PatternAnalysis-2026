"""Synthetic CPU parity check only; torchvision is an oracle, not a training model.

Reference: installed torchvision/models/convnext.py (version recorded in output).
https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
No pretrained weights, dataset, optimizer step or checkpoint is used.
"""
import hashlib
import inspect
import json
from datetime import datetime
from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision.models import convnext_tiny
from modules import ConvNeXtTiny


def parameter_mapping(model):
    mapping = {}
    def pair(a, b):
        for suffix in ('weight', 'bias'):
            mapping[a + '.' + suffix] = b + '.' + suffix
    pair('stem.0', 'features.0.0')
    pair('stem.1.norm', 'features.0.1')
    for stage, blocks in enumerate(model.stages):
        for block in range(len(blocks)):
            a, b = f'stages.{stage}.{block}', f'features.{2*stage+1}.{block}'
            for ours, theirs in [('depthwise','0'),('norm','2'),('expand','3'),('project','5')]:
                pair(f'{a}.{ours}',f'{b}.block.{theirs}')
            mapping[a+'.layer_scale'] = b+'.layer_scale'
        if stage < 3:
            pair(f'downsamples.{stage}.0.norm',f'features.{2*stage+2}.0')
            pair(f'downsamples.{stage}.1',f'features.{2*stage+2}.1')
    pair('final_norm','classifier.0')
    pair('classifier','classifier.2')
    return mapping


def check_close(a, b):
    torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-6)
    return float((a-b).abs().max())


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    ours = ConvNeXtTiny().cpu().eval()
    reference = convnext_tiny(weights=None, num_classes=2, stochastic_depth_prob=0.0).cpu().eval()
    reference.features[0][0] = nn.Conv2d(1,96,4,stride=4)
    mapping = parameter_mapping(ours)
    a, b = dict(ours.named_parameters()), dict(reference.named_parameters())
    if set(mapping) != set(a) or set(mapping.values()) != set(b):
        raise AssertionError('Parameter mapping must cover both models completely')
    if sum(p.numel() for p in a.values()) != sum(p.numel() for p in b.values()):
        raise AssertionError('Parameter count mismatch')
    with torch.no_grad():
        for name, other in mapping.items():
            b[other].copy_(a[name].reshape_as(b[other]))
    report = {'torch':str(torch.__version__), 'torchvision':str(torchvision.__version__),
              'device':'cpu', 'mapped_parameter_tensors':len(mapping),
              'parameters':sum(p.numel() for p in a.values()),
              'tolerance':{'rtol':1e-4,'atol':2e-6},
              'reference_adaptations':['single-channel stem','two outputs','stochastic depth disabled'],
              'checks':[]}
    # Full image size, intermediate activations and final output.
    with torch.no_grad():
        x = torch.rand(1,1,240,256)
        u, v = ours.stem(x), reference.features[0](x)
        report['checks'].append({'name':'stem','max_abs_error':check_close(u,v)})
        for i in range(4):
            u, v = ours.stages[i](u), reference.features[2*i+1](v)
            report['checks'].append({'name':f'stage_{i+1}','shape':list(u.shape),'max_abs_error':check_close(u,v)})
            if i < 3:
                u, v = ours.downsamples[i](u), reference.features[2*i+2](v)
                report['checks'].append({'name':f'downsample_{i+1}','max_abs_error':check_close(u,v)})
        report['checks'].append({'name':'full_size_logits','max_abs_error':check_close(ours(x),reference(x))})
    # Larger residual scale exercises branches that are tiny at initialization.
    with torch.no_grad():
        for name, other in mapping.items():
            if name.endswith('layer_scale'):
                a[name].fill_(0.1)
                b[other].fill_(0.1)
    ours.train()
    reference.train()
    x = torch.rand(2,1,64,64,requires_grad=True)
    z = x.detach().clone().requires_grad_(True)
    logits_a, logits_b = ours(x), reference(z)
    loss_a = nn.functional.cross_entropy(logits_a,torch.tensor([0,1]))
    loss_b = nn.functional.cross_entropy(logits_b,torch.tensor([0,1]))
    loss_a.backward()
    loss_b.backward()
    grad_errors = {}
    for name, other in mapping.items():
        if a[name].grad is None or b[other].grad is None:
            raise AssertionError('Missing gradient: '+name)
        grad_errors[name] = check_close(a[name].grad,b[other].grad.reshape_as(a[name]))
    report['checks'].append({'name':'train_mode_residual_scale_0.1',
        'logits_max_abs_error':check_close(logits_a.detach(),logits_b.detach()),
        'loss_abs_error':check_close(loss_a.detach(),loss_b.detach()),
        'input_gradient_max_abs_error':check_close(x.grad,z.grad),
        'parameter_gradient_max_abs_error':max(grad_errors.values()),
        'worst_parameter':max(grad_errors,key=grad_errors.get)})
    base = Path(__file__).resolve().parent
    source = Path(inspect.getfile(type(reference)))
    report['source_sha256'] = {name:hashlib.sha256(path.read_bytes()).hexdigest()
                              for name,path in [('custom_modules',base/'modules.py'),('installed_torchvision_convnext',source)]}
    report['result'] = 'PASS'
    report['limits'] = 'Synthetic matched-weight parity only; not a dataset, training-recipe, stochastic-depth or generalisation guarantee. Residual-scale stress change is in memory only.'
    output = base/'runs'/('architecture_parity_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2,allow_nan=False))
    print('Saved:',output)


if __name__ == '__main__':
    main()
