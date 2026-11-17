"""Exact original head-input tap and fixed gated MIL (Ilse et al., 2018).

Method: https://arxiv.org/abs/1802.04712. Supplied deep-research-report.md
motivates the study; its broader roadmap is not implemented. Codex assisted.
"""
from __future__ import annotations
import torch
from torch import nn
import train_convnext_v2 as v2
from g012 import core

FEATURE = 'per_slice_global_spatial_mean_then_original_final_LayerNorm_768'
ATTENTION = {'input':768,'hidden':128,'V_bias':True,'U_bias':True,'w_bias':False,
             'init_seed':4242,'VU_truncated_normal_std':.02,'VU_bias':0.,'w_init':0.,
             'dropout':0.,'temperature':1.,'production_pool':'weighted_original_slice_logits'}


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # Isolate constructor draws too, including Python/NumPy/CPU/CUDA state.
        with core.isolated_rng():
            v2.seed_everything(4242)
            self.V = nn.Linear(768,128); self.U = nn.Linear(768,128)
            self.w = nn.Linear(128,1,bias=False)
            for layer in (self.V,self.U):
                nn.init.trunc_normal_(layer.weight,std=.02); nn.init.zeros_(layer.bias)
            nn.init.zeros_(self.w.weight)

    def forward(self,h):
        core.require(h.ndim == 2 and h.shape[0] > 0 and h.shape[1] == 768,'Expected positive K x 768 features.')
        return torch.softmax(self.w(torch.tanh(self.V(h))*torch.sigmoid(self.U(h))).squeeze(-1),dim=0)


class GatedScan(nn.Module):
    """Keep original parameter names and original forward, tap without a second pass.

    A hook on the original head captures its actual input/output. Consequently
    original feature execution and per-slice stochastic-depth draws are unchanged.
    The hook is temporary, per-call, and neither detaches nor copies features.
    """
    def __init__(self, backbone):
        super().__init__()
        # Register original children at their original names, not under a prefix.
        for name, child in backbone.named_children(): self.add_module(name,child)
        self.depths=backbone.depths; self.dims=backbone.dims
        self.attention=Attention()
        self._original_forward = type(backbone).forward

    def forward(self,images):
        captured=[]
        handle=self.head.register_forward_hook(lambda module,inputs,output: captured.append((inputs[0],output)))
        try: logits=self._original_forward(self,images)
        finally: handle.remove()
        core.require(len(captured)==1,'Original head must execute exactly once.')
        h, original_logits=captured[0]
        weights=self.attention(h)
        return logits, (weights[:,None]*original_logits).sum(0), weights


def make_model(device='cpu'):
    return GatedScan(v2.make_model(core.base_config(device)))


def configuration():
    c=core.config('G2')
    c.update(variant='G3',loss='hard_CE_attention_weighted_slice_logits',
             scan_aggregation='softmax_attention_weighted_slice_logits',feature=FEATURE,attention=ATTENTION)
    return c


def loss(scan_logits,label):
    core.require(scan_logits.shape==(2,) and label in (0,1),'Expected NC=0/AD=1 scan logits.')
    return -torch.log_softmax(scan_logits,0)[label]


def strict_checkpoint(model,value,identity):
    core.require(set(value)=={'format_version','identity','epoch','initial_tensor_sha256','model_state'}
                 and value['format_version']=='G3-1' and value['identity']==identity,
                 'Strict G3 checkpoint schema/configuration/source binding differs.')
    core.require(identity['configuration']==configuration(),'G3 feature/head/aggregation configuration differs.')
    core.strict_state(model,value['model_state'])
