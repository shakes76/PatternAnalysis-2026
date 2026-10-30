""" Mamba Models but modified for our purpose

Original mamba code: 
https://github.com/csguoh/MambaIR/blob/main/basicsr/archs/mambairv2_arch.py 
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

MAMBAIR_BASE_CFG = dict(
    img_size=64,
    patch_size=1,
    in_chans=3,
    embed_dim=48,
    d_state=8,
    depths=(6, 6, 6, 6,),
    num_heads=(4, 4, 4, 4,),
    window_size=16,
    inner_rank=32,
    num_tokens=64,
    convffn_kernel_size=5,
    mlp_ratio=2.,
    qkv_bias=True,
    ape=False,
    patch_norm=True,
    upscale=1, # originally 2
    img_range=1.,
    upsampler='',
    resi_connection='1conv',
)

# Check that mambaIR is cloned
MAMBAIR_PATH = None
def clone():
    global MAMBAIR_PATH
    _here = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.join(_here, '..', '..', 'MambaIR'),   # repo root sibling
        os.path.join(_here, 'MambaIR'),                # inside project folder
        os.path.join(os.getcwd(), 'MambaIR'),          # current working dir
    ]
    for _p in _candidates:
        if os.path.isdir(_p) and os.path.isfile(os.path.join(_p, 'basicsr', 'archs', 'mambairv2light_arch.py')):
            MAMBAIR_PATH = os.path.abspath(_p)
            break

    if MAMBAIR_PATH is None:
        raise ImportError(
            "Cannot find MambaIR repo. Clone it with:\n"
            "  git clone https://github.com/csguoh/MambaIR\n"
            "and place it next to this project, or set MAMBAIR_PATH manually."
        )

    if MAMBAIR_PATH not in sys.path:
        sys.path.insert(0, MAMBAIR_PATH)

clone()
from basicsr.archs.mambairv2light_arch import MambaIRv2Light

# using pretrained model for this.
CHECKPOINT_URL = (
    "https://github.com/csguoh/MambaIR/releases/download/pretrained_weight/"
    "mambairv2_lightSR_x2.pth"
)

def download_checkpoint(save_path):
    """Download the pretrained Gaussian colour denoising checkpoint."""
    import urllib.request

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    urllib.request.urlretrieve(CHECKPOINT_URL, save_path)
    print("Downloaded pretrained.")

def load_pretrained(model: MambaIRv2Light, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('params_ema', ckpt.get('params', ckpt))
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    print(f"  Loaded {len(filtered)}/{len(model_state)} layers from checkpoint (shape-matched).")
    if len(filtered) == 0:
        print("  WARNING: No weights matched — training from scratch.")
    model.load_state_dict(filtered, strict=False)



# A pretrained model with different ins and outs
# Frozen inbetweens

class MambaIRv2Frozen(nn.Module):
    """ 
    - get a pretrained mamba model
    - freeze the backbone
    - replace the first and last because our inputs are different
    - train only new layers
    """

    def __init__(self, checkpoint_path, device):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.backbone = MambaIRv2Light(**MAMBAIR_BASE_CFG)

        if checkpoint_path is not None:
            if not os.path.isfile(checkpoint_path):
                download_checkpoint(checkpoint_path)
            load_pretrained(self.backbone, checkpoint_path, device)

        # Freeze insides
        for param in self.backbone.parameters():
            param.requires_grad = False

        embed_dim = MAMBAIR_BASE_CFG['embed_dim']

        # Replace in & out
        self.backbone.conv_first = nn.Conv2d(1, embed_dim, 3, 1, 1)
        self.backbone.conv_last = nn.Conv2d(embed_dim, 2, 3, 1, 1)

        self.tanh = nn.Tanh()
        self.backbone.mean = torch.zeros(1, 1, 1, 1)

        self.to(device)

    def forward(self, L):
        return self.tanh(self._forward_denoising(L))

    def _forward_denoising(self, x):
        "copy mambaIr forward, except no mean normalisation"
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.backbone.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]


        attn_mask = self.backbone.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.backbone.relative_position_index_SA}

        x_first = self.backbone.conv_first(x)

        with torch.no_grad():
            features = self.backbone.forward_features(x_first, params)

        res = self.backbone.conv_after_body(features) + x_first
        
        out = self.backbone.conv_last(res)

        out = out[..., :h_ori, :w_ori]
        return out

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


# Another one that trains all layers

class MambaIRv2FineTuned(nn.Module):

    def __init__(self, checkpoint_path, device):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.backbone = MambaIRv2Light(**MAMBAIR_BASE_CFG)

        #load pretrained if provided
        if checkpoint_path is not None:
            if not os.path.isfile(checkpoint_path):
                download_checkpoint(checkpoint_path)
            print(f"[MambaIRv2FineTuned] Loading checkpoint from {checkpoint_path}")
            load_pretrained(self.backbone, checkpoint_path, device)

        embed_dim = MAMBAIR_BASE_CFG['embed_dim']

        #replace IO
        self.backbone.conv_first = nn.Conv2d(1, embed_dim, 3, 1, 1)
        self.backbone.conv_last = nn.Conv2d(embed_dim, 2, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.backbone.mean = torch.zeros(1, 1, 1, 1)
        
        self.to(device)

    def forward(self, L):
        return self.tanh(self._forward_denoising(L))

    def _forward_denoising(self, x):
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.backbone.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]
        
        attn_mask = self.backbone.calculate_mask([h, w]).to(x.device)
        params = {
            'attn_mask': attn_mask,
            'rpi_sa': self.backbone.relative_position_index_SA,
        }

        x_first = self.backbone.conv_first(x)
        # checkpoint is slower but wont OOM ;)
        x_size = (h, w)
        feat = self.backbone.patch_embed(x_first)
        if self.backbone.ape:
            feat = feat + self.backbone.absolute_pos_embed
        
        for layer in self.backbone.layers:
            feat = checkpoint(layer, feat, x_size, params, use_reentrant=False)
        
        feat = self.backbone.norm(feat)
        feat = self.backbone.patch_unembed(feat, x_size)
        
        res = self.backbone.conv_after_body(feat) + x_first
        out = self.backbone.conv_last(res)
        out = out[..., :h_ori, :w_ori]
        return out

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}
        

# loss function

class L1_Saturation_Loss(nn.Module):
    """
    L1 + saturation penalty loss function
    """

    def __init__(self, l1_weight=1.0, saturation_weight=0.1):
        super().__init__()

        self.l1_weight = l1_weight
        self.saturation_weight = saturation_weight
        self.l1 = nn.L1Loss()

    def forward(self, pred_ab, target_ab):
        l1_loss = self.l1(pred_ab, target_ab)

        # saturation 
        pred_chroma = pred_ab.norm(dim=1, keepdim=True)
        true_chroma = target_ab.norm(dim=1, keepdim=True)

        sat_loss = torch.clamp(true_chroma - pred_chroma, min=0).mean()

        return self.l1_weight*l1_loss + self.saturation_weight*sat_loss


class LPIPS_Loss(nn.Module):
    """
    LPIPS loss function +l1 for stabilising
    """

    def __init__(self, lpips_weight=1.0, l1_weight=0.1, device=None):
        super().__init__()

        import lpips

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Alex is cheaper to run
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_weight = lpips_weight
        self.l1_weight = l1_weight
        self.l1 = nn.L1Loss()

        # freeze LPIPS weights
        for param in self.lpips_fn.parameters():
            param.requires_grad = False


    def forward(self, pred_ab, target_ab):
        #Need RGB not LAB
        B, _, H, W = pred_ab.shape
        zeros = torch.zeros(B, 1, H, W, device=pred_ab.device, dtype=pred_ab.dtype)
 
        pred_3ch   = torch.cat([zeros, pred_ab],   dim=1)  # (B, 3, H, W)
        target_3ch = torch.cat([zeros, target_ab], dim=1)
 
        lpips_loss = self.lpips_fn(pred_3ch, target_3ch).mean()
        l1_loss    = self.l1(pred_ab, target_ab)
 
        return self.lpips_weight * lpips_loss + self.l1_weight * l1_loss




# Quick check

if __name__ == "__main__":
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    for ModelClass in [MambaIRv2Frozen, MambaIRv2FineTuned]:
            print(f"\n--- {ModelClass.__name__} ---")
            model = ModelClass(checkpoint_path=None, device=device)
            counts = model.parameter_counts()
            print(f"  Total params:     {counts['total'] / 1e6:.2f}M")
            print(f"  Trainable params: {counts['trainable'] / 1e6:.4f}M  ({counts['trainable']:,})")
            print(f"  Frozen params:    {counts['frozen'] / 1e6:.2f}M")
    
            # Shape test
            dummy = torch.randn(1, 1, 256, 256).to(device)
            t0 = time.time()
            with torch.no_grad():
                out = model(dummy)
            elapsed = time.time() - t0
            print(f"  Input:  {tuple(dummy.shape)}")
            print(f"  Output: {tuple(out.shape)}")
            print(f"  Forward pass: {elapsed:.3f}s")
            assert out.shape == (1, 2, 256, 256), f"Unexpected output shape: {out.shape}"
            assert out.min() >= -1.0 - 1e-4 and out.max() <= 1.0 + 1e-4, \
                f"Output out of [-1,1]: min={out.min():.4f} max={out.max():.4f}"
            print("   Shape and range checks passed")
    
    print("\n modules.py sanity check complete")
        