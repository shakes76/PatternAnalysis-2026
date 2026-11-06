"""
GFNet. Takes a 224x224 slice and outputs either 0 = NC or AD = 1.
"""
import torch
from torch import nn


class PatchEmbed(nn.Module):
    """
    Splits the image into 16x16 patches
    It then projects each patch onto a dim-length vector
    
    (B, 1, 224, 224) -> (B, 14, 14, dim)
    """
    def __init__(self, dim=256):
        super().__init__()
        self.proj = nn.Conv2d(1, dim, kernel_size=16, stride=16)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)

class ConvStem(nn.Module):
    """
    4 convolutions with stride of 2
    
    (B, 1, 224, 224) -> (B, 14, 14, embed_dim)
    """
    def __init__(self, in_chans=1, embed_dim=256):
        super().__init__()
        chans = [in_chans, 32, 64, 128, embed_dim]
        layers = []
        # Creates 4 convolution layers
        for i in range(4):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(chans[i + 1])]
            if i < 3:
                # No ReLu on last convolution
                layers.append(nn.ReLU(inplace=True))
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)
    
class GlobalFilter(nn.Module):
    """
    Takes 2D FFT, multiplies by a learned filter, and then takes the inverse FFT
    
    (B, h, w, dim) -> (B, h, w, dim)
    """
    def __init__(self, dim, h=14, w=14):
        super().__init__()
        # One complex weight per freq and channel
        self.complex_weight = nn.Parameter(torch.randn(h, w // 2 + 1, dim, 2) * 0.02)

    def forward(self, x):
        H, W = x.shape[1:3]
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            X = torch.fft.rfft2(x.float(), dim=(1, 2), norm="ortho") # Fast fourier transform
            X = X * torch.view_as_complex(self.complex_weight) # Multiply by complex weight
            x = torch.fft.irfft2(X, s=(H, W), dim=(1, 2), norm="ortho") # Inverse fast fourier transform
        return x.to(dtype)


class DropPath(nn.Module):
    """
    Randomly skips a blocks residual path for a sample during training
    Improves model generalisations
    """
    
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.p
        return x * keep / (1 - self.p) # Rescales


class GFBlock(nn.Module):
    """
    Structure of each GFBlock
    LayerNorm -> global filter -> LayerNorm -> MLP (wrapped in a residual connection)
    """
    def __init__(self, dim, h, w, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilter(dim, h, w)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        # A residual around the whole block
        return x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
    
    
class GFNet(nn.Module):
    """
    Creating the actual model.
    Structure follows a patch embedding and postional embedding, a stack of blocks,
    then a global average pooling and linear head
    """
    def __init__(self, img_size=224, dim=256, depth=12, drop_path_rate=0.1,conv_stem=False):
        super().__init__()
        self.stem = ConvStem(1, dim) if conv_stem else PatchEmbed(dim)
        g = img_size // 16 # Creates a 14x14 grid of patches
        self.pos_embed = nn.Parameter(torch.zeros(1, g, g, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Drop path rate increases linearly from 0 in the first block to drop_path_rate in last block
        # More dropping in later blocks
        rates = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.Sequential(*[GFBlock(dim, g, g, r) for r in rates])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)

    
    def forward(self, x):
        x = self.stem(x) + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x).mean(dim=(1, 2)) # Average across all 14x14 patches
        return self.head(x)


def load_model(path, device):
    """Loads the model"""
    checkpoint = torch.load(path, map_location=device)
    model = GFNet(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model