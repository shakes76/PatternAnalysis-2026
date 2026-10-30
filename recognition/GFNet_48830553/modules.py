import torch
import torch.nn as nn
import os

# ==========================================
# Global Loss & Dataset Configuration
# ==========================================
IMG_SIZE = 256
IN_CHANS = 1          # Greyscale images
NUM_CLASSES = 2       # NC (0) vs AD (1)
WEIGHT = None         # Uniform class weights or balanced tensor
LABEL_SMOOTHING = 0.10 # Increased label smoothing (0.10) to strictly bound maximum cross-entropy penalties on ambiguous MRI slices and prevent validation loss spikes

# ==========================================
# ResNet-18 Hyperparameters (Default Configuration)
# ==========================================
RESNET18_DROPOUT_RATE = 0.2     # Regularisation before classification head (tuned for 2D MRI)
RESNET18_DROP_PATH_RATE = 0.05  # Stochastic depth across residual blocks
RESNET18_SE_REDUCTION = 8       # SE attention reduction ratio (8 preserves more fine-grained channel variance)
RESNET18_BN_MOMENTUM = 0.05     # Smoother running statistics tracking across heterogeneous MRI slice batches

# ==========================================
# GFNet Hyperparameters (Optimized Configuration)
# ==========================================
PATCH_SIZE = 8        # Spatial patch size: 256 / 8 = 32 patches per axis (1024 tokens for finer spatial resolution)
EMBED_DIM = 256       # Hidden embedding dimension
DEPTH = 6             # Number of GFNet Fourier Transformer blocks
MLP_RATIO = 4.0       # MLP expansion ratio
DROPOUT_RATE = 0.1    # Dropout probability across MLP and positional embeddings
DROP_PATH_RATE = 0.1  # Stochastic depth rate across GFNet blocks to prevent overfitting
HEAD_DROPOUT_RATE = 0.2 # Dropout probability before the classification head
INIT_VALUES = 1e-4    # LayerScale initial value calibrated to 1e-4 to start near identity and prevent variance explosion across layers



def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation channel attention block.
    """
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_path_rate=0.0, reduction=RESNET18_SE_REDUCTION, bn_momentum=RESNET18_BN_MOMENTUM):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.se = SqueezeExcitation(planes, reduction=reduction)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # ResNet-D downsampling: AvgPool2d(2, 2) + Conv2d(1x1) avoids throwing away 75% of spatial data
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if stride != 1:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes, momentum=bn_momentum)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes, momentum=bn_momentum)
                )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = residual + self.drop_path(out)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """
    Optimized ResNet-18 Model tailored for ADNI dataset (256x256 images).
    
    Key Optimizations:
    - Standard ResNet early downsampling stem: Conv2d(7x7, stride=2) + MaxPool2d(3x3, stride=2),
      downsampling 256x256 -> 64x64 to dramatically save memory and computation.
    - ResNet-D style downsampling in residual connections to retain spatial information.
    - Squeeze-and-Excitation (SE) channel attention module in BasicBlock with reduction=8.
    - Stochastic Depth (DropPath) regularization across residual stages.
    - Zero-initialized final BatchNorm in each residual block for identity-preserving initial state.
    - Parameter grouping method (get_parameter_groups) for weight decay decoupling.
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=RESNET18_DROPOUT_RATE,
                 drop_path_rate=RESNET18_DROP_PATH_RATE, se_reduction=RESNET18_SE_REDUCTION,
                 bn_momentum=RESNET18_BN_MOMENTUM):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        # Downsample 256x256 -> 128x128 -> 64x64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 8 residual blocks total across 4 stages
        total_blocks = 8
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1, dpr=dpr[0:2], reduction=se_reduction, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, dpr=dpr[2:4], reduction=se_reduction, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, dpr=dpr[4:6], reduction=se_reduction, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, dpr=dpr[6:8], reduction=se_reduction, bn_momentum=bn_momentum)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual block (accelerates convergence)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride, dpr, reduction=RESNET18_SE_REDUCTION, bn_momentum=RESNET18_BN_MOMENTUM):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, s in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride=s, drop_path_rate=dpr[i], reduction=reduction, bn_momentum=bn_momentum))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_parameter_groups(self, weight_decay=1e-2):
        """
        Separates parameters into decay and no-decay groups.
        Excludes 1D tensors (biases, BatchNorm weights/biases) from weight decay.
        """
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class PatchEmbed(nn.Module):
    """
    Splits 2D image into patches and linearly projects them into embedding dimension.
    """
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS, embed_dim=EMBED_DIM):
        super(PatchEmbed, self).__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, D, H_grid, W_grid) -> (B, H_grid, W_grid, D)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class GlobalFilter(nn.Module):
    """
    Global Filter layer performing 2D FFT, element-wise multiplication with learnable
    complex frequency weights, and 2D inverse FFT.
    """
    def __init__(self, dim=EMBED_DIM, h=IMG_SIZE // PATCH_SIZE, w=IMG_SIZE // PATCH_SIZE):
        super(GlobalFilter, self).__init__()
        self.h = h
        self.w = w
        # Real FFT along width dimension produces w // 2 + 1 non-redundant frequencies
        # Initialized with zero-mean Gaussian (std=0.02) following original GFNet specification.
        # Avoids biasing every frequency to unity gain at start, which prevents passing
        # unconstrained high-frequency noise that causes validation loss spikes.
        weight_init = torch.randn(h, w // 2 + 1, dim, 2, dtype=torch.float32) * 0.02
        self.complex_weight = nn.Parameter(weight_init)

    def forward(self, x):
        B, H, W, C = x.shape
        # Convert to float32 for FFT calculation
        x = x.to(torch.float32)

        # 2D Real FFT along spatial axes (H, W) -> (B, H, W // 2 + 1, C)
        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # Form complex weights and explicitly add batch dimension for robust broadcasting
        weight = torch.view_as_complex(self.complex_weight).unsqueeze(0)
        x_fft = x_fft * weight

        # Inverse 2D Real FFT back to spatial domain
        x = torch.fft.irfft2(x_fft, s=(H, W), dim=(1, 2), norm='ortho')
        return x


class Mlp(nn.Module):
    """
    Standard Feed-Forward Network (MLP) used in Transformer blocks.
    """
    def __init__(self, in_features=EMBED_DIM, hidden_features=None, out_features=None, drop=DROPOUT_RATE):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * MLP_RATIO)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GFNetBlock(nn.Module):
    """
    GFNet Transformer Block: LayerNorm -> GlobalFilter -> LayerNorm -> Mlp with residual connections, LayerScale, and DropPath.
    """
    def __init__(self, dim=EMBED_DIM, h=IMG_SIZE // PATCH_SIZE, w=IMG_SIZE // PATCH_SIZE,
                 mlp_ratio=MLP_RATIO, drop=DROPOUT_RATE, drop_path=0.0, init_values=INIT_VALUES):
        super(GFNetBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilter(dim=dim, h=h, w=w)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        # LayerScale parameters to stabilize deep residual learning
        self.gamma1 = nn.Parameter(init_values * torch.ones(dim)) if init_values > 0 else 1.0
        self.gamma2 = nn.Parameter(init_values * torch.ones(dim)) if init_values > 0 else 1.0
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # Global filter sub-layer with LayerScale, DropPath, and residual connection
        x = x + self.drop_path1(self.gamma1 * self.filter(self.norm1(x)))
        # MLP sub-layer with LayerScale, DropPath, and residual connection
        x = x + self.drop_path2(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class GFNet(nn.Module):
    """
    Global Filter Network (GFNet) tailored for 256x256 greyscale ADNI images.
    """
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS,
                 num_classes=NUM_CLASSES, embed_dim=EMBED_DIM, depth=DEPTH,
                 mlp_ratio=MLP_RATIO, drop_rate=DROPOUT_RATE, drop_path_rate=DROP_PATH_RATE,
                 head_drop_rate=HEAD_DROPOUT_RATE, init_values=INIT_VALUES):
        super(GFNet, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        h, w = self.patch_embed.grid_size

        # Learnable 2D positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, h, w, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule across transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks with global Fourier filters, LayerScale, and DropPath
        self.blocks = nn.ModuleList([
            GFNetBlock(
                dim=embed_dim, h=h, w=w, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], init_values=init_values
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(p=head_drop_rate) if head_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_parameter_groups(self, weight_decay=1e-2):
        """
        Separates parameters into decay and no-decay groups.
        Excludes 1D tensors (biases, LayerNorm, LayerScale) and positional embeddings.
        Applies weight decay to complex frequency filter weights to prevent runaway
        high-frequency amplification that leads to severe validation loss spikes.
        """
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name or "gamma" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def forward(self, x):
        # Patch embedding: (B, C, 256, 256) -> (B, H_grid, W_grid, D)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Global filter blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # Global average pooling over spatial grid dimensions (H_grid, W_grid)
        x = x.mean(dim=(1, 2))
        x = self.head_drop(x)
        logits = self.head(x)
        return logits


def get_criterion():
    """
    Returns the standard CrossEntropyLoss for ADNI classification.
    """
    return nn.CrossEntropyLoss(weight=WEIGHT, label_smoothing=LABEL_SMOOTHING)

def load_checkpoint(path, device):
    """
    Loads checkpoint if file exists in path or in script directory.
    """
    target_path = path
    if not os.path.exists(target_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, path)
        if os.path.exists(alt_path):
            target_path = alt_path
        else:
            return None
    try:
        return torch.load(target_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Warning: Could not load checkpoint from '{target_path}': {e}")
        return None

def get_device():
    """
    Selects the optimal hardware device available.
    Attempts to use GPU (CUDA / DirectML / MPS), falling back to CPU if unavailable.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
