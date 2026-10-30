import torch
import torch.nn as nn
import torch.nn.functional as F



def get_layers(dims):
    if dims == 2 :
        return  nn.Conv2d,nn.InstanceNorm2d,nn.Dropout2d
    if dims == 3:
        return nn.Conv3d, nn.InstanceNorm3d, nn.Dropout3d
    raise ValueError(f"dims must be 2 or 3, got {dims}")
        

def norm_block(channels,dims =2 ):
    _, Norm, _ = get_layers(dims)
    return nn.Sequential(Norm(channels,affine=True),
                         nn.LeakyReLU(0.01,inplace=True)
                         )

# Preactivates residual block
class ContextModule(nn.Module):
    def __init__(self,channels,dropout_prob=0.3,dims = 2):
        super().__init__()
        Conv,_,Dropout = get_layers(dims)
        self.block = nn.Sequential(norm_block(channels,dims),Conv(channels,channels,kernel_size=3,padding=1, bias=False),
                                   Dropout(dropout_prob),norm_block(channels,dims),
                                   Conv(channels,channels,kernel_size=3,padding=1, bias=False))


    def forward(self,x):
        return self.block(x) + x

# Upsample to the nearest ,then a 3x3 conv that halves the channels
class UpsampleModule(nn.Module):
    def __init__(self,in_channels,out_channels,dims=2):
        super().__init__()
        Conv,_,_ = get_layers(dims)
        self.conv = nn.Sequential(norm_block(in_channels,dims),Conv(in_channels,out_channels,kernel_size=3,padding=1, bias=False))


    def forward(self,x):
        x = F.interpolate(x,scale_factor=2,mode='nearest')
        return self.conv(x)

#3x3 conv keeps channels then 1x1 conv
class LocalizationModule(nn.Module):
    def __init__(self,in_channels,out_channels,dims):
        super().__init__()
        Conv,_,_ = get_layers(dims)
        self.block = nn.Sequential(norm_block(in_channels,dims),Conv(in_channels,in_channels,kernel_size=3,padding=1,bias=False),
                                   norm_block(in_channels,dims),Conv(in_channels,out_channels,kernel_size=1, bias=False))

    def forward(self,x):
        return self.block(x)

# Improved U-Net 2D or 3D

class ImprovedUNet(nn.Module):
    def __init__(self,in_channels=1,num_classes = 6,base_channels=16,dropout_prob=0.3,dims=2):
        super().__init__()
        ch = [base_channels * (2 ** i) for i in range(5)]
        self.dims = dims
        Conv,_,_ = get_layers(dims)
        # encoder pathway
        # building 4 downsampling blocks
        self.stem = Conv(in_channels,ch[0],kernel_size=3,padding=1,bias=False)
        self.down = nn.ModuleList([nn.Sequential(norm_block(ch[i - 1],dims),
                          Conv(ch[i - 1], ch[i], 3, stride=2, padding=1, bias=False)) for i in range(1, 5)])

        self.context = nn.ModuleList([ContextModule(c,dropout_prob=dropout_prob,dims=dims) for c in ch])

        # decoder pathway
        # building 4 upsampling blocks
        self.up = nn.ModuleList([UpsampleModule(ch[i],ch[i-1],dims) for i in range(4,0,-1)])
        self.local = nn.ModuleList([LocalizationModule(2*ch[i],ch[i],dims) for i in (3,2,1)])
        self.final_conv = nn.Sequential(norm_block(2 * ch[0],dims), Conv(2 * ch[0], 2*ch[0], kernel_size=3, padding=1, bias=False),
                                        norm_block(2 * ch[0],dims))


        self.segmentation_2 = Conv(ch[2], num_classes,kernel_size=1)
        self.segmentation_1 = Conv(ch[1], num_classes,kernel_size=1)
        self.segmentation_0 = Conv(2*ch[0], num_classes,kernel_size=1)

    
    def upsample(self,x):
        mode = 'bilinear' if self.dims == 2 else 'trilinear'
        return F.interpolate(x, scale_factor=2, mode=mode, align_corners=False)
    def forward(self,x):
        #keep output of each level as a skip connection
        skips = []
        x = self.context[0](self.stem(x))
        skips.append(x)
        for i in range(1,5):
            x = self.context[i](self.down[i-1](x))
            skips.append(x)

        # decoder
        # level 3
        x = self.up[0](skips[4])
        x = self.local[0](torch.cat([skips[3],x], dim=1))

        # level 2
        x = self.up[1](x)
        x = self.local[1](torch.cat([skips[2],x],dim=1))
        s2 = self.segmentation_2(x)

        # level 1
        x = self.up[2](x)
        x = self.local[2](torch.cat([skips[1],x],dim=1))
        s1 = self.segmentation_1(x)

        # level 0
        x = self.up[3](x)
        x = self.final_conv(torch.cat([skips[0],x],dim=1))
        s0 = self.segmentation_0(x)

        s1 = s1 +self.upsample(s2)
        s0 = s0 + self.upsample(s1)

        return s0 

class DiceCeLoss(nn.Module):
    def __init__(self,num_classes = 6 , dice_weight = 1.0,ce_weight = 1.0,
                 include_background = False, smooth = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.include_background = include_background
        self.smooth = smooth

    def forward(self,logits,target):
        ce = F.cross_entropy(logits,target)
        probs = logits.softmax(dim=1)

        onehot = F.one_hot(target,self.num_classes).movedim(-1,1).float()
        dims = (0,)+ tuple(range(2,logits.ndim))
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.smooth)/(denom + self.smooth)
        if not self.include_background:
            dice = dice[1:]
        dice_loss = 1 -dice.mean()

        return (self.dice_weight * dice_loss) + (self.ce_weight * ce)

@torch.no_grad()
def dice_per_class(logits , target,num_classes = 6,smooth = 1e-8):

    pred = logits.argmax(dim =1)
    scores = []
    for c in range(num_classes):
        p,t = pred==c,target == c
        inter = (p&t).sum().float()
        total = p.sum().float() + t.sum().float()
        scores.append(torch.tensor(1.0,device=logits.device)if total == 0
                      else 2 *inter / (total +smooth))

    return torch.stack(scores)

def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    for dims, shape in [(2,(2,1,128,128)),(3,(1,1,32,64,64))]:

        model = ImprovedUNet(dims=dims)
        x = torch.randn(*shape)
        y = model(x)
        print(f"{dims}D: {tuple(x.shape)} -> {tuple(y.shape)}, {count_parameters(model):,} params")
        target = torch.randint(0, 6, (shape[0],) + shape[2:])
        loss = DiceCeLoss()(y, target)
        loss.backward()
        print(f"  loss {loss.item():.4f} (backward OK)")
        print("  per-class Dice:", [round(v, 3) for v in dice_per_class(y, target).tolist()])