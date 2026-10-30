import torch
import torch.nn as nn
import torch.nn.functional as F #funcs that don't manage internal learnable weights - for forward blocks


"""
InstanceNorm =
It normalizes the input per channel and per instance (in batch) by subtracting its mean and dividing by its standard deviation, then applies weights and biases.

input data = (N, C, H, W)
N = batch size
C = num of channels
H = height
W = width

"""





"""
Reusable set up for iterations of network layers:
sequencing of  conv -> instance norm -> leaky ReLU. 

"""
class ConvNormAct(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        #sequential creates the sequence of actions for nn forward passes - self.block is just a container function of the layers
        self.block = nn.Sequential(
            #padding set for same output dimensions as input based on stride = 1
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2),
            nn.InstanceNorm2d(out_ch, affine=True), #affine = True allows learned weights and biases to be store
            nn.LeakyReLU(0.01, inplace=True),
        )

    """One forward pass through a convolutional hidden layer -> out is a feature map"""
    def forward(self, x):
        return self.block(x)


"""
Context layer in paper architecture:
Encoder Block = 2 x Hidden Layers with a dropout applied in between. 
After, add the input to the outputted feature map as residual skip connection. (Residual = within)
(norm, act, conv) -> dropout -> (norm, act, conv), then add the input back (paper: p_drop = 0.3).

"""
class ContextBlock(nn.Module):

    def __init__(self, ch, p_drop=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.Dropout2d(p_drop),
            nn.InstanceNorm2d(ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

"""
Upsize feature map to double its size, then smooth/extract distinct details with conv while changing channel count to match skip connection
Upsample -> ConvNormAct

Input: In and Out size of step to match flow and skip connections
"""
class UpsamplingBlock(nn.Module):
    """Repeat pixels 2x2, then a 3x3 conv that reduces the channels."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest") #nearest = choose nearest pixel value without interpolating or smoothing
        self.conv = ConvNormAct(in_ch, out_ch, 3)

    def forward(self, x):
        return self.conv(self.up(x))
        

"""
Re-processing local and global featires from skip concat
3x3 ConvNormAct (Combine nearby features from skip and feature map combination) -> 
1x1 ConvNormAct (reduce double channels from concat back to decoder process order)

Input: in and out channel sizes
"""
class LocalizationBlock(nn.Module):
    """3x3 conv, then 1x1 conv to cut the channels after concatenation."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3 = ConvNormAct(in_ch, in_ch, 3)
        self.conv1 = ConvNormAct(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv1(self.conv3(x))
        

"""
Creat sequence of class instances for processing image through entire UNET structure.

Takes input of input size, number of segmentation classes and base filters and levels:
    this can then set up all layers for the input and filter sizes of each level of the neural network.
"""
class UNet2D(nn.Module):
    """2D adaptation of the Isensee et al. (2017) U-Net.
    Input  (N, in_channels, H, W); output logits (N, num_classes, H, W).
    H and W must be divisible by 2**(levels-1) = 16.
    """
    def __init__(self, in_channels=1, num_classes=6, base_filters=16, levels=5):
        super().__init__()
        f = [16, 32, 64, 128, 256] #number of filters / inputs and outputf of feature maps
        self.levels = levels

        # ---- encoder (context pathway) ----
        #expand channels -> set up 5x context modules -> set up 4 x in between conv sequence
        self.stem = nn.Conv2d(in_channels, f[0], 3, padding=1) #channels -> 16 (hard coded for now)
        self.encode = nn.ModuleList([ContextBlock(c) for c in f]) #set up context in and out sizes
        #set up conv in and out sizes (only 4 since in between context)
        self.down_sample = nn.ModuleList(
            [ConvNormAct(f[i], f[i + 1], 3, stride=2) for i in range(levels - 1)]) 
        

        # ---- decoder (localization pathway); decoder[i] works on level levels-2-i ----
        #(up_sample -> concat (torch) -> localize decoder) x3 
        #  + (up_sample -> concat skip -> 3x3 ConvNormAct) 
        #  + (combined segmentaion -> softmax)

        # four upsample steps (levels 3,2,1,0 in 0-indexed terms)
        self.up_sample = nn.ModuleList(
            [UpsamplingBlock(f[l + 1], f[l]) for l in reversed(range(levels - 1))])

        # localization modules for the first three steps only (levels 3,2,1)
        self.localize = nn.ModuleList(
            [LocalizationBlock(2 * f[l], f[l]) for l in reversed(range(1, levels - 1))])

        # the last step is a plain 3x3 conv on the concatenated 2*f[0] channels
        self.final_conv = ConvNormAct(2 * f[0], 2 * f[0], 3)
        
        # ---- segmentation and head interpolating for deep supervision: levels 2, 1, 0 ----
        """ deep supervision = simplified connections in long layer paths to reduce noise and weaker
        signals through depths of processing"""
        
        #number of channels at each segmentation extraction level 
        #(2* for the last one since no localization to reduce channels)
        head_channels = [2 * f[0], f[1], f[2]]   
        self.seg = nn.ModuleList(
            [nn.Conv2d(c, num_classes, 1) for c in head_channels])

    def forward(self, x):
        # ---- encoder: keep each level's output for the skip connections
        
        #first stem level
        x = self.encode[0](self.stem(x))
        skips = [x]

        #process all levels of encoding and downsampling - storing skips
        for l in range(1, self.levels):
            x = self.encode[l](self.down_sample[l - 1](x))
            skips.append(x)

        # ---- decoder: from the bottom level back up to level 0
        out = None

        #loop all size levels in reverse for in and out channel sizes
        out = None
        for i, l in enumerate(reversed(range(self.levels - 1))):
            x = self.up_sample[i](x)
            x = torch.cat([skips[l], x], dim=1)
            x = self.localize[i](x) if l > 0 else self.final_conv(x)
            
            if l <= 2:
                s = self.seg[l](x) # all segmented outputs have channel num = number of classes for final output
                if out is None:
                    out = s
                else:
                    out = F.interpolate(out, scale_factor=2, mode="nearest") + s
                    
        return out


"""
Input: epsilon - added to top and bottom of dice function - safeguard against dividing by zero

Dice Loss function per batch:
Pros:

Stability where classes less present - if class is not there or close to zero, it won't sway loss
(in turn learning weights) as much as an individual image
simplicity - one number per batch
less noise - hard to distinguish images won't swing loss completely

Consequences: Network isnt trained to get images correct independently:

Slices with lots of prostate count for more than slices with a little, so small or faint cases can get less attention.
A small or unlucky batch can have very little prostate in it, which makes the loss less reliable.

"""
class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon # = a tiny number (1e-5) added to the Dice fraction to avoid dividing by zero

    """
    logits = raw unnormalized output from decoder
    target = labels for the batch of images
    """
    def forward(self, logits, targets):
        #convert raw output into class probabilities (with each class probability in its own channel)
        probs = F.softmax(logits, dim=1)        # (N, 6, H, W) probabilities
        num_classes = probs.shape[1]

        dice_per_class = []
        for k in range(num_classes):            # k = 0, 1, ..., 5
            p = probs[:, k]                     # class k's probability map, (N, H, W)
            t = targets[:, k]                    # class k's true mask (0s and 1s), (N, H, W)

            overlap = (p * t).sum()             # intersect: how much predicted probability lands on true pixels
            total = p.sum() + t.sum()           # predicted amount + true amount
            dice_k = (2 * overlap + self.epsilon) / (total + self.epsilon)
            dice_per_class.append(dice_k)

        mean_dice = torch.stack(dice_per_class).mean()
        return 1 - mean_dice

"""
For measuring performance validation
Return: summary of intersect total of true and predicted for each class - breakdown of diceloss contributors seperated by class
"""
@torch.no_grad()
def dice_counts(logits, target):
    predicted_class = logits.argmax(dim=1)      # (N, H, W): picks the highest-scoring class at each pixel
    num_classes = logits.shape[1]
    
    overlaps = []
    totals = []
    for k in range(num_classes):
        pred_k = (predicted_class == k).float() # 1 where we predicted class k, else 0
        true_k = target[:, k]                   # 1 where the truth is class k, else 0

        overlaps.append((pred_k * true_k).sum())
        totals.append(pred_k.sum() + true_k.sum())

    return torch.stack(overlaps), torch.stack(totals)

