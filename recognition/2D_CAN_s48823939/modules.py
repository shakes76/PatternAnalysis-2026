import torch
import torch.nn as nn

class ContextAggregationModule(nn.Module):
    """The CAN. Layers 1-6 apply 3x3 convolutions with exponentially increasing 
    dilation factors (1,1,4,8,16), layer 7 is a 3x3 refinement (dilation 1) and 
    layer 8 is 1x1 projection from `C = 64` intermediate channels to the 4 target classes. 
    """
    def __init__(self, in_channels=1, num_classes=4, C=64):
        super().__init__()
        
        # Exact 8 layer sequential architecture from Yu & Koltun (ICLR 2016)
        # Padding must equal dilation for 3x3 kernels to preserve spatial dimensions (256x256)
        self.layer1 = nn.Conv2d(in_channels, C, kernel_size=3, padding=1, dilation=1)
        self.layer2 = nn.Conv2d(C,           C, kernel_size=3, padding=1, dilation=1)
        self.layer3 = nn.Conv2d(C,           C, kernel_size=3, padding=2, dilation=2)
        self.layer4 = nn.Conv2d(C,           C, kernel_size=3, padding=4, dilation=4)
        self.layer5 = nn.Conv2d(C,           C, kernel_size=3, padding=8, dilation=8)
        self.layer6 = nn.Conv2d(C,           C, kernel_size=3, padding=16, dilation=16)
        self.layer7 = nn.Conv2d(C,           C, kernel_size=3, padding=1, dilation=1)
        
        # Layer 8 is a 1x1 convolution projecting to the 4 segmentation classes.
        self.layer8 = nn.Conv2d(C, num_classes, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        self._init_weights()
        
    def _init_weights(self):
        """ Yu & Koltun discovered that to prevent model hallucination and 'telephone game'-ness. 
        They set every weight in the network to 0 except for the dead-centre pixel of the 3x3 
        convolutional filter for the matching channels, which is set to 1.
        
        This acts as a perfect pass through.
        
        The model therefore does the following:
            The brain slice passes through smoothly with all 7 layers completely unaltered.
            The network makes a guess
            The loss function adjusts
            Backpropagation kicks in
            
        Because the signal wasn't randomly scrambled nonsense the signal can travel back through the network flawlessly.
        This allows the network to actually TRAIN
        """
        
        layers_to_init = [self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]
        
        for layer in layers_to_init:
            # Whiteout.
            nn.init.constant_(layer.weight, 0.0)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
                
            # Find the exact centre pixel of the 3x3 filter
            # A 3x3 matrix has indices: top=[0], middle=[1], bottom=[2]
            # So index [1,1] is the dead centre
            min_channels = min(layer.out_channels, layer.in_channels)
            for i in range(min_channels):
                # If input channel = output channel, (i,i), set the center pixel to 1.
                layer.weight.data[i,i,1,1] = 1.0
                      
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        return self.layer8(x)
    
    
## 2D UNET stuff below.
class Simple2DUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, C=32):
        super().__init__()
        
        # Encoder
        self.enc1_1 = nn.Conv2d(in_channels, C, 3, padding=1)
        self.enc1_2 = nn.Conv2d(C, C, 3, padding=1)
        
        self.enc2_1 = nn.Conv2d(C, C*2, 3, padding=1)
        self.enc2_2 = nn.Conv2d(C*2, C*2, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1_1 = nn.Conv2d((C*2) + C, C, 3, padding=1)
        self.dec1_2 = nn.Conv2d(C, C, 3, padding=1)
        
        self.out = nn.Conv2d(C, num_classes, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.enc1_1(x))
        x1 = self.relu(self.enc1_2(x1))
        
        p1 = self.pool(x1)
        x2 = self.relu(self.enc2_1(p1))
        x2 = self.relu(self.enc2_2(x2))
        
        # Decoder with skip connection
        d1 = self.up1(x2)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.relu(self.dec1_1(d1))
        d1 = self.relu(self.dec1_2(d1))
        
        return self.out(d1)