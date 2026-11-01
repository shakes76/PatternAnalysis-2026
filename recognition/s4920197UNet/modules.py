import torch
import torch.nn as nn
import torch.nn.functional as F

class PaperModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.depth = 5
        #aim to extract 16 learned filters on the first pass over the data
        first_filters = 16
        self.conv = nn.Conv2d(1, first_filters, kernel_size=3, padding=1)
        self.conv_last = nn.Sequential(
            nn.InstanceNorm2d(2*first_filters),
            nn.LeakyReLU(negative_slope=10e-2, inplace=True),
            nn.Conv2d(2*first_filters, 2*first_filters, kernel_size=3, padding=1)
        )

        self.context = nn.ModuleList([
            ContextModule(first_filters * 2**i) for i in range(self.depth)
        ])
        #reverse the order
        # for i in range(self.depth - 1, -1, -1) 2**(self.depth - 1 - i)
        self.aggregate = nn.ModuleList([
            #conv2d{16, 32 -> 32, 64 -> ... 
            nn.Sequential(
                nn.InstanceNorm2d(first_filters * 2**i),
                nn.LeakyReLU(negative_slope=10e-2, inplace=True),
                nn.Conv2d(first_filters * 2**i, first_filters * 2**(i+1), kernel_size=3, padding=1, stride=2) 
            )
            for i in range(self.depth - 1)
        ])
        self.upsample = nn.ModuleList([
            UpsamplingModule(first_filters * 2**i)
            for i in range(self.depth - 1, 0, -1)
        ])
        self.localise = nn.ModuleList([
            LocalizationModule(first_filters * 2**i)
            for i in range(self.depth - 1, 1, -1)
        ])

        self.segment = nn.Sequential(
            nn.InstanceNorm2d(first_filters * 2),
            nn.LeakyReLU(negative_slope=10e-2, inplace=True),
            nn.Conv2d(first_filters * 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        saved = []
        segments = []
        x = self.conv(x)
        ctx = self.context[0](x)
        x = x + ctx
        for i in range(1, self.depth):
            #save all but last layer
            saved.append(x)
            #norm + activate, then double the filters with a convolution
            #then put the result through the context module and add with its input
            #then save it to be used for upsampling
            x = self.aggregate[i - 1](x)
            ctx = self.context[i](x)
            x = x + ctx

        for i in range(self.depth - 1):
            #upsample -> concatonate with saved input from next level, then localize
            x = self.upsample[i](x)
            x = torch.cat((x, saved[self.depth - 2 - i]), dim=1)
            if i != 3:
                #last level, no need to localise
                x = self.localise[i](x)
        
        x = self.conv_last(x)
        x = self.segment(x)
        x = torch.softmax(x, dim=1)

        return x

class ContextModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.activation = nn.LeakyReLU(negative_slope=10e-2, inplace=True)
        self.basic_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        #take input normalize + activate it
        out = self.norm(x)
        out = self.activation(out)
        #first convolution
        out = self.basic_conv(out)
        #drop nuerons, then normalize + activate
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.activation(out)
        #2nd convolution
        out = self.basic_conv(out)
        return out

#localization halves the channel number
class LocalizationModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.activation = nn.LeakyReLU(negative_slope=10e-2, inplace=True)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.segment_conv = nn.Conv2d(channels, channels//2, kernel_size=1)

    def forward(self, x):
        # x = self.norm(x)
        # x = self.activation(x)
        x = self.conv(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.segment_conv(x)
        return x 

#upsampling also halves the channel number
class UpsamplingModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.norm = nn.InstanceNorm2d(channels)
        self.activation = nn.LeakyReLU(negative_slope=10e-2, inplace=True)
        self.conv = nn.Conv2d(channels, channels//2, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.scale(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x 

class BaselineModel(nn.Module):
    def __init__(self, depth=5, start=64, num_classes=6):
        super().__init__()
        self.depth = depth
        self.block1 = nn.Sequential(
            nn.Conv2d(1, start, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(start, start, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(start*2**(i-1),start*2**i, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(start*2**i,start*2**i,kernel_size=3, padding=1),
                nn.ReLU() 
            )
            for i in range(1, self.depth)
        ])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(start*2**i, start*2**(i-1), kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(start*2**(i-1), start*2**(i-1), kernel_size=3, padding=1),
                nn.ReLU()
            )
            for i in reversed(range(1, self.depth))
        ])

        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(start*2**i, start*2**(i-1), kernel_size=2, stride=2)
            for i in reversed(range(1,self.depth))
        ])

        self.segment = nn.Conv2d(start, num_classes, kernel_size=1)

    def forward(self, x):
        save = []
        x = self.block1(x)
        save.append(x)
        for block in self.blocks:
            save.append(x)
            x = self.pool(x)
            x = block(x)

        for upconv, image, block in zip(self.upconv, reversed(save), self.upblocks):
            x = upconv(x)
            x = torch.cat((image, x), dim=1)
            x = block(x)

        x = self.segment(x)
        x = torch.softmax(x, dim=1)

        return x