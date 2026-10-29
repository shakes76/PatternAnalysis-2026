import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Standard Double ocnvolution Block:
       [Conv2d (3x3) -> BatchNorm -> ReLU] x 2

       Preserves spatial dimensions (H, W) using padding=1.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """ Residual Convolution block.

        Main Path:
            [Conv2d (3x3) -> BatchNorm -> ReLU] x2
            (the second ReLU is applied after adding the shortcut)

        Shortcut path:
            Identity if the number of input and output channels matches.
            Otherwise, use a 1x1 convolution to match the channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Main convolution path: extract features using 2 convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut path: match channels if necessary
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Save or transform the input for the residual shortcut
        identity = self.shortcut(x)

        # Main path
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Combine the convolution features with the shortcut.
        out = out + identity
        out = self.relu(out)

        return out

class Baseline_UNet(nn.Module):
    """Standard 2D U-Net Baseline Model Architecture.
       Configured for 6-class pelvic MRI segmentation.

       Input tensor shape: (B, 1, 256, 128)
       Output logits shape: (B, 6, 256, 128)

       Arguments:
       in_channels (int): number of input channels (1 for grayscale MRI).
       num_classes (int): number of output segmentation classes (6 for HipMRI labels 0..5).
       base_filters (int): base feature channel capacity (default: 16).
    """
    def __init__(self, in_channels=1, num_classes=6, base_filters=16):
        super().__init__()

        # --- ENCODER: progressive spatial downsampling & feature extraction ---
        # Level 1: initial feature extraction without pooling -> (B, 16, 256, 128)
        self.enc1 = ConvBlock(in_channels, base_filters)
        # level 2: halves spatial dimensions, doubles channels -> (B, 32, 128, 64)
        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        # level 3: halves spatial dimensions, doubles channels -> (B, 64, 64, 32)
        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        # level 4: halves spatial dimensions, doubles channels -> (B, 128, 32, 16)
        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8)

        # 2x2 Max Pooling operator (stride=2, shared across downsampling levels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck: deepest feature representation (loest spatial resolution)
        # downsamples from level 4 features -> (B, 256, 16, 8)
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16)

        # --- DECODER: spatial upsampling and encoder skip feature fusion ---
        # Stage 4: upsample (16x8 -> 32x16), concat skip e4 (128 ch) -> 256 in, 128 out
        self.up4 = nn.ConvTranspose2d(
            base_filters * 16, base_filters * 8, 
            kernel_size=2, stride=2
        )
        self.dec4 = ConvBlock(base_filters * 16, base_filters * 8)

        # Stage 3: upsample (32x16 -> 64x32), concat skip e3 (64 ch) -> 128 in, 64 out
        self.up3 = nn.ConvTranspose2d(
            base_filters * 8, base_filters * 4, 
            kernel_size=2, stride=2
        )
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4)

        # Stage 2: upsample (64x32 -> 128x64), concat skip e2 (32 ch) -> 64 in, 32 out
        self.up2 = nn.ConvTranspose2d(
            base_filters * 4, base_filters * 2, 
            kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2)

        # Stage 1: upsample (128x64 -> 256x128), concat skip e1 (16 ch) -> 32 in, 16 out
        self.up1 = nn.ConvTranspose2d(
            base_filters * 2, base_filters, 
            kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock(base_filters * 2, base_filters)

        # Segmentation head: 1x1 convolution mapping 16 features to 6 class logits -> (B, 6, 256, 128)
        self.output_layer = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder forward pass with skip connections
        d4 = self.up4(b) # (B, 128, 32, 16)
        d4 = torch.cat([d4, e4], dim=1) # concaternate skip e4 -> (B, 256, 32, 16)
        d4 = self.dec4(d4) # Fuse features -> (B, 128, 32, 16)

        d3 = self.up3(d4) # (B, 64, 64, 32)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3) # (B, 32, 128, 64)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2) # (B, 16, 256, 128)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Final 1x1 convolution yielding unnormalized class logits 
        logits = self.output_layer(d1) # (B, 6, 256, 128) 
        return logits

class Improved_UNet(nn.Module):
    """2D Improved U-Net Architecture.
       Replaces standard double-convolution blocks with ResidualBlocks
       (residual shortcut connections) to evaluate whether residual 
       learning improves gradient propagation and prostate boundary segmentation.
    """
    def __init__(self, in_channels=1, num_classes=6, base_filters=16):
        super().__init__()

        # Encoder (Contracting Path): extract features at progressively smaller resolutions
        self.enc1 = ResidualBlock(in_channels, base_filters)
        self.enc2 = ResidualBlock(base_filters, base_filters * 2)
        self.enc3 = ResidualBlock(base_filters * 2, base_filters * 4)
        self.enc4 = ResidualBlock(base_filters * 4, base_filters * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck: deepest feature representation
        self.bottleneck = ResidualBlock(base_filters * 8, base_filters * 16)

        # Decoder: upsample and combine encoder features
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(base_filters * 16, base_filters * 8)

        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_filters * 2, base_filters)

        # Produce one output channel per segmentation class
        self.output_layer = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder stage 4
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        # Decoder stage 3
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        # Decoder stage 2
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        # Decoder stage 1
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Return raw logits; the loss function handles these
        logits = self.output_layer(d1)
        return logits    

if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING BASELINE & IMPROVED 2D U-NET FORWARD PASS SANITY TEST")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) 
    print(f"Device: {device}")

    # Simulate dummy batch of 2 MRI slices: (Batch=2, Channels=1, Height=256, Width=128) 
    dummy_input = torch.randn(2, 1, 256, 128).to(device) 
    print(f"Dummy Input Tensor Shape: {tuple(dummy_input.shape)}")

    # Test Baseline UNet
    b_model = Baseline_UNet(in_channels=1, num_classes=6, base_filters=16).to(device)
    b_params = sum(
        p.numel() for p in b_model.parameters() 
        if p.requires_grad
    ) 

    # Disable gradient tracking because this is only a sanity test to check
    # the output shape, not to train the model. This saves memory and avoids
    # unnecessary gradient calculations (same is applied to Improved Unet shape verification)
    with torch.no_grad():
        b_output = b_model(dummy_input)

    assert b_output.shape == (2, 6, 256, 128), (
        f"Shape mismatch! Expected (2, 6, 256, 128), "
        f"got {b_output.shape}"
    )

    # Test Imporved UNet
    i_model = Improved_UNet(in_channels=1, num_classes=6, base_filters=16).to(device)
    i_params = sum(
        p.numel() for p in i_model.parameters() 
        if p.requires_grad
    ) 

    with torch.no_grad():
        i_output = i_model(dummy_input) 

    assert i_output.shape == (2, 6, 256, 128), (
        f"Shape mismatch! Expected (2, 6, 256, 128), "
        f"got {i_output.shape}"
    )

    print(
        f"Baseline Trainable Parameters: {b_params:,} | Output Shape:"
        f" {tuple(b_output.shape)}"
    )
    print(
        f"Improved Trainable Parameters: {i_params:,} | Output Shape:"
        f" {tuple(i_output.shape)}"
    )

    print("\n" + "=" * 65) 
    print("SUCCESS: Baseline and Improved UNet passed shape verification!") 
    print("=" * 65)