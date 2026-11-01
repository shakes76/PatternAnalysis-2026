"""
modules.py: Containing the source code of the components of your model. Each component must be
implemented as a class or a function in PyTorch or TensorFlow/Keras. Your implementation of the
model should not depend on NumPy (or other Python libraries not part of PyTorch / TensorFlow)
in any way unless otherwise approved by the teaching staff.
"""

import torch.nn as nn

KERNEL_SIZE 		= 3
LEAKY_RELU_SLOPE 	= 0.01

class DilatedBlock(nn.Module):
	"""
	The fundamental element of the CAN network, containing a convolution, normalization and relu activation layer.
	"""
	def __init__(self, in_ch, out_ch, dilation, dim=3):
		super().__init__()
		Conv = nn.Conv3d if dim == 3 else nn.Conv2d
		Norm = nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d
		self.conv = Conv(in_ch, out_ch, KERNEL_SIZE, padding=dilation, dilation=dilation)
		self.norm = Norm(out_ch)
		self.act = nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True)

	def forward(self, x):
		return self.act(self.norm(self.conv(x)))

class CAN(nn.Module):
	"""
	The network: A daisychain of 5 DilatedBlocks with increasing dilations to improve receptive field.
	An output convolutional layer turning the NETWORK_WIDTH feature maps into 6 classes.
	Works for both 2d and 3d
	dim=2 gives the baseline, dim=3 gives CAN3D.
	dilation = (Depth, Height, Width)
	"""
	def __init__(self, in_ch=1, num_classes=6, width=24,
				 dilations=((1,1,1), (2,2,2), (2,4,4), (2,8,8), (1,1,1)), dim=3):
		super().__init__()
		layers, ch = [], in_ch
		for dilation in dilations:
			dilation = dilation if dim == 3 else dilation[1]
			layers.append(DilatedBlock(ch, width, dilation, dim))
			ch = width
		self.body = nn.Sequential(*layers)
		Conv = nn.Conv3d if dim == 3 else nn.Conv2d
		self.head = Conv(width, num_classes, kernel_size=1)

	def forward(self, x):
		return self.head(self.body(x))