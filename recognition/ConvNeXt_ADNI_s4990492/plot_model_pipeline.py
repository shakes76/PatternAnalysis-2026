"""Render the custom RGB224 architecture from its documented dimensions."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

base=Path(__file__).resolve().parent
fig,ax=plt.subplots(figsize=(11,4.2),layout='constrained')
ax.set(xlim=(0,11),ylim=(0,4.2)); ax.axis('off')
labels=['MRI slice\nRGB224 padding\n3 x 224 x 224','Stem\n4 x 4, stride 4\n96 x 56 x 56','Four stages\n3 / 3 / 9 / 3 blocks\n96 / 192 / 384 / 768','Global average\nLayerNorm\nLinear 768 to 2','Softmax\nNC / AD\nThreshold 0.50']
for i,label in enumerate(labels):
    x=.1+i*2.2
    ax.add_patch(FancyBboxPatch((x,2.35),1.95,1.25,boxstyle='round,pad=0.03',facecolor='#edf3fa',edgecolor='#325b85'))
    ax.text(x+.975,2.975,label,ha='center',va='center',fontsize=10)
    if i<4: ax.annotate('',xy=(x+2.17,2.975),xytext=(x+1.98,2.975),arrowprops=dict(arrowstyle='->',color='#325b85'))
ax.text(5.5,1.85,'Each ConvNeXt block',ha='center',fontsize=12,weight='bold')
ax.text(5.5,1.25,'7x7 depthwise convolution  >  LayerNorm  >  Linear (C to 4C)  >  GELU\n>  Linear (4C to C)  >  learned layer scale  >  residual addition',ha='center',va='center',fontsize=11,linespacing=1.6)
ax.text(5.5,.45,'Between stages: LayerNorm + 2x2 stride-2 downsampling. Final feature map: 768 x 7 x 7.\nCustom PyTorch architecture; ImageNet backbone initialization; randomly initialized binary head.',ha='center',fontsize=9,color='#404040')
fig.savefig(base/'figures/model_pipeline.png',dpi=180)
plt.close(fig)
