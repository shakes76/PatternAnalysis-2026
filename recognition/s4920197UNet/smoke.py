from train import BasicUnet, ImprovedUnet
from dataset import TrainData
import nibabel as nib
import torch
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wd = sys.argv[1]
print(wd)
data = f"{wd}/keras_slices_data"
train = TrainData(data)
dl = DataLoader(
    dataset = train,
    batch_size = 1,
    shuffle = True
)

unet = BasicUnet(data, True)
unet.train_model()
model = unet.model
for masks, images in dl:
    mask = masks
    inp = images
    out = model(inp)
    print(out.shape)
    print(out)
    break

improved_unet = ImprovedUnet(data, True)
improved_unet.train_model()
model = improved_unet.model
for masks, images in dl: 
    mask = masks
    inp = images
    out = model(inp)
    print(out.shape)
    print(out)
    break