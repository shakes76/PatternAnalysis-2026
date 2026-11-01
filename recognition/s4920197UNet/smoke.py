from train import BasicUnet, ImprovedUnet
from dataset import TrainData
import nibabel as nib
import torch
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


load = sys.argv[-1] == 'load'
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = "/home/andrey/Documents/comp3710/project/keras_slices_data"
train = TrainData(data)
dl = DataLoader(
    dataset = train,
    batch_size = 1,
    shuffle = True
)

unet = BasicUnet(dev, data, True)
if load:
    unet.load_model("/home/andrey/Documents/comp3710/project/model.pt")
else:
    unet.train_model()
    unet.test_model()
    unet.save_model("/home/andrey/Documents/comp3710/project/model.pt")
model = unet.use()
for masks, images in dl:
    mask = masks
    inp = images
    out = model(inp)
    print(out.shape)
    print(out)
    break

improved_unet = ImprovedUnet(dev, data, True)
if load:
    improved_unet.load_model("/home/andrey/Documents/comp3710/project/comparison_model.pt")
else:
    improved_unet.train_model()
    improved_unet.save_model("/home/andrey/Documents/comp3710/project/comparison_model.pt")
model = improved_unet.use()
for masks, images in dl: 
    mask = masks
    inp = images
    out = model(inp)
    print(out.shape)
    print(out)
    break