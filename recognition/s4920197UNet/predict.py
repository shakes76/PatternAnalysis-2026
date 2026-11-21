import sys 
from train import BasicUnet, ImprovedUnet, test_model
from torch.utils.data import DataLoader
from dataset import TestData
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

def draw_figures(worst_images, model):
    for i, (_dice, (image, out_mask, truth)) in enumerate(worst_images):
        out_mask = out_mask.to(torch.uint8)
        out_mask = torch.argmax(out_mask, dim=1)
        out_mask = out_mask[0]

        truth = truth.to(torch.uint8)
        truth = torch.argmax(truth, dim=1)
        truth = truth[0]

        h, w = truth.shape
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        error_map = torch.zeros((256, 128, 3), dtype=torch.uint8, device=dev)

        correct = (truth == out_mask) & (truth > 0)
        wrong_predict = (truth == 0) & (out_mask > 0)
        missed_predict = (out_mask == 0) & (truth > 0)
        wrong_class = (truth != out_mask) & (truth > 0) & (out_mask > 0)
        error_map[correct] = torch.tensor([0, 255, 0], dtype=torch.uint8, device=dev)
        error_map[wrong_predict] = torch.tensor([255, 0, 0], dtype=torch.uint8, device=dev)
        error_map[missed_predict] = torch.tensor([0, 0, 255], dtype=torch.uint8, device=dev)
        error_map[wrong_class] = torch.tensor([255, 255, 0], dtype=torch.uint8, device=dev)

        plt.figure(figsize=(10,10))
        plt.title(f"{model}\nError map (G = correct, R = false positive, B = false negative, Y = wrong prediction)")
        plt.imshow(error_map.detach().cpu().numpy())
        plt.savefig(f"visuals/{model}_err{i}.png")
        plt.close()
        plt.figure(figsize=(10,10))
        plt.imshow(image[0][0])
        plt.savefig(f"visuals/{model}_image{i}.png")
        plt.close()

        for j, cname in enumerate(['background','body','bone', 'bladder','rectum','prostate']):
            error_map = torch.zeros((256, 128, 3), dtype=torch.uint8, device=dev)
            correct = (truth == j) & (out_mask == j)
            wrong_predict = (truth != j) & (out_mask == j)
            missed_predict = (out_mask != j) & (truth == j)

            error_map[correct] = torch.tensor([0, 255, 0], dtype=torch.uint8, device=dev)
            error_map[wrong_predict] = torch.tensor([255, 0, 0], dtype=torch.uint8, device=dev)
            error_map[missed_predict] = torch.tensor([0, 0, 255], dtype=torch.uint8, device=dev)

            plt.figure(figsize=(10,10))
            plt.title(f"{cname} segmentation\nG = Correct segmentation, R = false positive, B = false negative")
            plt.imshow(error_map.detach().cpu().numpy())
            plt.savefig(f"visuals/{model}_{cname}_err{i}.png")
            plt.close()

unet_path = sys.argv[1]
iunet_path = sys.argv[2]
data_path = sys.argv[3]
data = TestData(data_path)       
unet = BasicUnet(data_path)
unet.load_model(unet_path)
worst_unet = test_model(unet.model, data)
draw_figures(worst_unet, 'unet')
iunet = ImprovedUnet(data_path)
iunet.load_model(iunet_path)
worst_iunet = test_model(iunet.model, data)
draw_figures(worst_iunet, 'iunet')

