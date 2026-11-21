from dataset import TrainData, TestData, ValidationData
from modules import BaselineModel, PaperModel
from metrics import DiceLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import torch.nn as nn
import torch
import sys
import time
import copy
import heapq
import json
import matplotlib.pyplot as plt

'''The class that allows training of a basic Unet network that plots its peak VRAM and loss over time 
during training. Allows for saving and loading models, and tunes hyperparameters 
with random search over 12 possible sets'''
class BasicUnet:
    def __init__(self, data_path, smoke_t=False):
        self.data_vram = []
        self.data_loss = []
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BaselineModel().to(self.dev)
        self.data_path = data_path 
        self.smoke_t = smoke_t

    def train_model(self):
        training = TrainData(self.data_path)
        validation = ValidationData(self.data_path)
        device = self.dev

        best_model = (self.model, -999)
        loss_fn = DiceLoss()
        early_stop = 4
        tries = early_stop


        sample_space = {
            "lr": [5e-4, 1e-3],
            "batch_size": [2, 8, 16, 128],
            "filters": [16, 32, 64],
            "depth": [4, 5],
            "decay": [0.0, 1e-5],
            "gamma": [1, 0.985, 0.970],
            "epochs": [30, 45, 60]
        }

        hyperparameters_set = []
        for _ in range(12):
            while True:
                parameters = {
                    k: random.choice(v) for k,v in sample_space.items()
                }
                if parameters not in hyperparameters_set:
                    hyperparameters_set.append(parameters)
                    break

        for hyperparameters in hyperparameters_set:
            data_loss = []
            data_vram = []
            print(f"started training on hyperparametes set: \n{hyperparameters}")
            model = BaselineModel(
                hyperparameters["depth"], 
                hyperparameters["filters"], 
            ).to(self.dev).train()

            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=hyperparameters["lr"],
                weight_decay=hyperparameters["decay"]
            )

            sheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma = hyperparameters["gamma"]
            )

            train = DataLoader(
                dataset=training,
                batch_size=hyperparameters["batch_size"],
                shuffle=True,
            )

            N = 1 if self.smoke_t else hyperparameters['epochs']
            t = time.time()
            for epoch in range(N):
                print(f"starting epoch {epoch}")
                for i, (mask, image) in enumerate(train):
                    mask = mask.to(device)
                    image = image.to(device)
                    output = model(image)
                    loss, _class_dice = loss_fn(mask, output)
                    if i % 25 == 0:
                        data_loss.append(loss.item())
                        if torch.cuda.is_available():
                            vram = torch.cuda.memory_allocated()
                            data_vram.append(vram)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if self.smoke_t:
                        break

                sheduler.step()  
            
            model.eval()
            correct = 0
            validate = DataLoader(
                dataset=validation,
                batch_size=1 if self.smoke_t else 32,
                shuffle=False,
            )

            for masks, images in validate: 
                masks = masks.to(self.dev)
                images = images.to(self.dev)

                outputs = model(images)
                classes = torch.argmax(outputs, dim=1)
                one_hot = F.one_hot(classes, num_classes=6)
                one_hot = one_hot.permute(0, 3, 1, 2)
                correct += (one_hot == masks).sum().item()
                if self.smoke_t:
                    break
                    
            tries -= 1
            if best_model[1] < correct:
                best_model = (copy.deepcopy(model), correct)
                self.model = model
                self.data_vram = data_vram
                self.data_loss = data_loss
                self.save_model()
                tries = early_stop
                taken = time.time() - t
                print(f"training finished in {taken}s")
                with open("unet/time.json", "w") as f:
                    json.dump(taken, f)
            elif tries == 0:
                break
            if self.smoke_t:
                break
        
        self.model = best_model[0]

    def save_model(self):
        torch.save(self.model, 'unet/model.pt')
        with open('unet/loss.json', 'w') as f:
            json.dump(self.data_loss, f)
        with open('unet/vram.json','w') as f:
            json.dump(self.data_vram, f)
    
    def load_model(self, path):
        self.model = torch.load(path, weights_only=False, map_location=self.dev)

'''An implementation of the improved Unet, 
this class can load a pretrained model or train one and save it by itself. All training parameters
match the parameters used in the paper and training metrics like VRAM use and loss over time, training time are saved as figures'''
class ImprovedUnet: 
    def __init__(self, data_path, smoke_t=False):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model = PaperModel().to(self.dev)
        self.data_path = data_path 
        self.data_vram = []
        self.data_loss = []
        self.smoke_t = smoke_t

    #no validation function and test function for this model, as it is our basline that we compare with our basic model
    def train_model(self):
        t = time.time()
        self.model.train()
        device = self.dev
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=5e-4,
            weight_decay=1e-5
        )

        sheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = 0.985
        )

        loss_fn = DiceLoss()
        training = TrainData(self.data_path)
        train = DataLoader(
            dataset=training,
            batch_size=2,
            shuffle=True,
        )

        N = 1 if self.smoke_t else 45
        data_loss = []
        data_vram = []
        for epoch in range(N):
            print(f"starting epoch {epoch}")
            for i, (mask, image) in enumerate(train):
                mask = mask.to(device)
                image = image.to(device)
                output = self.model(image)
                loss, class_dice = loss_fn(mask, output)
                if i % 25 == 0:
                    data_loss.append(loss.item())
                    if torch.cuda.is_available():
                        data_vram.append(torch.cuda.memory_allocated())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.smoke_t: 
                    break

            sheduler.step()   

        with open("iunet/loss.json", "w") as f:
            json.dump(data_loss, f)

        with open("iunet/vram.json", "w") as f:
            json.dump(data_vram, f)

        t = time.time() - t
        print(f"training finished in {t}s")
        with open("iunet/time.json", "w") as f:
            json.dump(t, f)
        self.save_model()

    def save_model(self):
        torch.save(self.model, 'iunet/model.pt')
    
    def load_model(self, path):
        self.model = torch.load(path, weights_only=False, map_location=self.dev)

'''
Given a model (Unet or Improved Unet), calculates the per class Iou and Dice scores,
displays them together with the average inference time, and returns the images which produce
the lowest prostate dice scores 
'''
def test_model(model, data, smoke=False):
    model.eval()
    test = DataLoader(
        dataset=data,
        batch_size= 1,
        shuffle=False,
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lowest_three = [(999, None), (999, None), (999, None)]
    total_intersect = torch.zeros(6, dtype=torch.float64, device=dev)
    total_denom = torch.zeros(6, dtype=torch.float64, device=dev)
    total_union = torch.zeros(6, dtype=torch.float64, device=dev)

    with torch.no_grad():
        total = data.__len__() * 128 * 256
        correct = 0
        ni = 0
        start = time.time()
        for mask, image in test:
            ni += 1
            mask = mask.to(dev).bool()
            image = image.to(dev)
            output = model(image)
            output = torch.argmax(output, dim=1)
            out_prostate = (output == 5)
            mask_prostate = mask[:, 5].bool()
            output = F.one_hot(output, num_classes=6).movedim(-1,1).bool()
            intersect = (output & mask).sum(dim=(0,2,3)).float()
            denom = output.sum(dim=(0,2,3)).float() + mask.sum(dim=(0,2,3)).float()
            union = (output | mask).sum(dim=(0,2,3)).float()

            total_intersect += intersect
            total_denom += denom
            total_union += union

            if denom[5] == 0:
                continue

            score = 2 * intersect[5] / denom[5]
            el = (score, (image, output, mask))
            if score < lowest_three[0][0]:
                lowest_three.insert(0, el)
                lowest_three.pop()
            elif score < lowest_three[1][0]:
                lowest_three.insert(1, el)
                lowest_three.pop()
            elif score < lowest_three[2][0]:
                lowest_three.pop()
                lowest_three.append(el)

            if smoke:
                break
        
        start = time.time() - start
        start /= ni
        print(f"average inference time: {start}")

        dice_scores = torch.where(
            total_denom > 0, 
            2 * total_intersect / total_denom,
            torch.tensor(float('nan'), dtype=torch.float64, device=dev)
        )
        mean_dice = dice_scores
        print(f"dice scores per class: {mean_dice}")

        iou = torch.where(
            total_union > 0,
            total_intersect / total_union,
            torch.tensor(float('nan'), dtype=torch.float64, device=dev)
        )

        mean_iou = iou
        print(f"average per class IoU score: {mean_iou}")
        #returns the worst 3 images
        return lowest_three

# actually trains the model and plots the training curves    
if __name__ == "__main__":    
    wd = sys.argv[1]
    print(wd)
    data = f"{wd}/keras_slices_data"
    unet = BasicUnet(data)
    unet.train_model()
    # unet.load_model("unet/unet.pt")
    np = sum(p.numel() for p in unet.model.parameters())
    print(f"Unet parameter count: {np}")

    with open("unet/time.json", 'r') as f:
        data = json.load(f)
        print(f"training took {data/60} minutes")

    with open('unet/loss.json', 'r') as f:
        data = json.load(f)
        x = [25 * i for i in range(len(data))]
        plt.figure(figsize=(10,10))
        plt.title("Basic Unet training loss over time")
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.plot(x, data, color='b')
        plt.grid(True)
        plt.savefig("visuals/unet_loss.png")

    with open('unet/vram.json', 'r') as f:
        data = json.load(f)
        x = [25 * i for i in range(len(data))]
        y = [d / (1024 ** 3) for d in data]
        plt.figure(figsize=(10,10))
        plt.title("Basic Unet VRAM usage over time")
        plt.xlabel("iterations")
        plt.ylabel("VRAM (GB)")
        plt.plot(x, y, color='b')
        plt.grid(True)
        plt.savefig("visuals/unet_vram.png")

    improved_unet = ImprovedUnet(data)
    improved_unet.train_model()
    # improved_unet.load_model("iunet/improved-model.pt")
    np = sum(p.numel() for p in improved_unet.model.parameters())
    print(f"Improved Unet parameter count: {np}")
    with open("iunet/time.json", 'r') as f:
        data = json.load(f)
        print(f"training took {data/60} minutes")

    with open('iunet/loss.json', 'r') as f:
        data = json.load(f)
        x = [25 * i for i in range(len(data))]
        plt.figure(figsize=(10,10))
        plt.title("Improved Unet training loss over time")
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.plot(x, data, color='b')
        plt.grid(True)
        plt.savefig("visuals/iunet_loss.png")

    with open('iunet/vram.json', 'r') as f:
        data = json.load(f)
        x = [25 * i for i in range(len(data))]
        y = [d / (1024 ** 3) for d in data]
        plt.figure(figsize=(10,10))
        plt.title("Improved Unet VRAM usage over time")
        plt.xlabel("iterations")
        plt.ylabel("VRAM (GB)")
        plt.plot(x, y, color='b')
        plt.grid(True)
        plt.savefig("visuals/iunet_vram.png")
