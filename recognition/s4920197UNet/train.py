from dataset import TrainData, TestData, ValidationData
from modules import BaselineModel, PaperModel
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import torch.nn as nn
import torch
import sys
import time

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask, output):
        loss = 0.0
        epsilon = 1e-6
        dc = []
        for k in range(6):
            p = output[:, k, :, :]
            y = mask[:, k, :, :]

            p_flat = p.contiguous().view(-1)
            y_flat = y.contiguous().view(-1)
            intersection = torch.sum(p_flat * y_flat)
            total = torch.sum(p_flat) + torch.sum(y_flat)

            dice_coef = (2.0 * intersection + epsilon) / (total + epsilon)
            dc.append(dice_coef)
            loss += (1.0 - dice_coef)

        return loss/6, dc

class BasicUnet:
    def __init__(self, device, data_path, smoke_t=False):
        self.inference = []
        self.vram_usage = []
        self.model = BaselineModel().to(device)
        self.dev = device
        self.data_path = data_path 
        self.smoke_t = smoke_t
        
    def test_model(self):
        self.model.eval()
        testing = TestData(self.data_path)       
        test = DataLoader(
            dataset=testing,
            batch_size= 1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        dice_scores = [0,0,0,0,0,0]
        with torch.no_grad():
            total = testing.__len__() * 128 * 256
            correct = 0
            ni = 0
            start = time.time()
            for masks, images in test:
                ni += 1
                masks = masks.to(self.dev)
                images = images.to(self.dev)
                outputs = self.model(images)
                epsilon = 1e-6
                
                for k in range(6):
                    p = outputs[:, k, :, :]
                    y = masks[:, k, :, :]

                    p_flat = p.contiguous().view(-1)
                    y_flat = y.contiguous().view(-1)
                    intersection = torch.sum(p_flat * y_flat)
                    correct += intersection
                    total = torch.sum(p_flat) + torch.sum(y_flat)

                    dice_scores[k] += (2.0 * intersection + epsilon) / (total + epsilon)

                if self.smoke_t:
                    break
            
            accuracy = correct / total
            start = time.time() - start
            start /= ni
            for d in dice_scores: 
                d /= ni

            iou = sum(dice_scores)/6 / (2 - sum(dice_scores)/6)
            print(f"average inference time: {start}")
            print(f"dice scores prer class: \n{dice_scores}")
            accuracy = correct / total
            print(f"average IoU score: {iou}")

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
        i = 0
        for hyperparameters in hyperparameters_set:
            print(f"starte training on hyperparametes set {i}\n{hyperparameters}")
            model = BaselineModel(
                hyperparameters["depth"], 
                hyperparameters["filters"], 
                num_classes=6
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
                num_workers=4,
            )

            for epoch in range(hyperparameters["epochs"]):
                print(f"starting epoch {epoch}")
                for mask, image in train:
                    mask = mask.to(device)
                    image = image.to(device)
                    output = model(image)
                    loss, class_dice = loss_fn(mask, output)
                    if i % 30 == 0:
                        print(f"{i}: loss = {loss}")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if self.smoke_t:
                        break
                sheduler.step()  
                if self.smoke_t:
                    break
            
            model.eval()
            correct = 0
            validate = DataLoader(
                dataset=validation,
                batch_size=2 if self.smoke_t else 32,
                shuffle=False,
                num_workers=4,
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
                best_model = (model.deepcopy(), correct)
                tries = early_stop
            elif tries == 0:
                self.model = model
                return

        self.model = model

    def save_model(self, path):
        torch.save(self.model, path)
    
    def load_model(self, path):
        self.model = torch.load(path, weights_only=False)

    def use(self):
        return self.model

class ImprovedUnet: 
    def __init__(self, device, data_path, smoke_t=False):
        self.model = PaperModel().to(device)
        self.dev = device
        self.data_path = data_path 
        self.smoke_t = smoke_t

    def test_model(self):
        self.model.eval()
        testing = TestData(self.data_path)       
        test = DataLoader(
            dataset=testing,
            batch_size= 1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        dice_scores = [0,0,0,0,0,0]
        with torch.no_grad():
            total = testing.__len__() * 128 * 256
            correct = 0
            ni = 0
            start = time.time()
            for masks, images in test:
                ni += 1
                masks = masks.to(self.dev)
                images = images.to(self.dev)
                outputs = self.model(images)
                epsilon = 1e-6
                
                for k in range(6):
                    p = outputs[:, k, :, :]
                    y = masks[:, k, :, :]

                    p_flat = p.contiguous().view(-1)
                    y_flat = y.contiguous().view(-1)
                    intersection = torch.sum(p_flat * y_flat)
                    correct += intersection
                    total = torch.sum(p_flat) + torch.sum(y_flat)

                    dice_scores[k] += (2.0 * intersection + epsilon) / (total + epsilon)

                if self.smoke_t:
                    break
            
            accuracy = correct / total
            start = time.time() - start
            start /= ni
            for d in dice_scores: 
                d /= ni

            iou = sum(dice_scores)/6 / (2 - sum(dice_scores)/6)
            print(f"average inference time: {start}")
            print(f"dice scores prer class: \n{dice_scores}")
            accuracy = correct / total
            print(f"average IoU score: {iou}")

    #no validation function and test function for this model, as it is our basline that we compare with our basic model
    def train_model(self):
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
            num_workers=4,
        )

        for epoch in range(2):
            print(f"starting epoch {epoch}")
            for i, (mask, image) in enumerate(train):
                mask = mask.to(device)
                image = image.to(device)
                output = self.model(image)
                loss, class_dice = loss_fn(mask, output)
                if i % 30 == 0:
                    print(f"{i}: loss = {loss}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if self.smoke_t:
                    break

            sheduler.step()   
            if self.smoke_t:
                break

    def save_model(self, path):
        torch.save(self.model, path)
    
    def load_model(self, path):
        self.model = torch.load(path, weights_only=False)
    
    def use(self):
        return self.model

    def save_model(self, path):
        torch.save(self.model, path)
    
    def load_model(self, path):
        self.model = torch.load(path, weights_only=False, map_location=self.dev)
    
    def use(self):
        return self.model