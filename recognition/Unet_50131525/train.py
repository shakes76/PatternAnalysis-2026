import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import UNet3d
from dataset import HipMRIDataset
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

def dice_score(prediction,masks,class_id):
    epsilon = 1e-6
    pred_mask = (prediction == class_id)
    true_mask = (masks == class_id)

    #calculate the intersection,prediction and target
    intersection = torch.sum(pred_mask*true_mask)
    pred_total = torch.sum(pred_mask)
    target = torch.sum(true_mask)

    dice = ((2.0 * intersection + epsilon) / (pred_total+target+epsilon))
    return dice

def dice_loss(outputs,masks):
    probabilities = torch.softmax(outputs,dim=1)
    targets = nn.functional.one_hot(
        masks.long(), num_classes=6).permute(0,4,1,2,3).float()

    #calculate dice for each class
    dims = (0,2,3,4)
    intersection = torch.sum(probabilities*targets,dim = dims)
    denominator = torch.sum(probabilities+targets,dim = dims)

    dice = (2*intersection +1e-6)/(denominator+1e-6)

    return 1 - dice[1:].mean()

def main():
    data_dir = Path("/home/groups/comp3710/HipMRI_Study_open")

    #initialize datasets
    train_dataset = HipMRIDataset(data_dir, split="train")
    val_dataset = HipMRIDataset(data_dir, split="validate")

    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True)

    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)
    print(f"Train loader batches:{len(train_loader)}\n"
          f"Data loader batches:{len(val_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Check SLURM GPU allocation.")

    model = UNet3d(in_channels=1,num_classes=6).to(device)

    #define loss function
    criterion = nn.CrossEntropyLoss()

    def combined_loss(outputs,masks):
        if not isinstance(outputs, tuple):
            return criterion(outputs, masks) + dice_loss(outputs, masks)

        logits,aux2,aux3 = outputs
        main_loss = criterion(logits, masks) + dice_loss(logits, masks)
        mask2 = F.interpolate(
            masks.unsqueeze(1).float(),
            size = aux2.shape[2:],
            mode = "nearest"
        ).squeeze(1).long()

        mask3 = F.interpolate(
            masks.unsqueeze(1).float(),
            size=aux3.shape[2:],
            mode="nearest"
        ).squeeze(1).long()

        loss2 = criterion(aux2,mask2) + dice_loss(aux2,mask2)
        loss3 = criterion(aux3, mask3) + dice_loss(aux3, mask3)

        return main_loss+0.25*loss2+0.5*loss3

    #define optimizer
    optimizer = optim.AdamW(model.parameters(),lr = 0.0001)

    #Setting the number of epochs
    epochs = 20
    best_val_loss = float("inf")
    best_class5_dice = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss =0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs,masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0

        intersections = [0.0]*6
        pred_totals = [0.0]*6
        target_totals =[0.0]*6

        with torch.no_grad():
            for images,masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = combined_loss(outputs,masks)

                val_loss += loss.item()

                #convert logits into predicted class labels
                preds = torch.argmax(outputs,dim=1)

                for class_id in range(6):
                    pred_mask = (preds == class_id)
                    true_mask = (masks == class_id)

                    intersection = torch.sum(pred_mask * true_mask)
                    pred_total = torch.sum(pred_mask)
                    target = torch.sum(true_mask)

                    intersections[class_id] += intersection.item()
                    pred_totals[class_id] += pred_total.item()
                    target_totals[class_id] += target.item()

            for class_id in range(6):
                denominator = pred_totals[class_id] + target_totals[class_id]

                if denominator > 0:
                    dice = 2*intersections[class_id]/denominator
                    print(f"Class {class_id}: Dice = {dice:.4f}")
                else:
                    print(f"Class {class_id}: Dice = N/A (absent)")

        avg_val_loss = val_loss/len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")

        #Saving best model
        if best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_DeepSupervision_UNet3d.pth")

        denominator = pred_totals[5] + target_totals[5]
        if denominator > 0:
            class5_dice = 2*intersections[5]/denominator

            if class5_dice > best_class5_dice:
                best_class5_dice = class5_dice
                torch.save(model.state_dict(), "best_DeepSupervision_class5.pth")
                print(f"Saved best Class 5 model: Dice = {class5_dice:.4f}")


if __name__ == "__main__":
    main()