from dataset import HipMRI3D, HipMRI2D, get_training_test_split
from modules import CAN3D, CAN2D, compute_loss, compute_loss_2d, calculate_descrete_dsc
import torch
import torch.nn as nn
import time

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch.optim as optim

import monai.transforms as mt

augmentations = mt.Compose([
    mt.Rand3DElasticd(keys=["image", "label"], prob=0.3, mode=["bilinear", "nearest"], sigma_range=(5, 7), magnitude_range=(50, 120), padding_mode="zeros"),
    mt.RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
])
def transform_training_data(images, labels):
    #Note: extra dimension it technically the batch, but its size is 1, so it may as well be channel
    augmented = augmentations({"image": images, "label": labels})
    return augmented['image'], augmented['label']


def train_3d():
    print("Starting...")
    # Hyper Parameters
    batch_size = 8
    learning_rate = 0.01
    num_epochs = 100

    # Create Database
    database = HipMRI3D(early_stop=0)
    train_set, val_set, test_set = get_training_test_split(database, test_size=0.2, val_size=0.1)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Select Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    # Train Models
    model = CAN3D()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_steps = (len(train_loader) * num_epochs) // batch_size
    schedular = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)

    start = time.time()
    model.train()

    validation_scores = []
    loss_scores = []

    weights = torch.tensor([0.03, 0.03, 0.6, 0.3, 8.0, 18.0]).to(device)

    steps = 0
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Load Data
            images = images.to(device)
            labels = labels.to(device)
            images, labels = transform_training_data(images, labels)
            labels = labels.long()

            # Evaluate and Backpropogate
            predictions = model(images)
            loss = compute_loss(predictions, labels, weights, weights * 3, gamma=2) / batch_size
            loss.backward()

            if (steps + 1) % batch_size == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                schedular.step()
            steps += 1

            if i == 0 and epoch == 0:
                print("First Loss: {:.5f}".format(loss.item() * batch_size), flush=True)

        # Save the model every tenth epoch
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 'parameters_3d_epoch_{}.pth'.format(epoch))

        # Perform Validation Calculation
        total_sum = torch.zeros((6), device=device)
        total_validation_samples = 0

        with torch.inference_mode():
            for i, (image, label) in enumerate(val_loader):
                image = image.to(device)
                label = label.to(device).long()

                predictions = model(image)

                sum_dice_per_class, no_in_batch = calculate_descrete_dsc(predictions, label)
                total_sum += sum_dice_per_class
                total_validation_samples += no_in_batch

        total_sum = total_sum / total_validation_samples
        validation_scores.append(total_sum[5].item())
        loss_scores.append(loss.item())

        print("Epoch [{}/{}],  Loss: {:.5f}, Validation: {} Time: {} secs"
            .format(epoch+1, num_epochs, loss.item(), total_sum, time.time() - start), flush=True)

    torch.save(model.state_dict(), 'final_parameters_3d.pth')

    plt.subplot(1, 2, 1)
    plt.plot(list(range(num_epochs)), loss_scores)
    plt.title("Epochs vs Loss")
    plt.subplot(1, 2, 2)
    plt.plot(list(range(num_epochs)), validation_scores)
    plt.title("Epochs vs Validation Score")
    plt.savefig("validation_curves.png")


def volume_to_2d_batch(volume : torch.Tensor):
    # take height-wise slices
    return volume.permute(1, 0, 2)

def batch_output_to_3d_volume_batch(batch : torch.Tensor):
    return batch.permute(1, 2, 0, 3)[None,:,:,:,:]

def train_2d():
    print("Starting...")
    # Hyper Parameters
    batch_size = 8
    learning_rate = 0.01
    num_epochs = 70

    # Create Database
    database = HipMRI3D(early_stop=0)
    train_set, val_set, test_set = get_training_test_split(database, test_size=0.2, val_size=0.1)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Select Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    # Train Models
    model = CAN2D()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_steps = (len(train_loader) * num_epochs) // batch_size
    schedular = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)

    start = time.time()
    model.train()

    validation_scores = []
    loss_scores = []

    weights = torch.tensor([0.03, 0.03, 0.6, 0.3, 8.0, 18.0]).to(device)

    steps = 0
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Load Data
            images = images.to(device)
            labels = labels.to(device)
            images, labels = transform_training_data(images, labels)
            labels = labels.long()

            # Evaluate and Backpropogate
            predictions = batch_output_to_3d_volume_batch(model(volume_to_2d_batch(images[0])))
            loss = compute_loss(predictions, labels, weights, weights * 3, gamma=2) / batch_size
            loss.backward()

            if (steps + 1) % batch_size == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                schedular.step()
            steps += 1

            if i == 0 and epoch == 0:
                print("First Loss: {:.5f}".format(loss.item() * batch_size), flush=True)

        # Save the model every tenth epoch
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 'parameters_2d_epoch_{}.pth'.format(epoch))

        # Perform Validation Calculation
        total_sum = torch.zeros((6), device=device)
        total_validation_samples = 0

        with torch.inference_mode():
            for i, (image, label) in enumerate(val_loader):
                image = image.to(device)
                label = label.to(device).long()

                predictions = batch_output_to_3d_volume_batch(model(volume_to_2d_batch(images[0])))

                sum_dice_per_class, no_in_batch = calculate_descrete_dsc(predictions, label)
                total_sum += sum_dice_per_class
                total_validation_samples += no_in_batch

        total_sum = total_sum / total_validation_samples
        validation_scores.append(total_sum[5].item())
        loss_scores.append(loss.item())

        print("Epoch [{}/{}],  Loss: {:.5f}, Validation: {} Time: {} secs"
            .format(epoch+1, num_epochs, loss.item(), total_sum, time.time() - start), flush=True)

    torch.save(model.state_dict(), 'final_parameters_2d.pth')

    plt.subplot(1, 2, 1)
    plt.plot(list(range(num_epochs)), loss_scores)
    plt.title("Epochs vs Loss")
    plt.subplot(1, 2, 2)
    plt.plot(list(range(num_epochs)), validation_scores)
    plt.title("Epochs vs Validation Score")
    plt.savefig("validation_curves_2d.png")

#train_2d()
train_3d()