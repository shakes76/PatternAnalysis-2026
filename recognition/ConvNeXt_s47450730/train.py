# This file performs the training and validation for a model, used for different experiments
# It does not perform any analysis using the test dataset, as this is to be used after all experiments have completed
# this file records important metrics and saves important data used for analysis after training has completed

# Training and evaluation procedured here were adapted from the code on blackboard for the cnn MNIST model training
# This code has been refactored into functions, and adapted to add in metrics to record training/validation accuracy,
# latency, peak VRAM, f1 scores and relevant values to create a confusion matrix

import time
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from modules_convnext import classifier
from dataset import train_loader, validation_loader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_vram = torch.cuda.get_device_properties(device).total_memory

if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU.")


def test_accuracy(model, data_loader):
    """
    Calculates the accuracy of a model on any given dataset 
    
    Given a model and a data loader, calculates the prediction on each image in the dataset and 
    caluclates the overall accuracy

    Returns: (float) accuracy 
    
    """
    was_training = model.training # important to ensure model is restored to state after evaluation 
    model.eval() # ensure model is in evaluation mode

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            # Get predictions for this batch
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train(was_training) # restore model
    return correct / total


MAX_TRAINING_TIME = 55 * 60 # give buffer of 5 minutes for other tasks

def train_model(model, data_loader, num_epochs):
    """
    Trains the model based off the given data_loader

    The model will train until it finishes the number of epochs provided, or when the max training time is up

    Returns: (epochs, losses, test_accuracies, train_accuracies) as arrays to be used later for plotting of 
    training progrssion.
    """

    # These fields will be saved and reloaded later to plot graphs to see how the model progressed during training
    epochs = [x + 1 for x in range(num_epochs)]
    losses = []
    test_accuracies = []
    train_accuracies = []

    print("> Training")
    start = time.time()

    # Reset peak GPU memory statistics
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(num_epochs):

        if time.time() - start > MAX_TRAINING_TIME:
            break

        model.train()

        loss = None
        running_loss = 0
        for i, (images, labels) in enumerate(data_loader): #train_loader

            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Prints the loss throughout training
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{len(data_loader)}], "
                    f"Loss: {loss.item():.5f}"
                )

            running_loss += loss.item()

        # at the end of each epoch, we need to store data representing the model's current state
        if (loss is not None):
            losses.append(running_loss / (i + 1))
        else:
            print("this should not happen, please check implementation")
        test_accuracies.append(test_accuracy(model, validation_loader))
        train_accuracies.append(test_accuracy(model, train_loader))

    elapsed = time.time() - start

    # Peak GPU memory used during training
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated(device)
        print(f"Peak GPU memory allocated: {peak_allocated / (1024**2):.2f} MiB") # max VRAM statistics

    print(f"Training took {elapsed:.2f} seconds "
        f"({elapsed / 60:.2f} minutes)")


    return(epochs, losses, test_accuracies, train_accuracies) 


def validate_model(model, data_loader):
    """
    Evaluates the provided model with the given data loader

    Keeps track of all metrics (false positives, false negatives etc...) to calculated precision, recall, f1 score and accuracy

    Returns: (actual_labels, predicted_labels) - these are lists of the predictions made by the model and actual labels for the data
    This is to be used after evaluation to perform further analysis and plot confusion matricies
    """
    print("> Evaluating on validation set")
    start = time.time()

    model.eval()

    total_AD = 0
    total_NC = 0
    correct_AD = 0
    correct_NC = 0
    correct = 0
    total = 0
    false_negative = 0
    false_positive = 0

    actual_labels = []
    predicted_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            # Get predictions for this batch
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)

            # add labels and predictions to accumulating lists
            actual_labels.extend(labels.cpu().tolist())
            predicted_probs.extend(
                probabilities[:, 1].cpu().tolist()
            )

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Class masks - this is so we can work out false positives, false negatives etc
            AD_mask = labels == 1
            NC_mask = labels == 0

            total_AD += AD_mask.sum().item()
            total_NC += NC_mask.sum().item()

            # Correct predictions
            correct_AD += (
                (predicted == 1) & AD_mask
            ).sum().item()

            correct_NC += (
                (predicted == 0) & NC_mask
            ).sum().item()

            # Incorrect predictions
            false_negative += (
                (predicted == 0) & AD_mask
            ).sum().item()

            false_positive += (
                (predicted == 1) & NC_mask
            ).sum().item()

    elapsed = time.time() - start
    print(f"Evaluation took {elapsed:.2f} seconds "
      f"({elapsed / 60:.2f} minutes)")

    # Calculate metrics safely
    accuracy = correct / total

    precision_denominator = correct_AD + false_positive
    recall_denominator = correct_AD + false_negative

    precision = (
        correct_AD / precision_denominator
        if precision_denominator > 0 else 0
    )

    recall = (
        correct_AD / recall_denominator
        if recall_denominator > 0 else 0
    )

    f1_denominator = precision + recall
    f1_score = (
        2 * precision * recall / f1_denominator
        if f1_denominator > 0 else 0
    )

    print(f"Validation Accuracy: {100 * accuracy:.2f}%")
    print(f"Precision (AD):      {100 * precision:.2f}%")
    print(f"Recall (AD):         {100 * recall:.2f}%")
    print(f"F1 Score (AD):       {f1_score:.4f}")
    print(f"True Positives:      {correct_AD}")
    print(f"True Negatives:      {correct_NC}")
    print(f"False Positives:     {false_positive}")
    print(f"False Negatives:     {false_negative}")
    print(f"Total AD images:     {total_AD}")
    print(f"Total NC images:     {total_NC}")

    return (actual_labels, predicted_probs) # so we can pickle this data and evaluate later

def calculateLatency(model, data_loader):
    """
    Calculates the average latency for a given model to make one prediction
    """
    model.eval()
    images, _ = next(iter(data_loader))
    single_image = images[:1].to(device)

    timed_iterations = 100 # do 100 predictions and divide the time by this so the measurement is more accuracte

    with torch.no_grad():

        # waits until all previous queued jobs on the gpu have run first
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter() # used for benchmarking, more accurate than time.time()

        with torch.no_grad():
            for _ in range(timed_iterations):
                _ = model(single_image)

        # wait until the jobs we just queued have run first before we measure the time again
        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    return (elapsed / timed_iterations) * 1000


#script to run it all

# Hyperparameters
num_epochs = 150 # we will use the 1 hour mark to early terminate, it will not get to this many epochs
learning_rate = 1e-3

model = classifier.to(device) # this is our imported classifier model

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#train the model

(epochs, losses, test_accuracies, train_accuracies) = train_model(model, train_loader, num_epochs)

#dump the weights so we can use later if need to run test set and analyse failed predictions
torch.save(model.state_dict(), "convnext_model.pth")

#validate the model on the validation set
model.eval()
(actual_labels, predicted_probs) = validate_model(model, validation_loader)

#pickle dump for later
payload = (epochs, losses, test_accuracies, train_accuracies, actual_labels, predicted_probs)
with open('data.pickle', 'wb') as file:
    pickle.dump(payload, file)

# latency statistic
print(f"Average inference latency: {calculateLatency(model, validation_loader):.3f} ms")
