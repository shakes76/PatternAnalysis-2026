from dataset import HipMRI3D, HipMRI2D, get_training_test_split
from modules import CAN3D, CAN2D, compute_loss, calculate_descrete_dsc
import torch
import time
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


# Create Database
database = HipMRI2D(early_stop=5, images_path="../data/2dimages/", labels_path="../data/2dsemantic/")
train_loader = DataLoader(database, batch_size=1, shuffle=True)
test_loader = DataLoader(database, batch_size=1, shuffle=False)



# Select Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

# Train Models
model = CAN2D(256, 128, 6)
#model = UNet()
model = model.to(device)
state_dict = torch.load('final_parameters_2d.pth', weights_only=True, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Do visualisation of the first few things
print("Visualising...")

with torch.no_grad():
    images, labels = next(iter(train_loader))
    images = images.to(device)


    plt.subplot(2, 2, 1)
    plt.imshow(images.cpu().numpy()[0, :, :])

    plt.subplot(2, 2, 2)
    plt.imshow(labels.cpu().numpy()[0, :, :])

    output = model(images)
    predictions = torch.argmax(output, dim=1)

    plt.subplot(2, 2, 3)
    plt.imshow(output.cpu().numpy()[0, 1, :, :])

    plt.subplot(2, 2, 4)
    plt.imshow(predictions.cpu().numpy()[0, :, :])
plt.savefig("prediction_example_2d.png")

# Perform all tests.

print("Testing...")

exit()

number_of_tests = len(test_loader)
total_sum = torch.zeros((6), device=device)
total_validation_samples = 0

with torch.inference_mode():
    for i, (image, label) in enumerate(test_loader):
        print("{}/{} Test Batches".format(i+1, number_of_tests))
        image = image.to(device)
        label = label.to(device).long()

        predictions = model(image)

        sum_dice_per_class, no_in_batch = calculate_descrete_dsc(predictions, label)
        total_sum += sum_dice_per_class
        total_validation_samples += no_in_batch

total_sum = total_sum / total_validation_samples

print("Test Scores: ", total_sum)