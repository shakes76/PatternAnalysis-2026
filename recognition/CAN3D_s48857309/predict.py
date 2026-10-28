from dataset import HipMRI3D, get_training_test_split, load_single_hipmri_image
from modules import CAN3D, CAN2D, compute_loss, calculate_descrete_dsc, calculate_intersection_over_union
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

# Select Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

def predict_3d():
    # Create Database
    database = HipMRI3D(early_stop=5)
    train_set, val_set, test_set = get_training_test_split(database, test_size=0.2, val_size=0.1)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Load Models
    model = CAN3D()
    model = model.to(device)
    state_dict = torch.load('final_parameters_3d.pth', weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Do visualisation of the first few things
    print("Visualising...")

    with torch.no_grad():
        images, labels = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device).long()


        plt.subplot(1, 3, 1)
        plt.imshow(images.cpu().numpy()[0, :, 128, :])

        plt.subplot(1, 3, 2)
        plt.imshow(labels.cpu().numpy()[0, :, 128, :])

        output = model(images)
        predictions = torch.argmax(output, dim=1)
        plt.subplot(1, 3, 3)
        plt.imshow(predictions.cpu().numpy()[0, :, 128, :])

    plt.savefig("prediction_example_3d.png")

    # Perform all tests.

    print("Testing...")

    number_of_tests = len(test_loader)
    total_sum_dice = torch.zeros((6), device=device)
    total_sum_IoU = torch.zeros((6), device=device)
    total_validation_samples = 0

    with torch.inference_mode():
        for i, (image, label) in enumerate(test_loader):
            print("{}/{} Test Batches".format(i+1, number_of_tests))
            image = image.to(device)
            label = label.to(device).long()

            output = model(image)

            sum_dice_per_class, no_in_batch = calculate_descrete_dsc(output, label)
            sum_IoU_per_class, _ = calculate_intersection_over_union(output, label)
            total_sum_dice += sum_dice_per_class
            total_sum_IoU += sum_IoU_per_class
            total_validation_samples += no_in_batch

    total_sum_dice = total_sum_dice / total_validation_samples
    total_sum_IoU = total_sum_IoU / total_validation_samples

    print("Test Scores: Dice: ", total_sum_dice, "IoU: ", total_sum_IoU)

def volume_to_2d_batch(volume : torch.Tensor):
    # take height-wise slices
    return volume.permute(1, 0, 2)

def batch_output_to_3d_volume_batch(batch : torch.Tensor):
    return batch.permute(1, 2, 0, 3)[None,:,:,:,:]

def predict_2d():
    # Create Database
    database = HipMRI3D(early_stop=5)
    train_set, val_set, test_set = get_training_test_split(database, test_size=0.2, val_size=0.1)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Load Models
    model = CAN2D()
    model = model.to(device)
    state_dict = torch.load('final_parameters_2d.pth', weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Do visualisation of the first few things
    print("Visualising...")

    with torch.no_grad():
        images, labels = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device).long()

        plt.subplot(1, 3, 1)
        plt.imshow(images.cpu().numpy()[0, :, 128, :])

        plt.subplot(1, 3, 2)
        plt.imshow(labels.cpu().numpy()[0, :, 128, :])

        output = batch_output_to_3d_volume_batch(model(volume_to_2d_batch(images[0])))
        predictions = torch.argmax(output, dim=1)
        plt.subplot(1, 3, 3)
        plt.imshow(predictions.cpu().numpy()[0, :, 128, :])

    plt.savefig("prediction_example_2d.png")

    # Perform all tests.

    print("Testing...")

    number_of_tests = len(test_loader)
    total_sum_dice = torch.zeros((6), device=device)
    total_sum_IoU = torch.zeros((6), device=device)
    total_validation_samples = 0

    with torch.inference_mode():
        for i, (image, label) in enumerate(test_loader):
            print("{}/{} Test Batches".format(i+1, number_of_tests))
            image = image.to(device)
            label = label.to(device).long()

            output = batch_output_to_3d_volume_batch(model(volume_to_2d_batch(images[0])))

            sum_dice_per_class, no_in_batch = calculate_descrete_dsc(output, label)
            sum_IoU_per_class, _ = calculate_intersection_over_union(output, label)
            total_sum_dice += sum_dice_per_class
            total_sum_IoU += sum_IoU_per_class
            total_validation_samples += no_in_batch

    total_sum_dice = total_sum_dice / total_validation_samples
    total_sum_IoU = total_sum_IoU / total_validation_samples

    print("Test Scores: Dice: ", total_sum_dice, "IoU: ", total_sum_IoU)


def make_figures(sample = 'J005_Week2', figure_type='differences3d'):
    images, labels = load_single_hipmri_image(sample, images_path='../data/images', labels_path='../data/semantic')

    print("Loading Models")
    # Load Models
    model3d = CAN3D()
    model3d = model3d.to(device)
    state_dict = torch.load('final_parameters_3d.pth', weights_only=True, map_location=device)
    model3d.load_state_dict(state_dict)
    model3d.eval()

    model2d = CAN2D()
    model2d = model2d.to(device)
    state_dict = torch.load('final_parameters_2d.pth', weights_only=True, map_location=device)
    model2d.load_state_dict(state_dict)
    model2d.eval()


    print("Visualising...")
    if figure_type == 'differences3d':
        negative_cmap = plt.get_cmap('Blues').copy()
        positive_cmap = plt.get_cmap('Reds').copy()

        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device).long()

            output = model3d(images)
            predictions = torch.argmax(output, dim=1)
            for i in range(0, 8):
                height = 112 + i * 4
                plt.subplot(1, 3, 1)
                plt.imshow(images.cpu().numpy()[0, :, height, :])

                plt.subplot(1, 3, 2)
                plt.imshow(labels.cpu().numpy()[0, :, height, :], vmin =0, vmax=5)


                plt.subplot(1, 3, 3)
                plt.imshow(predictions.cpu().numpy()[0, :, height, :], vmin =0, vmax=5)
                false_positive = (predictions.cpu().numpy()[0, :, height, :] == 5) * (labels.cpu().numpy()[0, :, height, :] != 5)
                masked = np.ma.masked_where(false_positive <= 0, false_positive)
                plt.imshow(masked, cmap=positive_cmap, alpha=1.0, vmin=0.1, vmax=1.5)
                false_negative = (predictions.cpu().numpy()[0, :, height, :] != 5) * (labels.cpu().numpy()[0, :, height, :] == 5)
                masked = np.ma.masked_where(false_negative <= 0, false_negative)
                plt.imshow(masked, cmap=negative_cmap, alpha=1.0, vmin=0.1, vmax=1.5)

                plt.title("MRI slice (height = {})".format(height))
                plt.show()

    if figure_type == '2dvs3d':
        negative_cmap = plt.get_cmap('Blues').copy()
        positive_cmap = plt.get_cmap('Reds').copy()

        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device).long()
            output3d = model3d(images)
            predictions3d = torch.argmax(output3d, dim=1)

            output2d = batch_output_to_3d_volume_batch(model2d(volume_to_2d_batch(images[0])))
            predictions2d = torch.argmax(output2d, dim=1)

            for i in range(0, 12):
                height = 100 + i * 4
                plt.subplot(2, 2, 1)
                plt.imshow(images.cpu().numpy()[0, :, height, :])
                plt.title("Input")

                plt.subplot(2, 2, 2)
                plt.imshow(labels.cpu().numpy()[0, :, height, :], vmin =0, vmax=5)
                plt.title("True Labels")

                
                plt.subplot(2, 2, 3)
                plt.imshow(predictions3d.cpu().numpy()[0, :, height, :], vmin =0, vmax=5)
                false_positive = (predictions3d.cpu().numpy()[0, :, height, :] == 5) * (labels.cpu().numpy()[0, :, height, :] != 5)
                masked = np.ma.masked_where(false_positive <= 0, false_positive)
                plt.imshow(masked, cmap=positive_cmap, alpha=1.0, vmin=0.1, vmax=1.5)
                false_negative = (predictions3d.cpu().numpy()[0, :, height, :] != 5) * (labels.cpu().numpy()[0, :, height, :] == 5)
                masked = np.ma.masked_where(false_negative <= 0, false_negative)
                plt.imshow(masked, cmap=negative_cmap, alpha=1.0, vmin=0.1, vmax=1.5)
                plt.title("CAN3D prediction")

                
                plt.subplot(2, 2, 4)
                plt.imshow(predictions2d.cpu().numpy()[0, :, height, :], vmin =0, vmax=5)
                false_positive = (predictions2d.cpu().numpy()[0, :, height, :] == 5) * (labels.cpu().numpy()[0, :, height, :] != 5)
                masked = np.ma.masked_where(false_positive <= 0, false_positive)
                plt.imshow(masked, cmap=positive_cmap, alpha=1.0, vmin=0.1, vmax=1.5)
                false_negative = (predictions2d.cpu().numpy()[0, :, height, :] != 5) * (labels.cpu().numpy()[0, :, height, :] == 5)
                masked = np.ma.masked_where(false_negative <= 0, false_negative)
                plt.imshow(masked, cmap=negative_cmap, alpha=1.0, vmin=0.1, vmax=1.5)
                plt.title("CAN2D prediction")
                plt.show()

def perform_benchmark(which = '3d'):
    images, labels = load_single_hipmri_image('R039_Week4', images_path='../data/images', labels_path='../data/semantic')
    images = images.to(device)

    print("Loading Models")
    if which == '3d':
        print("Testing 3D")
        model3d = CAN3D()
        model3d = model3d.to(device)
        state_dict = torch.load('final_parameters_3d.pth', weights_only=True, map_location=device)
        model3d.load_state_dict(state_dict)
        model3d.eval()

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device=device)

        start = time.time()
        output = model3d(images).cpu()
        end = time.time()
        total_params = sum(p.numel() for p in model3d.parameters())

        peak_memory_usage = 'N/A'
        if device.type == 'cuda':
            peak_memory_usage = torch.cuda.max_memory_allocated(device=device)

        print("Inference time: ", end - start
              , "Peak Usage: ", peak_memory_usage
              , "Parameter Count: ", total_params)

    else:
        print("Testing 2D")
        model2d = CAN2D()
        model2d = model2d.to(device)
        state_dict = torch.load('final_parameters_2d.pth', weights_only=True, map_location=device)
        model2d.load_state_dict(state_dict)
        model2d.eval()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device=device)

        start = time.time()
        output = batch_output_to_3d_volume_batch(model2d(volume_to_2d_batch(images[0]))).cpu()
        end = time.time()
        total_params = sum(p.numel() for p in model2d.parameters())

        peak_memory_usage = 'N/A'
        if device.type == 'cuda':
            peak_memory_usage = torch.cuda.max_memory_allocated(device=device)

        print("Inference time: ", end - start
              , "Peak Usage: ", peak_memory_usage
              , "Parameter Count: ", total_params)


#predict_2d()
#make_figures()
#perform_benchmark()
predict_3d()