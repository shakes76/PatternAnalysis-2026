import json

import matplotlib.pyplot as plt

is_3d = False

if is_3d:
    with open('3D_Model/unet3d_plot_data.json') as file:
        data = json.load(file)
    epochs = data['epoch']

    # loss vs epoch
    plt.figure()
    plt.plot(epochs, data['train_loss'], label='Train loss')
    plt.plot(epochs, data['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('3D_Model/unet3d_loss.png')

    # dice vs epoch
    plt.figure()
    for name, values in data['val_dice'].items():
        plt.plot(epochs, values, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation DSC')
    plt.legend()
    plt.savefig('3D_Model/unet3d_dice.png')
else:
    with open('2D_Unet_output/unet_2d_plot_data.json') as file:
        data = json.load(file)
    epochs = data['epoch']

    # loss vs epoch
    plt.figure()
    plt.plot(epochs, data['train_loss'], label='Train loss')
    plt.plot(epochs, data['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('2D_Unet_output/unet_2d_loss.png')

    # dice vs epoch
    plt.figure()
    for name, values in data['val_dice'].items():
        plt.plot(epochs, values, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation DSC')
    plt.legend()
    plt.savefig('2D_Unet_output/unet_2d_dice.png')

plt.show()
