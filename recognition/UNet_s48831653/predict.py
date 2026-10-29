import torch
import time

import modules
import dataset
import train

def predict(UNet=True):
    """
    Prepare the dataloaders, model, optimiser and scheduler. Evaluate's the
    model's predictions for the test dataset, saves the resulting images and
    reports on the model's performance.
    """

    # Set up
    _, _, test_loader = dataset.get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if UNet:
        model_path = "best_2DUnet.pth"
        model = modules.UNet_2D(in_channels=1, num_categories=5).to(device)
        image_directory = "UNet_predictions"
        image_name = "UNet"
        print("Evaluating 2D UNet predictions... (images stored under UNet_predictions)")
    else:
        model_path = "best_2DCAN.pth"
        model = modules.CAN_2D().to(device)
        image_name = "CAN"
        image_directory = "CAN_predictions"
        print("Evaluating 2D CAN predictions... (images stored under CAN_predictions)")
    model.load_state_dict(torch.load(model_path))

    # Begin timing training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Evaluate test case predictions and save corresponding images
    all_channel_dice, all_channel_iou = train.test(model, test_loader, device, image_name, image_directory)

    # Report testing duration, peak VRAM, and dice
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_gb = peak_vram_bytes / (1024.0 * 1024.0 * 1024.0)

    end_time = time.perf_counter()
    print(f"Average Testing Time = {(end_time - start_time)/len(test_loader.dataset)} per MRI slice")
    print(f"Peak VRAM during Testing = {peak_vram_gb:.4f} GB")
    print(f"    DICE VALUES...")
    print(f"        C0 Background/Rectum: {all_channel_dice[0].item()}")
    print(f"        C1  Body/Soft Tissue: {all_channel_dice[1].item()}")
    print(f"        C2              Bone: {all_channel_dice[2].item()}")
    print(f"        C3           Bladder: {all_channel_dice[3].item()}")
    print(f"        C4          Prostate: {all_channel_dice[4].item()}")
    print(f"    IOU VALUES...")
    print(f"        C0 Background/Rectum: {all_channel_iou[0].item()}")
    print(f"        C1  Body/Soft Tissue: {all_channel_iou[1].item()}")
    print(f"        C2              Bone: {all_channel_iou[2].item()}")
    print(f"        C3           Bladder: {all_channel_iou[3].item()}")
    print(f"        C4          Prostate: {all_channel_iou[4].item()}\n")
    return

if __name__ == "__main__":
    predict(UNet=True) # evaluate 2D U-Net predictions of test set
    predict(UNet=False) # evaluate 2D CAN predictions of test set