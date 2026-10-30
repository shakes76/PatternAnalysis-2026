import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from modules import UNet3d
from dataset import HipMRIDataset
import time

def main():
    data_dir = Path("/home/groups/comp3710/HipMRI_Study_open")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    #load the test dataset
    test_dataset = HipMRIDataset(data_dir,split='test')

    #create a data loader
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

    #initialise the same architecture used during training
    model = UNet3d(in_channels=1, num_classes=6).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    aux_params = sum(p.numel() for p in model.aux2.parameters())
    aux_params += sum(p.numel() for p in model.aux3.parameters())

    print(f"Intermediate parameters: {total_params - aux_params:,}")
    print(f"Advanced parameters: {total_params:,}")
    print(f"Additional parameters: {aux_params:,}")

    checkpoint = torch.load("best_DeepSupervision_class5.pth",map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint,strict=True)
    model.eval()
    inference_times = []
    num_classes = 6
    intersections = torch.zeros(num_classes,dtype=torch.float64)
    pred_counts = torch.zeros(num_classes,dtype=torch.float64)
    true_counts = torch.zeros(num_classes,dtype=torch.float64)
    case_scores = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with (torch.no_grad()):
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            inference_times.append(time.perf_counter() - start)

            predictions = torch.argmax(outputs, dim=1)
            pred5 = (predictions == 5)
            true5 = (masks == 5)

            intersection5 = (pred5 & true5).sum().item()
            total5 = pred5.sum().item() + true5.sum().item()

            dice5 = (2 * intersection5 / total5) if total5 > 0 else 1.0

            case_scores.append((batch_idx, dice5))

            if batch_idx in  [12,13,17]:
                mri = images[0,0].cpu().numpy()
                true_mask = masks[0].cpu().numpy()
                pred_mask = predictions[0].cpu().numpy()
                false_positive = (pred_mask == 5) & (true_mask != 5)
                false_negative = (true_mask == 5) & (pred_mask != 5)

                z = (true_mask == 5).sum(axis=(0,1)).argmax()

                mri = mri[:,:,z]
                true_mask = true_mask[:,:,z]
                pred_mask = pred_mask[:,:,z]
                false_positive = false_positive[:, :, z]
                false_negative = false_negative[:, :, z]

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))

                axes[0].imshow(mri, cmap="gray")
                axes[0].set_title("Original MRI")

                axes[1].imshow(true_mask, cmap="tab10", vmin=0, vmax=5,interpolation="nearest")
                axes[1].set_title("Ground Truth")

                axes[2].imshow(pred_mask,cmap="tab10",vmin=0,vmax=5,interpolation="nearest")
                axes[2].set_title("U-Net Prediction")

                axes[3].imshow(mri, cmap="gray")
                axes[3].imshow(false_positive, cmap="Reds", alpha=0.6)
                axes[3].imshow(false_negative, cmap="Blues", alpha=0.6)
                axes[3].set_title("Red: FP | Blue: FN")

                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()
                plt.savefig(f"failure_case_{batch_idx}.png", dpi=150)
                plt.close()

            for class_id in range(num_classes):
                pred_mask = (predictions == class_id)
                true_mask = (masks == class_id)

                intersection = torch.sum(pred_mask * true_mask)
                pred_total = torch.sum(pred_mask)
                target = torch.sum(true_mask)

                intersections[class_id] += intersection.item()
                pred_counts[class_id] += pred_total.item()
                true_counts[class_id] += target.item()

    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        print(f"Peak GPU VRAM: {peak_vram:.3f} GB")
    avg_time = sum(inference_times) / len(inference_times)

    print(f"Average inference time: {avg_time * 1000:.2f} ms/volume")

    case_scores.sort(key=lambda x: x[1])

    print("\nWorst 3 Class 5 segmentation cases:")

    for case_idx, score in case_scores[:3]:
        print(f"Volume {case_idx}: Dice = {score:.4f}")

    dice = (2 * intersections + 1e-6)/(pred_counts + true_counts + 1e-6)
    union = pred_counts + true_counts - intersections
    iou = (intersections + 1e-6) / (union + 1e-6)

    for class_id, score in enumerate(dice):
        print(f"Class {class_id}: Dice = {score.item():.4f},Iou = {iou[class_id]:.4f}")

    print("Test images:", len(test_dataset))
    print("Test batches:", len(test_loader))

if __name__ == "__main__":
    main()