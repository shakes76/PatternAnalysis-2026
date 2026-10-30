import os
import sys
import shutil
import argparse
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules import GFNet, ResNet18, get_device, load_checkpoint
from dataset import ADNIDataset, data_preprocess, _load_test_samples, DATA_TEST_PATH, BATCH_SIZE
from train import PATH_BEST_MODEL


def get_best_model_checkpoint(model_name: str) -> str:
    """
    Resolves the best model checkpoint path imported from train.py
    based on the chosen model architecture.
    """
    return PATH_BEST_MODEL + "_" + str(model_name) + ".pt"


def load_model(model_name: str, checkpoint_path: str, device: torch.device):
    """
    Initializes the chosen model and loads the best checkpoint weights.
    """
    model_name_lower = model_name.lower()
    if model_name_lower == "gfnet":
        model = GFNet()
    elif model_name_lower == "rn18":
        model = ResNet18()
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose 'gfnet' or 'resnet18'.")

    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint is not None:
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded best checkpoint for {model_name} from: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at '{checkpoint_path}'. Initializing model with default weights.")

    model = model.to(device)
    model.eval()
    return model


def run_inference(model, test_loader, test_samples, device):
    """
    Runs model inference on the test dataset and records correctness and confidence scores.
    """
    results = []
    sample_idx = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            confidences, preds = torch.max(probabilities, dim=1)

            for i in range(images.size(0)):
                img_path, true_label = test_samples[sample_idx]
                pred_label = preds[i].item()
                conf = confidences[i].item()
                correctness = "Correct" if pred_label == true_label else "Wrong"

                results.append({
                    "sample_idx": sample_idx,
                    "img_path": img_path,
                    "image_tensor": images[i].cpu(),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": conf,
                    "correctness": correctness
                })
                sample_idx += 1

    return results


def select_samples_by_confidence(samples, num_samples=5):
    """
    Sorts samples by confidence score and partitions them into High, Medium, and Low tiers.
    """
    # Sort samples by confidence descending
    sorted_samples = sorted(samples, key=lambda x: x["confidence"], reverse=True)
    n = len(sorted_samples)

    if n == 0:
        return {"High": [], "Medium": [], "Low": []}

    # High confidence: top samples
    high_samples = sorted_samples[:min(num_samples, n)]

    # Low confidence: bottom samples
    low_samples = sorted_samples[max(0, n - num_samples):]

    # Medium confidence: centered around the median
    mid_start = max(0, (n // 2) - (num_samples // 2))
    med_samples = sorted_samples[mid_start:min(n, mid_start + num_samples)]

    return {
        "High": high_samples,
        "Medium": med_samples,
        "Low": low_samples
    }


def save_selected_images(categorized_samples, output_dir):
    """
    Saves selected images into Results/CORRECTNESS_CONFIDENCE/ directories.
    Checks and creates each directory if required.
    """
    for correctness, conf_dict in categorized_samples.items():
        for confidence, samples in conf_dict.items():
            dir_name = f"{correctness}_{confidence}"
            target_dir = os.path.join(output_dir, dir_name)

            # Check and create directory if required
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)

            for item in samples:
                src_path = item["img_path"]
                idx = item["sample_idx"]
                conf = item["confidence"]
                t_lbl = item["true_label"]
                p_lbl = item["pred_label"]

                base_name = os.path.basename(src_path)
                filename = f"sample_{idx}_true{t_lbl}_pred{p_lbl}_conf{conf:.4f}_{base_name}"
                dest_path = os.path.join(target_dir, filename)

                if os.path.exists(src_path):
                    shutil.copy2(src_path, dest_path)
                else:
                    # Fallback to saving unnormalized tensor if source image file is unavailable
                    img_tensor = item["image_tensor"] * 0.5 + 0.5
                    vutils.save_image(img_tensor.clamp(0, 1), dest_path)

            print(f"Saved {len(samples)} images to: {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Inference script for chosen ADNI classification model.")
    parser.add_argument("--model", default="GFNET", choices=["RN18", "GFNET"], help="Model to train: 'RN18' or 'GFNET'")
    parser.add_argument("--test_dir", type=str, default=DATA_TEST_PATH,
                        help="Path to the test dataset.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for test DataLoader.")
    parser.add_argument("--output_dir", type=str, default="Results",
                        help="Root output directory for categorized images.")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Selected model: {args.model}")

    # Resolve best checkpoint path imported from train.py
    checkpoint_path = get_best_model_checkpoint(args.model)
    model = load_model(args.model, checkpoint_path, device)

    # Prepare test DataLoader
    _, eval_transform = data_preprocess()
    test_samples = _load_test_samples(test_dir=args.test_dir)
    test_dataset = ADNIDataset(test_samples, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    print(f"Total test samples: {len(test_samples)}")

    # Run inference
    inference_results = run_inference(model, test_loader, test_samples, device)

    # Separate into Correct and Wrong
    correct_samples = [r for r in inference_results if r["correctness"] == "Correct"]
    wrong_samples = [r for r in inference_results if r["correctness"] == "Wrong"]

    print(f"Correct predictions: {len(correct_samples)}, Wrong predictions: {len(wrong_samples)}")

    # Select 5 images per confidence tier (High, Medium, Low)
    categorized = {
        "Correct": select_samples_by_confidence(correct_samples, num_samples=5),
        "Wrong": select_samples_by_confidence(wrong_samples, num_samples=5)
    }

    output_path = args.output_dir + "/" + args.model

    # Save images into Results/CORRECTNESS_CONFIDENCE/
    save_selected_images(categorized, output_path)
    print("Inference and image categorization completed.")


if __name__ == "__main__":
    main()
