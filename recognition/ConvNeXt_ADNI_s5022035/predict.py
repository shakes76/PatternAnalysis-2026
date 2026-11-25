import os
import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from dataset import ADNIDataset, SPLIT_PROTOCOLS, checkpoint_normalization, get_adni_dataloaders, verify_checkpoint_split
from torchvision import transforms
from modules import (
    CONVNEXT_CONFIGS,
    BaselineCNN,
    ConvNeXtTiny,
)


def prediction_annotation(row):
    colour = "yellow" if row["decision"] == "REFER" else (
        "green" if row.get("correct") is True else "red" if row.get("correct") is False else "black"
    )
    title = f"pred {row['decision']} ({row['confidence']:.3f})"
    if "true_class" in row:
        title += f"\ntrue {row['true_class']}"
    return title, colour


def predict_images(model, paths, device, checkpoint, referral_threshold, output, cache_dir=None,
                   true_labels=None):
    if not math.isfinite(referral_threshold) or not 0.5 <= referral_threshold <= 1:
        raise ValueError("Referral threshold must be finite and in [0.5, 1]")
    if not paths:
        raise ValueError("At least one image is required")
    if true_labels is not None and (
        len(true_labels) != len(paths) or any(label not in {"NC", "AD"} for label in true_labels)
    ):
        raise ValueError("Supply one NC/AD true label for each image, in the same order")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new private output directory: {output}")
    mean, std = checkpoint_normalization(checkpoint)
    dataset = ADNIDataset(paths, [0] * len(paths), [""] * len(paths),
                          transforms.Compose([
                              transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize([mean], [std]),
                          ]), cache_dir=cache_dir)
    rows = []
    model.eval()
    output.mkdir(parents=True)
    with torch.no_grad():
        for index in range(len(dataset)):
            image, _ = dataset[index]
            probabilities = F.softmax(model(image.unsqueeze(0).to(device)), dim=1)[0]
            if probabilities.shape != (2,) or not torch.isfinite(probabilities).all():
                raise ValueError("Model must return finite NC/AD probabilities")
            probability = probabilities[1].item()
            prediction = "AD" if probability >= 0.5 else "NC"
            confidence = max(probability, 1 - probability)
            decision = "REFER" if confidence < referral_threshold else prediction
            row = {"image_index": index + 1, "probability_AD": probability,
                   "predicted_class": prediction, "confidence": confidence, "decision": decision}
            if true_labels is not None:
                row["true_class"] = true_labels[index]
                row["correct"] = prediction == true_labels[index]
            rows.append(row)
            fig, axis = plt.subplots(figsize=(4, 4))
            try:
                axis.imshow((image[0] * std + mean).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
                title, colour = prediction_annotation(row)
                axis.set_title(title, color=colour)
                axis.axis("off")
                fig.savefig(output / f"image_{index + 1:03d}.png", bbox_inches="tight")
            finally:
                plt.close(fig)
            print(f"Image {index + 1}: P(AD)={probability:.4f} -> {decision} "
                  f"(confidence {confidence:.4f})" + (
                      f" | true {row['true_class']} | correct={row['correct']}"
                      if true_labels is not None else ""
                  ))
    for start in range(0, len(rows), 12):
        page = rows[start:start + 12]
        columns = min(4, len(page))
        fig, axes = plt.subplots(math.ceil(len(page) / columns), columns,
                                 figsize=(4 * columns, 4 * math.ceil(len(page) / columns)),
                                 squeeze=False)
        try:
            for axis in axes.flat:
                axis.axis("off")
            for axis, row in zip(axes.flat, page):
                image, _ = dataset[row["image_index"] - 1]
                axis.imshow((image[0] * std + mean).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
                title, colour = prediction_annotation(row)
                axis.set_title(title, color=colour)
            fig.tight_layout()
            fig.savefig(output / f"prediction_examples_{start // 12 + 1:03d}.png")
        finally:
            plt.close(fig)
    report = {"temperature": 1.0, "referral_threshold": referral_threshold, "images": rows,
              "limitations": "Uncalibrated slice-level demo; user-supplied threshold is not a validated clinical rule"}
    with (output / "predictions.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def evaluate_and_autopsy(model, test_loader, device, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_preds = []
    all_labels = []
    failure_cases = {"false_positive": [], "false_negative": []}
    patient_ids = test_loader.dataset.patient_ids
    image_offset = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            for i in range(images.size(0)):
                prob_ad = probs[i][1].item()
                pred = preds[i].item()
                label = labels[i].item()

                all_preds.append(pred)
                all_labels.append(label)

                if pred != label:
                    error_type = "false_positive" if pred == 1 else "false_negative"
                    case_patient_id = patient_ids[image_offset + i]
                    if not any(
                        case["patient_id"] == case_patient_id
                        for case in failure_cases[error_type]
                    ):
                        failure_cases[error_type].append({
                            'patient_id': case_patient_id,
                            'image': images[i].cpu().squeeze().numpy(),
                            'true_label': 'AD' if label == 1 else 'NC',
                            'pred_label': 'AD' if pred == 1 else 'NC',
                            'confidence': prob_ad if pred == 1 else (1.0 - prob_ad),
                            'error_type': error_type,
                        })
                        failure_cases[error_type].sort(
                            key=lambda case: case["confidence"], reverse=True
                        )
                        failure_cases[error_type] = failure_cases[error_type][:3]
            image_offset += images.size(0)

    print(classification_report(all_labels, all_preds, target_names=['NC', 'AD'], digits=4))

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NC', 'AD'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("ADNI Test Set Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "test_confusion_matrix.png"))
    plt.close()

    selected_failure_cases = (
        failure_cases["false_positive"] + failure_cases["false_negative"]
    )
    if selected_failure_cases:
        fig, axes = plt.subplots(
            1, len(selected_failure_cases),
            figsize=(4 * len(selected_failure_cases), 3),
            squeeze=False,
        )

        for idx, case in enumerate(selected_failure_cases):
            axes[0][idx].imshow(case['image'], cmap='gray')
            axes[0][idx].set_title(
                f"{case['error_type'].replace('_', ' ').title()}\n"
                f"Patient: {case['patient_id']} | Conf: {case['confidence']:.2f}"
            )
            axes[0][idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "failure_case_autopsy.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Inference and Evaluation Script")
    parser.add_argument('--model', type=str, default='convnext', choices=['baseline', 'convnext'])
    parser.add_argument(
        '--convnext_variant',
        type=str,
        default='tiny',
        choices=sorted(CONVNEXT_CONFIGS),
        help='Custom ConvNeXt configuration matching the checkpoint',
    )
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='/home/groups/comp3710/ADNI/AD_NC')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_protocol', choices=SPLIT_PROTOCOLS, default='original')
    parser.add_argument('--metadata_path', default=None)
    parser.add_argument('--images', nargs='+', help='Private single-image demo; does not load dataset splits')
    parser.add_argument('--true_labels', nargs='+', choices=['NC', 'AD'],
                        help='Optional known diagnoses, one per --images entry in the same order')
    parser.add_argument('--referral_threshold', type=float, default=None,
                        help='Explicit uncalibrated confidence cutoff for --images')
    parser.add_argument('--cache_dir', default=None, help='Optional private decoded image cache')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory for test plots and failure-case images')
    args = parser.parse_args()
    if args.true_labels is not None and (
        not args.images or len(args.true_labels) != len(args.images)
    ):
        parser.error('--true_labels requires one label per --images entry')
    if args.images and args.referral_threshold is None:
        parser.error('--images requires an explicit --referral_threshold')
    if args.referral_threshold is not None and (
        not args.images or not math.isfinite(args.referral_threshold)
        or not 0.5 <= args.referral_threshold <= 1
    ):
        parser.error('--referral_threshold requires --images and a finite value in [0.5, 1]')

    if args.checkpoint is None:
        if args.model == 'baseline':
            args.checkpoint = './checkpoints/best_baseline_model.pt'
        else:
            filename = ("best_convnext_model.pt" if args.convnext_variant == 'tiny'
                        else f"best_convnext_{args.convnext_variant}_model.pt")
            args.checkpoint = os.path.join('./checkpoints', filename)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = Path(args.checkpoint).parent / "training_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["model"] != args.model or (
            args.model == "convnext" and config["convnext_variant"] != args.convnext_variant
        ):
            parser.error('Model/variant must match the checkpoint training_config.json')

    if args.model == 'baseline':
        model = BaselineCNN(in_channels=1, num_classes=2).to(device)
    else:
        model = ConvNeXtTiny(
            in_channels=1,
            num_classes=2,
            variant=args.convnext_variant,
        ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    if args.images:
        predict_images(model, args.images, device, args.checkpoint, args.referral_threshold,
                       args.results_dir, args.cache_dir, args.true_labels)
        return
    loaders = get_adni_dataloaders(
        base_dir=args.data_dir, batch_size=32, random_state=args.seed,
        split_protocol=args.split_protocol,
        metadata_path=args.metadata_path,
        normalization=checkpoint_normalization(args.checkpoint), cache_dir=args.cache_dir,
    )
    verify_checkpoint_split(args.checkpoint, loaders, args.data_dir, args.split_protocol, args.seed)
    _, _, test_loader = loaders
    if args.split_protocol != 'original':
        print("Subject/pooled protocol evaluation; do not use test results to tune the model.")
    evaluate_and_autopsy(model, test_loader, device, save_dir=args.results_dir)


if __name__ == '__main__':
    main()