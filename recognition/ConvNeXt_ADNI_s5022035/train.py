import os
import argparse
import json
import hashlib
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt

from dataset import (
    get_adni_dataloaders,
    position_balanced_sample_weights,
    seed_data_worker,
    SPLIT_PROTOCOLS,
    split_manifest,
)
from utils import era_balanced_sample_weights
from modules import (
    CONVNEXT_CONFIGS,
    BaselineCNN,
    ConvNeXtTiny,
)


def initialize_imagenet_trunk(model, source):
    target = model.state_dict()
    mapping = {}
    for suffix in ("weight", "bias"):
        mapping[f"downsample_layers.0.0.{suffix}"] = f"features.0.0.{suffix}"
        mapping[f"downsample_layers.0.1.norm.{suffix}"] = f"features.0.1.{suffix}"
        mapping[f"norm.{suffix}"] = f"classifier.0.{suffix}"
    for stage_index, depth in enumerate((3, 3, 9, 3)):
        for block_index in range(depth):
            prefix = f"stages.{stage_index}.{block_index}"
            original = f"features.{2 * stage_index + 1}.{block_index}"
            mapping[f"{prefix}.gamma"] = f"{original}.layer_scale"
            for name, index in (("dwconv", 0), ("norm", 2), ("pwconv1", 3), ("pwconv2", 5)):
                for suffix in ("weight", "bias"):
                    mapping[f"{prefix}.{name}.{suffix}"] = f"{original}.block.{index}.{suffix}"
        if stage_index:
            for suffix in ("weight", "bias"):
                mapping[f"downsample_layers.{stage_index}.0.norm.{suffix}"] = (
                    f"features.{2 * stage_index}.0.{suffix}"
                )
                mapping[f"downsample_layers.{stage_index}.1.{suffix}"] = (
                    f"features.{2 * stage_index}.1.{suffix}"
                )
    if set(mapping) != {name for name in target if not name.startswith("head.")}:
        raise ValueError("ImageNet initialization requires the complete paper_tiny trunk")
    if set(source) != set(mapping.values()) | {"classifier.2.weight", "classifier.2.bias"}:
        raise ValueError("ImageNet checkpoint contains an unexpected set of weights")
    converted = {}
    for name, original in mapping.items():
        value = source[original]
        if name == "downsample_layers.0.0.weight":
            if tuple(value.shape) != (96, 3, 4, 4):
                raise ValueError("Expected a three-channel ImageNet stem")
            # Summation preserves the response to an identical grayscale input in all RGB channels.
            value = value.sum(dim=1, keepdim=True)
        elif name.endswith(".gamma"):
            if tuple(value.shape) != (target[name].numel(), 1, 1):
                raise ValueError(f"Unexpected ImageNet layer-scale shape: {original}")
            value = value.reshape(-1)
        if value.shape != target[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"ImageNet tensor shape or values are invalid: {original}")
        converted[name] = value
    model.load_state_dict({**target, **converted}, strict=True)
    return len(converted)


def load_imagenet_initialization(model, save_dir):
    from torchvision.models import ConvNeXt_Tiny_Weights
    from urllib.parse import urlparse

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    source = weights.get_state_dict(progress=True, check_hash=True)
    cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    digest = hashlib.sha256(cached.read_bytes()).hexdigest()
    if not digest.startswith(cached.stem.rsplit("-", 1)[1]):
        raise ValueError("Cached ImageNet weights do not match the published hash prefix")
    count = initialize_imagenet_trunk(model, source)
    report = {
        "weights": "torchvision.ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "url": weights.url,
        "weights_file_sha256": digest,
        "loaded_trunk_tensors": count,
        "grayscale_stem": "Sum the three RGB input-channel kernels",
        "classifier": "Original randomly initialized two-class head retained",
        "preprocessing": "Project fixed 0.5/0.5 grayscale scaling; not ImageNet RGB preprocessing",
        "fine_tuning": "All parameters trainable",
    }
    (Path(save_dir) / "pretrained_initialization.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("ImageNet initialization: " + json.dumps(report, allow_nan=False), flush=True)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        augmentation = getattr(dataloader.dataset, "device_augmentation", None)
        if augmentation is not None:
            images = augmentation(images)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("Cannot evaluate an empty data loader")
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, targets):
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        target_probabilities = torch.exp(-cross_entropy)
        focal_factor = (1 - target_probabilities).pow(self.gamma)
        class_weights = self.alpha[targets]
        return (class_weights * focal_factor * cross_entropy).mean()


def plot_metrics(history, save_path):
    """
    Plots and saves training & validation loss and accuracy curves.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_title('Training vs Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patient_balanced_sample_weights(dataset):
    patient_labels = {}
    patient_counts = {}
    for patient_id, label in zip(dataset.patient_ids, dataset.labels):
        if patient_id in patient_labels and patient_labels[patient_id] != label:
            raise ValueError(f"Patient {patient_id!r} has conflicting training labels")
        patient_labels[patient_id] = label
        patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1

    patients_per_class = {}
    for patient_id, label in patient_labels.items():
        patients_per_class.setdefault(label, set()).add(patient_id)

    weights = []
    for patient_id, label in zip(dataset.patient_ids, dataset.labels):
        class_patient_count = len(patients_per_class[label])
        weights.append(
            1.0 / (
                len(patients_per_class)
                * class_patient_count
                * patient_counts[patient_id]
            )
        )
    return torch.as_tensor(weights, dtype=torch.double)


def class_balanced_patient_weights(dataset, beta, ad_weight):
    patient_labels = {}
    for patient_id, label in zip(dataset.patient_ids, dataset.labels):
        if patient_id in patient_labels and patient_labels[patient_id] != label:
            raise ValueError(f"Patient {patient_id!r} has conflicting training labels")
        patient_labels[patient_id] = label

    patient_counts = torch.bincount(
        torch.tensor(list(patient_labels.values()), dtype=torch.long),
        minlength=2,
    ).float()
    if torch.any(patient_counts == 0):
        raise ValueError("Both classes must be represented in the training split")
    effective_counts = 1 - torch.pow(beta, patient_counts)
    weights = (1 - beta) / effective_counts
    weights = weights / weights.mean()
    weights[1] *= ad_weight
    return weights / weights.mean()


def save_validation_gradcam(model, dataloader, device, save_path):
    model.eval()
    examples = {}
    categories = {
        (0, 0): "Correct NC",
        (1, 1): "Correct AD",
        (0, 1): "False positive",
        (1, 0): "False negative",
    }

    with torch.no_grad():
        for images, labels in dataloader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for image, label, prediction in zip(images, labels, predictions):
                key = (int(label), int(prediction))
                if key not in examples:
                    examples[key] = image.clone()
            if len(examples) == len(categories):
                break

    if not examples:
        return

    activation = {}

    def capture_activation(_module, _inputs, output):
        output.retain_grad()
        activation["value"] = output

    hook = model.stages[-1].register_forward_hook(capture_activation)
    try:
        fig, axes = plt.subplots(
            1, len(examples), figsize=(4 * len(examples), 4), squeeze=False
        )
        for axis, (key, image) in zip(axes[0], examples.items()):
            model.zero_grad(set_to_none=True)
            input_image = image.unsqueeze(0).to(device)
            logits = model(input_image)
            target_class = key[1]
            logits[0, target_class].backward()

            feature_map = activation["value"]
            gradients = feature_map.grad
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * feature_map).sum(dim=1, keepdim=True).relu()
            cam = F.interpolate(
                cam, size=input_image.shape[-2:], mode="bilinear", align_corners=False
            )[0, 0]
            cam = cam / cam.max().clamp_min(1e-8)

            display_image = (image.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
            axis.imshow(display_image, cmap="gray")
            axis.imshow(cam.detach().cpu(), cmap="jet", alpha=0.4)
            axis.set_title(categories[key])
            axis.axis("off")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
    finally:
        hook.remove()


def main():
    parser = argparse.ArgumentParser(description="Train Baseline or ConvNeXt model on ADNI dataset")
    parser.add_argument('--model', type=str, default='convnext', choices=['baseline', 'convnext'], help='Model architecture')
    parser.add_argument(
        '--initialization', choices=['scratch', 'imagenet'], default='scratch',
        help='Optional ImageNet trunk initialization for the validation-only paper_tiny model',
    )
    parser.add_argument(
        '--convnext_variant',
        type=str,
        default='tiny',
        choices=sorted(CONVNEXT_CONFIGS),
        help='Custom ConvNeXt configuration to use for scratch training',
    )
    parser.add_argument('--data_dir', type=str, default='/home/groups/comp3710/ADNI/AD_NC', help='ADNI dataset directory')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='AdamW weight decay for regularization')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                        help='Cross-entropy label smoothing in [0, 1)')
    parser.add_argument(
        '--loss',
        choices=['cross_entropy', 'focal'],
        default='cross_entropy',
        help='Training loss; focal loss uses effective-number patient weights',
    )
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--class_balance_beta', type=float, default=0.99)
    parser.add_argument('--ad_loss_weight', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=6,
                        help='Stop after this many epochs without validation-accuracy improvement; 0 disables')
    parser.add_argument(
        '--save_best_val_loss', action='store_true',
        help='Also save the lowest-validation-cross-entropy checkpoint; early stopping stays accuracy-based',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--split_protocol', choices=SPLIT_PROTOCOLS, default='original',
                        help='Pooled protocol is a separate exploratory experiment, not the original benchmark')
    parser.add_argument('--metadata_path', default=None,
                        help='ADNI metadata required for subject_70_15_15')
    parser.add_argument('--normalization', choices=['fixed', 'training'], default='fixed',
                        help='Use legacy 0.5/0.5 scaling or resized training-only pixel statistics')
    parser.add_argument('--num_workers', type=int, default=1, help='DataLoader worker count')
    parser.add_argument('--cache_dir', default=None, help='Optional private decoded uint8 image cache')
    parser.add_argument('--gpu_augmentation', action='store_true',
                        help='Apply training affine/jitter transforms on CUDA tensors')
    parser.add_argument('--drop_path_rate', type=float, default=0.0,
                        help='Maximum stochastic-depth rate for scratch ConvNeXt')
    parser.add_argument('--head_dropout', type=float, default=0.0,
                        help='Classifier dropout for scratch ConvNeXt')
    parser.add_argument('--layer_scale_init', type=float, default=1e-6,
                        help='Initial residual layer scale for scratch ConvNeXt')
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        '--patient_balanced_sampling',
        action='store_true',
        help='Sample classes equally and patients equally within each class',
    )
    sampling.add_argument(
        '--position_balanced_sampling',
        action='store_true',
        help='Balance NC/AD within training-derived position bins; preserve bin mass',
    )
    sampling.add_argument(
        '--era_balanced_sampling',
        action='store_true',
        help='Keep NC/AD draw mass equal and split AD draws equally by acquisition era',
    )
    parser.add_argument('--position_bins', type=int, default=5,
                        help='Training quantile bins for position balancing')
    parser.add_argument(
        '--augmentation',
        choices=['standard', 'standard_flip', 'mild'],
        default='standard',
        help='Training-only profile; standard_flip adds experimental 50%% horizontal flips',
    )
    parser.add_argument(
        '--save_validation_gradcam',
        action='store_true',
        help='Save Grad-CAM examples from validation data',
    )
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--skip_test', action='store_true',
                        help='Skip test-set evaluation while comparing training experiments')
    parser.set_defaults(skip_test=True)
    parser.add_argument(
        '--evaluate_test',
        dest='skip_test',
        action='store_false',
        help='Explicitly evaluate the test set after training (prefer predict.py for final reporting)',
    )
    args = parser.parse_args()

    if args.save_best_val_loss and (
        args.split_protocol != 'subject_70_15_15' or not args.skip_test
    ):
        parser.error('--save_best_val_loss requires subject_70_15_15 and --skip_test')
    if args.initialization == "imagenet" and (
        args.model != "convnext" or args.convnext_variant != "paper_tiny"
        or args.split_protocol != "subject_70_15_15" or not args.skip_test
        or args.normalization != "fixed"
    ):
        parser.error('ImageNet initialization requires paper_tiny, subject_70_15_15, fixed normalization and --skip_test')
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error('--label_smoothing must be in [0, 1)')
    if args.patience < 0:
        parser.error('--patience must be non-negative')
    if not 0.0 <= args.drop_path_rate < 1.0:
        parser.error('--drop_path_rate must be in [0, 1)')
    if not 0.0 <= args.head_dropout < 1.0:
        parser.error('--head_dropout must be in [0, 1)')
    if args.layer_scale_init < 0.0:
        parser.error('--layer_scale_init must be non-negative')
    if args.focal_gamma < 0.0:
        parser.error('--focal_gamma must be non-negative')
    if not 0.0 <= args.class_balance_beta < 1.0:
        parser.error('--class_balance_beta must be in [0, 1)')
    if args.ad_loss_weight <= 0.0:
        parser.error('--ad_loss_weight must be positive')
    if args.position_bins < 1:
        parser.error('--position_bins must be positive')
    if args.loss == 'focal' and args.label_smoothing != 0.0:
        parser.error('--label_smoothing must be 0 when using focal loss')
    if args.loss == 'focal' and (
        args.patient_balanced_sampling
        or args.position_balanced_sampling
        or args.era_balanced_sampling
    ):
        parser.error(
            'Choose focal loss or balanced sampling, not both, '
            'to avoid applying class balancing twice'
        )
    if args.era_balanced_sampling and args.split_protocol != 'subject_70_15_15':
        parser.error('--era_balanced_sampling requires --split_protocol subject_70_15_15')
    if args.era_balanced_sampling and not args.metadata_path:
        parser.error('--era_balanced_sampling requires --metadata_path')
    if args.save_validation_gradcam and args.model != 'convnext':
        parser.error('--save_validation_gradcam requires --model convnext')
    if args.normalization == 'training' and (
        args.split_protocol != 'subject_70_15_15' or not args.skip_test
    ):
        parser.error('Training-derived normalization requires subject_70_15_15 and --skip_test')

    separate_run = args.split_protocol != 'original' or (
        args.model == 'convnext' and args.convnext_variant == 'paper_tiny'
    )
    if separate_run:
        if args.save_dir == './checkpoints' or (
            os.path.exists(args.save_dir) and os.listdir(args.save_dir)
        ):
            parser.error('Separate protocol or paper_tiny experiments require an empty --save_dir')
    if args.split_protocol == 'subject_70_15_15':
        print("Subject-disjoint validation experiment; the held-out test split is not evaluated.")
    elif args.split_protocol != 'original':
        print(f"EXPLORATORY {args.split_protocol} split; original benchmark test images may enter training.")
        print("This does not restore an untouched test set or replace original benchmark results.")
    if args.split_protocol != 'subject_70_15_15':
        print("WARNING: legacy scan-prefix grouping does not enforce unique-subject separation.")
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.gpu_augmentation and device.type != 'cuda':
        parser.error('--gpu_augmentation requires CUDA')

    train_loader, val_loader, test_loader = get_adni_dataloaders(
        base_dir=args.data_dir,
        batch_size=args.batch_size,
        random_state=args.seed,
        num_workers=args.num_workers,
        augmentation=args.augmentation,
        split_protocol=args.split_protocol,
        metadata_path=args.metadata_path,
        normalization='training' if args.normalization == 'training' else (0.5, 0.5),
        cache_dir=args.cache_dir,
        gpu_augmentation=args.gpu_augmentation,
    )
    if separate_run:
        manifest = split_manifest(
            (train_loader, val_loader, test_loader), args.data_dir, args.split_protocol, args.seed
        )
        with open(os.path.join(args.save_dir, "split_manifest.json"), "w") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
        if args.normalization == 'training':
            manifest_path = Path(args.save_dir) / "split_manifest.json"
            statistics = {
                **train_loader.dataset.channel_statistics,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            }
            (Path(args.save_dir) / "normalization.json").write_text(
                json.dumps(statistics, indent=2, allow_nan=False), encoding="utf-8"
            )
            print("Training-only normalization: " + json.dumps(statistics, allow_nan=False))
        with open(os.path.join(args.save_dir, "training_config.json"), "w") as stream:
            json.dump(vars(args), stream, indent=2, allow_nan=False)
        for name, rows in manifest["splits"].items():
            print(f"{name}: {len(rows)} slices / {len({row['patient_id'] for row in rows})} patients")

    if args.model == 'baseline':
        model = BaselineCNN(in_channels=1, num_classes=2).to(device)
        checkpoint_name = "best_baseline_model.pt"
    else:
        model = ConvNeXtTiny(
            in_channels=1,
            num_classes=2,
            drop_path_rate=args.drop_path_rate,
            head_dropout=args.head_dropout,
            layer_scale_init=args.layer_scale_init,
            variant=args.convnext_variant,
        ).to(device)
        checkpoint_name = ("best_convnext_model.pt" if args.convnext_variant == "tiny"
                           else f"best_convnext_{args.convnext_variant}_model.pt")

    if args.initialization == "imagenet":
        load_imagenet_initialization(model, args.save_dir)

    if args.loss == 'focal':
        alpha = class_balanced_patient_weights(
            train_loader.dataset,
            beta=args.class_balance_beta,
            ad_weight=args.ad_loss_weight,
        )
        train_criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma).to(device)
    else:
        train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    if (args.patient_balanced_sampling or args.position_balanced_sampling
            or args.era_balanced_sampling):
        if args.era_balanced_sampling:
            with open(args.metadata_path, encoding="utf-8") as stream:
                metadata = json.load(stream)
            sample_weights, sampling_report = era_balanced_sample_weights(
                train_loader.dataset, metadata
            )
            sampling_report["seed"] = args.seed
            with open(os.path.join(args.save_dir, "acquisition_sampling.json"), "w") as stream:
                json.dump(sampling_report, stream, indent=2, allow_nan=False)
            print("Acquisition-era sampling:", json.dumps(sampling_report))
        elif args.position_balanced_sampling:
            sample_weights, sampling_report = position_balanced_sample_weights(
                train_loader.dataset, args.position_bins
            )
            sampling_report["seed"] = args.seed
            with open(os.path.join(args.save_dir, "position_sampling.json"), "w") as stream:
                json.dump(sampling_report, stream, indent=2, allow_nan=False)
            print("Position sampling:", json.dumps(sampling_report))
        else:
            sample_weights = patient_balanced_sample_weights(train_loader.dataset)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_loader.dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            worker_init_fn=seed_data_worker,
            generator=torch.Generator().manual_seed(args.seed),
        )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = float('-inf')
    best_val_loss = float('inf')
    selection_records = {}
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, train_criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, eval_criterion, device
        )
        if not np.isfinite([train_loss, train_acc, val_loss, val_acc]).all():
            raise ValueError(f"Non-finite training or validation metrics at epoch {epoch}")
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path)
            selection_records["accuracy"] = {
                "epoch": epoch, "val_acc": val_acc, "val_loss": val_loss,
                "checkpoint": checkpoint_name,
            }
            if args.save_best_val_loss:
                selection_records["accuracy"]["checkpoint_sha256"] = hashlib.sha256(
                    Path(checkpoint_path).read_bytes()
                ).hexdigest()
        else:
            epochs_without_improvement += 1

        if args.save_best_val_loss and val_loss < best_val_loss:
            best_val_loss = val_loss
            loss_checkpoint = f"{Path(checkpoint_name).stem}_val_loss.pt"
            torch.save(model.state_dict(), os.path.join(args.save_dir, loss_checkpoint))
            selection_records["val_loss"] = {
                "epoch": epoch, "val_acc": val_acc, "val_loss": val_loss,
                "checkpoint": loss_checkpoint,
                "checkpoint_sha256": hashlib.sha256(
                    (Path(args.save_dir) / loss_checkpoint).read_bytes()
                ).hexdigest(),
            }
        if args.save_best_val_loss:
            (Path(args.save_dir) / "checkpoint_selection.json").write_text(
                json.dumps({
                    "early_stopping": "Validation accuracy; unchanged",
                    "tie_policy": "First epoch wins exact ties",
                    "selections": selection_records,
                }, indent=2, allow_nan=False), encoding="utf-8"
            )

        if args.patience and epochs_without_improvement >= args.patience:
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes. Best Val Acc: {best_val_acc:.4f}")
    if separate_run:
        with open(os.path.join(args.save_dir, "training_history.json"), "w") as stream:
            json.dump({
                "status": f"Exploratory {args.split_protocol} protocol; {args.model}/{args.convnext_variant}",
                "history": history, "best_validation_accuracy": best_val_acc,
                "training_minutes": total_time / 60,
            }, stream, indent=2, allow_nan=False)

    best_checkpoint = os.path.join(args.save_dir, checkpoint_name)
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    if args.save_validation_gradcam:
        save_validation_gradcam(
            model,
            val_loader,
            device,
            os.path.join(args.save_dir, "validation_gradcam.png"),
        )
    if not args.skip_test:
        test_loss, test_acc = evaluate(
            model, test_loader, eval_criterion, device
        )
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    curve_name = (
        f"convnext_{args.convnext_variant}_training_curves.png"
        if args.model == 'convnext'
        else f"{args.model}_training_curves.png"
    )
    plot_metrics(history, os.path.join(args.save_dir, curve_name))


if __name__ == '__main__':
    main()