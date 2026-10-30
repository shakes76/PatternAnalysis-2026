import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score)
from modules import ResNet50, ConvNeXt
from dataset import train_loader, val_loader, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_amp = device.type == "cuda"

torch.backends.cudnn.benchmark = True
def train_ResNet50():
    model = ResNet50(num_classes=2).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 30

    best_val_bal_acc = 0.0
    best_epoch = 0

    patience = 10
    epochs_since_improvement = 0

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    train_times = []
    val_times = []

    for epoch in range(num_epochs):

        if device.type == "cuda":
            torch.cuda.synchronize()

        train_start = time.perf_counter()

        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        # Training loop
        for images, labels in train_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp
            ):
                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)
            
            scaler.update()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            running_correct += (predicted == labels).sum().item()

            running_total += labels.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        train_time = time.perf_counter() - train_start
        train_times.append(train_time)

        epoch_loss = (running_loss / len(train_loader.dataset))

        epoch_accuracy = (running_correct / running_total)

        if device.type == "cuda":
            torch.cuda.synchronize()

        val_start = time.perf_counter()

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_labels = []
        val_predictions = []

        with torch.no_grad():

            # Validation Loop
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += (loss.item() * images.size(0))

                _, predicted = torch.max(outputs, 1)

                val_labels.extend(labels.cpu().tolist())

                val_predictions.extend(predicted.cpu().tolist())
                
                val_correct += (predicted == labels).sum().item()

                val_total += labels.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        val_time = time.perf_counter() - val_start
        val_times.append(val_time)

        val_epoch_loss = (val_loss / len(val_loader.dataset))
        val_accuracy = (val_correct / val_total)
        val_bal_acc = balanced_accuracy_score(val_labels, val_predictions)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_accuracy)

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_epoch = epoch + 1

            epochs_since_improvement = 0

            torch.save(
                model.state_dict(),
                "resnet50_best.pth"
            )
            print(
                f"Saved new best ResNet50 at epoch "
                f"{best_epoch} "
                f"(val balanced acc "
                f"{best_val_bal_acc:.4f}, "
                f"val acc {val_accuracy:.4f})"
            )
        else:
            epochs_since_improvement += 1

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {epoch_loss:.4f} - "
            f"Train Accuracy: {epoch_accuracy:.4f} - "
            f"Val Loss: {val_epoch_loss:.4f} - "
            f"Val Accuracy: {val_accuracy:.4f} - "
            f"Val Balanced Accuracy: {val_bal_acc:.4f} - "
            f"Train Time: {train_time:.2f}s - "
            f"Val Time: {val_time:.2f}s - "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        scheduler.step()

        # Early stopping
        if epochs_since_improvement >= patience:
            print(
                f"Early stopping at epoch "
                f"{epoch + 1} "
                f"(no balanced accuracy improvement "
                f"for {patience} epochs)"
            )

            break

    total_train_time = sum(train_times)
    total_val_time = sum(val_times)

    average_train_time = (total_train_time / len(train_times))

    average_val_time = (total_val_time / len(val_times))

    print("\nTiming Summary")

    print(f"Total training time: {total_train_time:.2f} seconds")

    print(f"Average training time/epoch: {average_train_time:.2f} seconds")

    print(f"Total validation time: {total_val_time:.2f} seconds")

    print(f"Average validation time/epoch: {average_val_time:.2f} seconds")

    print(f"Best ResNet50 validation balanced accuracy: {best_val_bal_acc:.4f} at epoch {best_epoch}")
    
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet50 Training and Validation Loss")
    plt.legend()
    plt.savefig("resnet50_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet50 Training and Validation Accuracy")
    plt.legend()
    plt.savefig("resnet50_accuracy.png")
    plt.close()

def train_ConvNeXt():
    model = ConvNeXt(num_classes=2).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    num_epochs = 30

    best_val_bal_acc = 0.0
    best_epoch = 0

    patience = 10
    epochs_since_improvement = 0

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    train_times = []
    val_times = []

    for epoch in range(num_epochs):

        if device.type == "cuda":
            torch.cuda.synchronize()

        train_start = time.perf_counter()

        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        # Training loop
        for images, labels in train_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp
            ):
                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)

            scaler.update()

            running_loss += (loss.item() * images.size(0))

            _, predicted = torch.max(outputs, 1)

            running_correct += (predicted == labels).sum().item()

            running_total += labels.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        train_time = time.perf_counter() - train_start
        train_times.append(train_time)

        epoch_loss = (running_loss / len(train_loader.dataset))

        epoch_accuracy = (running_correct / running_total)

        if device.type == "cuda":
            torch.cuda.synchronize()

        val_start = time.perf_counter()

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_labels = []
        val_predictions = []

        with torch.no_grad():

            # Validation Loop
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += (loss.item() * images.size(0))

                _, predicted = torch.max(outputs, 1)

                val_labels.extend(labels.cpu().tolist())
                
                val_predictions.extend(predicted.cpu().tolist())

                val_correct += (predicted == labels).sum().item()

                val_total += labels.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        val_time = time.perf_counter() - val_start
        val_times.append(val_time)

        val_epoch_loss = (val_loss / len(val_loader.dataset))
        val_accuracy = (val_correct / val_total)
        val_bal_acc = balanced_accuracy_score(val_labels, val_predictions)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_accuracy)

        if val_bal_acc > best_val_bal_acc:

            best_val_bal_acc = val_bal_acc
            best_epoch = epoch + 1

            epochs_since_improvement = 0

            torch.save(
                model.state_dict(),
                "convnext_best.pth"
            )

            print(
                f"Saved new best model at epoch "
                f"{best_epoch} "
                f"(val balanced acc "
                f"{best_val_bal_acc:.4f}, "
                f"val acc {val_accuracy:.4f})"
            )
        else:
            epochs_since_improvement += 1

        print(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {epoch_loss:.4f} - "
                    f"Train Accuracy: {epoch_accuracy:.4f} - "
                    f"Val Loss: {val_epoch_loss:.4f} - "
                    f"Val Accuracy: {val_accuracy:.4f} - "
                    f"Val Balanced Accuracy: {val_bal_acc:.4f} - "
                    f"Train Time: {train_time:.2f}s - "
                    f"Val Time: {val_time:.2f}s - "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
        
        scheduler.step()

        # Early stopping
        if epochs_since_improvement >= patience:
            print(
                f"Early stopping at epoch "
                f"{epoch + 1} "
                f"(no balanced accuracy improvement "
                f"for {patience} epochs)"
            )
            break

    print(f"Best ConvNeXt validation balanced accuracy: {best_val_bal_acc:.4f} at epoch {best_epoch}")

    total_train_time = sum(train_times)
    total_val_time = sum(val_times)

    average_train_time = (total_train_time / len(train_times))
    
    average_val_time = (total_val_time / len(val_times))
    
    print("\nTiming Summary")
    
    print(f"Total training time: {total_train_time:.2f} seconds")
    
    print(f"Average training time/epoch: {average_train_time:.2f} seconds")
    
    print(f"Total validation time: {total_val_time:.2f} seconds")
    
    print(f"Average validation time/epoch: {average_val_time:.2f} seconds")
    
    print(f"Best ResNet50 validation balanced accuracy: {best_val_bal_acc:.4f} at epoch {best_epoch}")

    epochs = range(1, len(train_losses) + 1)
    
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ConvNeXt Training and Validation Loss")
    plt.legend()
    plt.savefig("convnext_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ConvNeXt Training and Validation Accuracy")
    plt.legend()
    plt.savefig("convnext_accuracy.png")
    plt.close()

def evaluate_model(model, data_loader, device):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_confidences = []
    all_correct = []

    running_loss = 0.0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probabilities, dim=1)

            correct = predicted == labels

            running_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)

            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().tolist())

            all_predictions.extend(predicted.cpu().tolist())
            
            all_confidences.extend(confidence.cpu().tolist())
            
            all_correct.extend(correct.cpu().tolist())

            # NC is class 1 in your current mapping
            all_probabilities.extend(probabilities[:, 1].cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="macro")
    recall = recall_score( all_labels, all_predictions, average="macro")
    f1 = f1_score( all_labels, all_predictions, average="macro")
    precision_per_class = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    auroc = roc_auc_score( all_labels, all_probabilities)
    cm = confusion_matrix( all_labels, all_predictions )
    average_loss = (running_loss / total_samples)

    return {"loss": average_loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "precision_per_class": precision_per_class, 
            "recall_per_class": recall_per_class, "auroc": auroc, "confusion_matrix": cm, "labels": all_labels, "predictions": all_predictions, 
            "confidences": all_confidences, "correct": all_correct}

def analyse_confidence(results):

    correct_confidences = [confidence for confidence, correct in zip(results["confidences"], results["correct"]) if correct]

    incorrect_confidences = [confidence for confidence, correct in zip(results["confidences"],results["correct"]) if not correct]

    average_correct = (sum(correct_confidences) / len(correct_confidences))

    if len(incorrect_confidences) > 0:
        average_incorrect = (sum(incorrect_confidences) / len(incorrect_confidences))
    else:
        average_incorrect = 0.0

    print(f"Average confidence when correct: {average_correct:.4f}")

    print(f"Average confidence when incorrect: {average_incorrect:.4f}")

    return (correct_confidences, incorrect_confidences)

def count_parameters(model):

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

def measure_peak_vram(model, data_loader, device):

    if device.type != "cuda":
        print("GPU VRAM measurement requires CUDA.")
        return None

    model.eval()

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)

    images, _ = next(iter(data_loader))

    images = images.to(device)

    with torch.no_grad():
        _ = model(images)

    torch.cuda.synchronize()

    peak_memory_bytes = (torch.cuda.max_memory_allocated(device))

    peak_memory_mb = (peak_memory_bytes / (1024 ** 2))

    return peak_memory_mb

def measure_inference_latency(model, data_loader, device, repetitions=20):

    model.eval()

    images, _ = next(iter(data_loader))

    images = images.to(device)

    # Warm-up
    with torch.no_grad():

        for _ in range(5):

            _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():

        for _ in range(repetitions):

            _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    total_time = (end_time - start_time)

    average_batch_time = (total_time / repetitions)

    latency_per_image = (average_batch_time / images.size(0))

    latency_ms = (latency_per_image * 1000)

    return latency_ms


if __name__ == "__main__":

    i = input("ResNet or ConvNeXt: ").strip().lower()

    mode = input("Train or evaluate: ").strip().lower()

    if mode == "train":
        if i == "resnet":
            train_ResNet50()

        elif i == "convnext":
            train_ConvNeXt()

        raise SystemExit

    if i == "resnet":
        model = ResNet50(num_classes=2).to(device)

        model.load_state_dict(torch.load("resnet50_best.pth", map_location=device))
    
    elif i == "convnext":
        model = ConvNeXt(num_classes=2).to(device)
        
        model.load_state_dict(torch.load("convnext_best.pth", map_location=device))

    else:
        print("Invalid choice. Enter 'resnet' or 'convnext'.")
        raise SystemExit

    eval_start = time.perf_counter()

    results = evaluate_model(model, test_loader, device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    evaluation_time = (time.perf_counter() - eval_start)

    print(f"Total evaluation time: {evaluation_time:.2f} seconds")

    analyse_confidence(results)

    print(f"\nBest {i} Test Results")
    print(f"Loss:      {results['loss']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"Macro F1:  {results['f1']:.4f}")
    print(f"AUROC:     {results['auroc']:.4f}")
    print(f"AD Precision: {results['precision_per_class'][0]:.4f}")
    print(f"NC Precision: {results['precision_per_class'][1]:.4f}")
    print(f"AD Recall: {results['recall_per_class'][0]:.4f}")
    print(f"NC Recall: {results['recall_per_class'][1]:.4f}")
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    parameter_count = count_parameters(model)
    print(f"\nTrainable parameters: {parameter_count:,}")

    peak_vram = measure_peak_vram(model, test_loader,device)

    if peak_vram is not None:
        print(f"Peak GPU VRAM: {peak_vram:.2f} MB")

    latency = measure_inference_latency(model, test_loader, device)

    print(f"Inference latency: {latency:.3f} ms/image")
