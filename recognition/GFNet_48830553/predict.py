import torch
import time
import argparse
import os
from modules import load_checkpoint, get_device, ResNet18, GFNet, get_criterion
from dataset import data_loaders
from train import PATH_BEST_MODEL, PATH_LATEST_MODEL
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryConfusionMatrix, BinaryAccuracy
import matplotlib.pyplot as plt


def compute_stats(tensor):
    """
    Computes mean, variance, and standard deviation of a PyTorch tensor using PyTorch library calls.
    """
    if tensor.numel() == 0:
        return 0.0, 0.0, 0.0
    mean = torch.mean(tensor).item()
    var = torch.var(tensor, unbiased=False).item() if tensor.numel() > 1 else 0.0
    std = torch.std(tensor, unbiased=False).item() if tensor.numel() > 1 else 0.0
    return mean, var, std


def plot_confidence_spread(tp_confs, tn_confs, fp_confs, fn_confs, save_path, model_name="Model"):
    """
    Visualises the spread of confidences across TP, TN, FP, and FN as vertically aligned histograms.
    """
    categories = [
        ("True Positive (TP)", tp_confs, "#2ca02c"),
        ("True Negative (TN)", tn_confs, "#1f77b4"),
        ("False Positive (FP)", fp_confs, "#ff7f0e"),
        ("False Negative (FN)", fn_confs, "#d62728"),
    ]

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 12), sharex=True)
    fig.suptitle(f"{model_name} - Confidence Spread by Prediction Category", fontsize=14, y=0.99)

    bins = [i / 20.0 for i in range(21)]

    for ax, (label, confs, color) in zip(axes, categories):
        if len(confs) > 0:
            ax.hist(confs, bins=bins, color=color, alpha=0.7, edgecolor="black")
            mean_val = sum(confs) / len(confs)
            ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
            ax.legend(loc="upper left")
        else:
            ax.text(0.5, 0.5, "No samples", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes)

        ax.set_title(f"{label} (n={len(confs)})", fontsize=11)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Confidence", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path)
    plt.close()


def fast_evaluate_model(model, test_loader, device):
    """
    Computes total inference latency across the test dataset with CUDA acceleration.
    """
    model.eval()
    use_cuda = device.type == "cuda"
    device_type = "cuda" if use_cuda else "cpu"
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    if use_cuda:
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device, non_blocking=use_cuda)
            batch_y = batch_y.to(device, non_blocking=use_cuda)
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_cuda):
                outputs = model(batch_X)
                _, _ = torch.max(outputs, 1)
    if use_cuda:
        torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    return total_time

def evaluate_model(model, device, output_path, conf_graph_path=None, model_name="Model"):
    """
    Evaluates the model on the test dataset with CUDA acceleration and sample-accurate confidence statistics.
    """

    print("Constructing data loaders...")
    train_loader, val_loader, test_loader = data_loaders()
    criterion = get_criterion()

    precision_metric = BinaryPrecision(device=device)
    recall_metric = BinaryRecall(device=device)
    confusion_matrix_metric = BinaryConfusionMatrix(device=device)
    test_accuracy_metric = BinaryAccuracy(device=device)
    test_loss = 0
    test_total = 0
    val_accuracy_metric = BinaryAccuracy(device=device)
    val_loss = 0
    val_total = 0
    train_accuracy_metric = BinaryAccuracy(device=device)
    train_loss = 0
    train_total = 0

    use_cuda = device.type == "cuda"
    device_type = "cuda" if use_cuda else "cpu"
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    model.eval()
    test_correct = 0

    tp_confs = []
    tn_confs = []
    fp_confs = []
    fn_confs = []

    with torch.no_grad():
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=use_cuda)
            batch_y = batch_y.to(device, non_blocking=use_cuda)
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_cuda):
                outputs = model(batch_X)

            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, 1)
            train_accuracy_metric.update(preds, batch_y)
            train_loss += criterion(outputs, batch_y).item() * batch_X.size(0)
            train_total += batch_y.size(0)

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device, non_blocking=use_cuda)
            batch_y = batch_y.to(device, non_blocking=use_cuda)
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_cuda):
                outputs = model(batch_X)

            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, 1)
            val_accuracy_metric.update(preds, batch_y)
            val_loss += criterion(outputs, batch_y).item() * batch_X.size(0)
            val_total += batch_y.size(0)

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device, non_blocking=use_cuda)
            batch_y = batch_y.to(device, non_blocking=use_cuda)
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_cuda):
                outputs = model(batch_X)

            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, 1)

            precision_metric.update(preds, batch_y)
            recall_metric.update(preds, batch_y)
            confusion_matrix_metric.update(preds, batch_y)
            test_accuracy_metric.update(preds, batch_y)

            # Create boolean masks for each outcome category
            tp_mask = (preds == 1) & (batch_y == 1)  # True Positive
            tn_mask = (preds == 0) & (batch_y == 0)  # True Negative
            fp_mask = (preds == 1) & (batch_y == 0)  # False Positive
            fn_mask = (preds == 0) & (batch_y == 1)  # False Negative

            # Save sample confidences into lists per category
            if tp_mask.any():
                tp_confs.extend(max_probs[tp_mask].detach().cpu().tolist())
            if tn_mask.any():
                tn_confs.extend(max_probs[tn_mask].detach().cpu().tolist())
            if fp_mask.any():
                fp_confs.extend(max_probs[fp_mask].detach().cpu().tolist())
            if fn_mask.any():
                fn_confs.extend(max_probs[fn_mask].detach().cpu().tolist())

            test_correct += (preds == batch_y).sum().item()
            test_total += batch_y.size(0)
            test_loss += criterion(outputs, batch_y).item() * batch_X.size(0)

    # Convert confidence lists to PyTorch tensors and compute statistics via PyTorch calls
    tp_tensor = torch.tensor(tp_confs, dtype=torch.float32)
    tn_tensor = torch.tensor(tn_confs, dtype=torch.float32)
    fp_tensor = torch.tensor(fp_confs, dtype=torch.float32)
    fn_tensor = torch.tensor(fn_confs, dtype=torch.float32)

    true_tensor = torch.cat([tp_tensor, tn_tensor]) if (len(tp_confs) + len(tn_confs) > 0) else torch.tensor([], dtype=torch.float32)
    false_tensor = torch.cat([fp_tensor, fn_tensor]) if (len(fp_confs) + len(fn_confs) > 0) else torch.tensor([], dtype=torch.float32)

    tp_mean, tp_var, tp_std = compute_stats(tp_tensor)
    tn_mean, tn_var, tn_std = compute_stats(tn_tensor)
    fp_mean, fp_var, fp_std = compute_stats(fp_tensor)
    fn_mean, fn_var, fn_std = compute_stats(fn_tensor)
    true_mean, true_var, true_std = compute_stats(true_tensor)
    false_mean, false_var, false_std = compute_stats(false_tensor)

    precision = precision_metric.compute()
    recall = recall_metric.compute()
    conf_matrix = confusion_matrix_metric.compute()
    train_accuracy = train_accuracy_metric.compute()
    val_accuracy = val_accuracy_metric.compute()
    test_accuracy = test_accuracy_metric.compute()
    total_time = fast_evaluate_model(model=model, test_loader=test_loader, device=device)

    print(f"True positive confidence: {tp_mean:.2f} (Var: {tp_var:.4f}, Std: {tp_std:.4f})")
    print(f"True negative confidence: {tn_mean:.2f} (Var: {tn_var:.4f}, Std: {tn_std:.4f})")
    print(f"False positive confidence: {fp_mean:.2f} (Var: {fp_var:.4f}, Std: {fp_std:.4f})")
    print(f"False negative confidence: {fn_mean:.2f} (Var: {fn_var:.4f}, Std: {fn_std:.4f})")
    print(f"True confidence: {true_mean:.2f} (Var: {true_var:.4f}, Std: {true_std:.4f})")
    print(f"False confidence: {false_mean:.2f} (Var: {false_var:.4f}, Std: {false_std:.4f})")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Train Accuracy: {train_accuracy:.2f} | Val Accuracy: {val_accuracy:.2f} |Test Accuracy: {test_accuracy:.2f}")
    print(f"Train Loss: {train_loss/train_total:.2f} | Val Loss: {val_loss/val_total:.2f} | Test Loss: {test_loss/test_total:.2f}")
    print(f"Inference time: {total_time:.2f}")
    print(f"Confusion Matrix: {conf_matrix}")

    with open(output_path, "w") as f:
        f.write(f"True positive confidence: {tp_mean:.2f} (Var: {tp_var:.4f}, Std: {tp_std:.4f})\n")
        f.write(f"True negative confidence: {tn_mean:.2f} (Var: {tn_var:.4f}, Std: {tn_std:.4f})\n")
        f.write(f"False positive confidence: {fp_mean:.2f} (Var: {fp_var:.4f}, Std: {fp_std:.4f})\n")
        f.write(f"False negative confidence: {fn_mean:.2f} (Var: {fn_var:.4f}, Std: {fn_std:.4f})\n")
        f.write(f"True confidence: {true_mean:.2f} (Var: {true_var:.4f}, Std: {true_std:.4f})\n")
        f.write(f"False confidence: {false_mean:.2f} (Var: {false_var:.4f}, Std: {false_std:.4f})\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"Train Accuracy: {train_accuracy:.2f} | Val Accuracy: {val_accuracy:.2f} |Test Accuracy: {test_accuracy:.2f}\n")
        f.write(f"Train Loss: {train_loss/train_total:.2f} | Val Loss: {val_loss/val_total:.2f} | Test Loss: {test_loss/test_total:.2f}\n")
        f.write(f"Inference time: {total_time:.2f}\n")
        f.write(f"Confusion Matrix: {conf_matrix}\n")

    if conf_graph_path:
        plot_confidence_spread(tp_confs, tn_confs, fp_confs, fn_confs, conf_graph_path, model_name=model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet-18 model and GFNet model testing")
    parser.add_argument("--model", default="RN18", choices=["RN18", "GFNET"], help="Model to test: 'RN18' or 'GFNET'")
    parser.add_argument("--version", default="BEST", choices=["BEST", "LATEST"], help="Version to test: 'BEST' or 'LATEST'")
    args = parser.parse_args()
    model_version = args.model
    version = args.version

    print("Obtaining device...")
    device = get_device()

    print("Obtaining model path...")
    if model_version == "RN18":
        output_path = version + "_" + model_version
        model = ResNet18()
    else:
        output_path = version + "_" + model_version
        model = GFNet()

    if version == "BEST":
            model_path = PATH_BEST_MODEL + "_" + model_version + ".pt"
    else:
        model_path = PATH_LATEST_MODEL + "_" + model_version + ".pt"

    result_path = "Results/" + output_path + "_information.txt"
    graph_path = "Results/" + output_path + "_graph.pdf"
    conf_graph_path = "Results/" + output_path + "_confidence_spread.pdf"

    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(graph_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(conf_graph_path) or ".", exist_ok=True)

    print("Loading model...")
    model_info = load_checkpoint(model_path, device=device)
    if model_info == None:
        print("Model is None")
        raise ValueError
    
    model_weights = model_info.get("model_state_dict", None)
    if model_weights == None:
        print("Model weights is None")
        raise ValueError
    
    model.load_state_dict(model_weights)
    model.to(device=device)

    print("Begin testing...")
    evaluate_model(model=model, device=device, output_path=result_path, conf_graph_path=conf_graph_path, model_name=f"{model_version} ({version})")
    
    print("Testing Visualisation")
    print(f"Result Path: {result_path}")
    print(f"Graph Path: {graph_path}")
    print(f"Confidence Spread Graph Path: {conf_graph_path}")
    
    loss_info = model_info.get("information").get("losses")
    train_losses, val_losses, epochs = loss_info
    train_losses = train_losses
    val_losses = val_losses
    epochs = epochs
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title(f"{model_version} ({version}) Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(graph_path)
    plt.close()
