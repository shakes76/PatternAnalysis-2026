from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import modules
import dataset
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Change these to fit current folder structure
train_folder = "./ADNI/AD_NC/train"
test_folder = "./ADNI/AD_NC/test"

## Removes Randomness and allows for Reproducibility
## Based off suggestions from https://docs.pytorch.org/docs/2.15/notes/randomness.html
torch.manual_seed(50)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

train_generator = torch.Generator()
train_generator.manual_seed(50)

## Splits the ADNI dataset into training, testing and validation datasets by patient id
train_split, val_split, test_split = dataset.train_val_test_split(train_folder, test_folder, "AD", "NC")

train_loader = DataLoader(dataset.ADNI_train(train_folder, test_folder, "AD", "NC", train_split), batch_size=32, shuffle=True, generator=train_generator)
val_loader = DataLoader(dataset.ADNI_test(train_folder, test_folder, "AD", "NC", val_split), batch_size=32, shuffle=False)
test_loader = DataLoader(dataset.ADNI_test(train_folder, test_folder, "AD", "NC", test_split), batch_size=32, shuffle=False)

## Loads a saved model
check_model = modules.ConvNeXt()
check_model.to(device)
check_model.load_state_dict(torch.load('./model_weights/convnext.pth', weights_only=True, map_location=device))

## Check number of parameters in model
## https://www.geeksforgeeks.org/deep-learning/check-the-total-number-of-parameters-in-a-pytorch-model/
total_params = sum(p.numel() for p in check_model.parameters())
print(f"Total Parameters: {total_params}")

check_model.eval()

for loader in [train_loader, val_loader, test_loader]:
    correct, tp, fp, tn, fn, total = 0, 0, 0, 0, 0, 0
    all_preds, all_labels = 0, 0
    init_tensor = False

    ## Total inference time
    total_inference = 0

    ## Records logits for visualisation
    tp_x = []
    tp_y = []
    fp_x = []
    fp_y = []
    tn_x = []
    tn_y = []
    fn_x = []
    fn_y = []

    with torch.no_grad():
        for images, labels in loader:
            ## Retrieves logits and makes predictions
            images, labels = images.to(device), labels.to(device)

            ## Reset Peak GPU VRAM
            torch.cuda.reset_peak_memory_stats()

            total_inference -= time.time_ns()

            outputs = check_model(images)

            total_inference += time.time_ns()

            ## Detect Peak GPU VRAM
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Peak GPU VRAM usage: {peak_vram:.2f} GB")

            preds = outputs.argmax(dim=1)

            one = torch.ones(preds.size())
            one = one.to(device)

            ## Tensors used to calculate true and false positives and negatives
            matching = (preds == labels)
            not_matching = (preds != labels)
            ad_predicted = (preds == one)
            nc_predicted = (preds != one)

            correct += matching.sum().item()

            ## Creates a Boolean Tensor for logit identification
            ## PyTorch Documentation for logical_and found at https://docs.pytorch.org/docs/2.15/generated/torch.logical_and.html
            tp_tensor = torch.logical_and(matching, ad_predicted)
            tn_tensor = torch.logical_and(matching, nc_predicted)
            fp_tensor = torch.logical_and(not_matching, ad_predicted)
            fn_tensor = torch.logical_and(not_matching, nc_predicted)

            ## Records the number of each true/false positive/negative
            tp += tp_tensor.sum().item()
            tn += tn_tensor.sum().item()
            fp += fp_tensor.sum().item()
            fn += fn_tensor.sum().item()

            ## Records the predictions for AUROC calculation
            if init_tensor:
                all_preds = torch.cat([all_preds, preds], dim=0)
                all_labels = torch.cat([all_labels, labels], dim=0)
            else:
                init_tensor = True
                all_preds = preds
                all_labels = labels

            ## Saves logits to their corresponding category for confidence visualisation
            for i in range(outputs.size(dim=0)):
                if tp_tensor[i].item():
                    tp_x.append(outputs[i][0].item())
                    tp_y.append(outputs[i][1].item())
                elif tn_tensor[i].item():
                    tn_x.append(outputs[i][0].item())
                    tn_y.append(outputs[i][1].item())
                elif fp_tensor[i].item():
                    fp_x.append(outputs[i][0].item())
                    fp_y.append(outputs[i][1].item())
                elif fn_tensor[i].item():
                    fn_x.append(outputs[i][0].item())
                    fn_y.append(outputs[i][1].item())

                    ## Retrieves significantly confident false positive filenames for analysis
                    if outputs[i][1].item() < -1:
                        print(loader.dataset.getfilename(total + i))

            total += labels.size(0)

    print(f"Average Inference Time per Image: {total_inference / total}")

    ## Calculates the per-class precision, recall, and macro f1 score
    precision_ad = tp / (tp + fp)
    precision_nc = tn / (tn + fn)
    recall_ad = tp / (tp + fn)
    recall_nc = tn / (tn + fp)
    f1_ad = 2 * precision_ad * recall_ad / (precision_ad + recall_ad)
    f1_nc = 2 * precision_nc * recall_nc / (precision_nc + recall_nc)
    macro_f1 = (0.5 * f1_ad) + (0.5 * f1_nc)

    ## Calculates the AUROC score
    ## https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    all_labels = all_labels.to('cpu')
    all_preds = all_preds.to('cpu')
    auroc = roc_auc_score(all_labels, all_preds)

    print(f"Test accuracy: {correct/total:.4f}")
    print(f"Number of True Positives: {tp}, Number of False Positives: {fp}, Number of True Negatives: {tn}, Number of False Negatives: {fn}")
    print(f"Alzheimer's Precision: {precision_ad}, Normal Cognition Precision: {precision_nc}")
    print(f"Alzheimer's Recall: {recall_ad}, Normal Cognition Recall: {recall_nc}")
    print(f"Alzheimer's F1 Score: {f1_ad}, Normal Cognition F1 Score: {f1_nc}")
    print(f"Macro F1 Score: {macro_f1}, AUROC Score: {auroc}")

    ## Displays the logits for each image to graph the model's confidence in the assigned categories
    plt.title("True Positive")
    plt.xlabel("Logit for Cognitively Normal")
    plt.ylabel("Logit for Alzheimer's Disease")
    plt.scatter(tp_x, tp_y, c="blue", alpha=0.05)
    plt.show()

    plt.title("True Negative")
    plt.xlabel("Logit for Cognitively Normal")
    plt.ylabel("Logit for Alzheimer's Disease")
    plt.scatter(tn_x, tn_y, c="green", alpha=0.05)
    plt.show()

    plt.title("False Positive")
    plt.xlabel("Logit for Cognitively Normal")
    plt.ylabel("Logit for Alzheimer's Disease")
    plt.scatter(fp_x, fp_y, c="yellow", alpha=0.05)
    plt.show()

    plt.title("False Negative")
    plt.xlabel("Logit for Cognitively Normal")
    plt.ylabel("Logit for Alzheimer's Disease")
    plt.scatter(fn_x, fn_y, c="red", alpha=0.05)
    plt.show()