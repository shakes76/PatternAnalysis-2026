"""
COMP3710 - Pattern Recognition - train.py

Author: Nicholas Ardika
Student ID: 48030102

This file provides the functionallity to train and evaluate the FLAN-T5 model.
It includes the maain experimentation process of the project which compares the 
training results using both full fine-tuning and LoRA. It also compares the 
generated summaries of the baseline, full fine-tuned and LoRA models, which is
used for the manual safety audit.
"""
import os
import torch
import random
import evaluate
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from modules import FLANT5
from dataset import BioLaySummData
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

ROOT_PATH = "./recognition/Flan_T5_s4803010"

def train_single_epoch(model, trainLoader, optimiser, scheduler, scaler):
    """Trains a single epoch of the given model and returns the training loss of the epoch
    
    Args:
        model: the FLAN-T5 model being trained
        trainLoader: a PyTorch DataLoader containing the training data
        optimiser: the optimiser used for updating parameters
        scheduler: the scheduler used to update learning rate per batch
        scaler: the Gradient Scaler (currently should be disabled since bfloat16 mixed precision is used)
    """
    model.train()
    device = next(model.parameters()).device

    totalLoss = 0.0
    for batch in tqdm(trainLoader, desc="Batch Progress", unit="batches", ncols=100):
        batch = {key: value.to(device) for key, value in batch.items()}

        optimiser.zero_grad()

        # Utilises mixed precision
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # Passes the input ids, attention masks and labels to FLAN-T5 for a forward pass
            output = model(**batch)

            # FLAN-T5 automatically produces loss, so specifying loss function isn't needed
            loss = output.loss

        # Backprop with scaled losses due to mixed precision
        scaler.scale(loss).backward()

        # Apply gradient clipping to limit gradients exploding 
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        totalLoss += loss.item()

    return totalLoss / len(trainLoader)

def validate(model, valLoader):
    """Evaluates the current model using a validation set and returns the validation loss
    
    Args:
        model: the FLAN-T5 model being trained
        valLoader: a PyTorch DataLoader containing the validation data
    """
    model.eval()
    device = next(model.parameters()).device
    
    totalLoss = 0.0
    with torch.no_grad():
        for batch in valLoader:
            batch = {key: value.to(device) for key, value in batch.items()}

            # Utilises mixed precision same as training
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = model(**batch)
                loss = output.loss

            totalLoss += loss.item()

    return totalLoss / len(valLoader)

def train(model, trainLoader, valLoader, epochs, lr, patience):
    """Main training loop for the fine tuning
    
    Trains and evaluates the model using the given hyperparameters. The final model is restored
    to the checkpoint that had the lowest validation loss. Returns training statistics including
    the training and validation losses per epoch, the training time and the peak VRAM used.
    
    Args:
        model: the FLAN-T5 model being trained
        trainLoader: a PyTorch DataLoader containing the training data
        valLoader: a PyTorch DataLoader containing the validation data
        epochs: the max epochs the model will train for
        lr: the initial learning rate used
        patience: early stopping criteria, the number of epoch where the validation
                  loss can increase before the training is stopped early 
    """
    device = next(model.parameters()).device

    startTime = time()
    trainableParams = [param for param in model.parameters() if param.requires_grad]

    # Optimiser is applied to only the parameters that can be trained
    optimiser = torch.optim.AdamW(trainableParams, lr=lr, weight_decay=0.01)

    # 'Warm up' for 5% of the total training steps, then linearly decays the learning rate
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimiser,
        num_training_steps=len(trainLoader) * epochs,
        num_warmup_steps=int(len(trainLoader) * epochs * 0.05))

    # Used for mixed precision (mp) training, however mp was changed to bfloat16 which 
    # doesn't need gradient scaling therefor the scaling is disabled 
    scaler = torch.amp.GradScaler(device.type, enabled=False)

    trainLosses, valLosses = [], []
    bestValLoss = float("inf")
    epochsWithoutImproving = 0
    bestModelPath = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    trainProgress = tqdm(range(epochs), desc="Training", unit="epoch", ncols=100)
    for epoch in trainProgress:
        trainLoss = train_single_epoch(model, trainLoader, optimiser, scheduler, scaler)
        valLoss = validate(model, valLoader)

        trainLosses.append(trainLoss)
        valLosses.append(valLoss)

        trainProgress.set_postfix({"Train Loss": f"{trainLoss:.4f}",
                                   "Val Loss": f"{valLoss:.4f}"})

        # Early stopping process
        if valLoss < bestValLoss:
            bestValLoss = valLoss
            epochsWithoutImproving = 0

            # During training the models are saved if the validation loss is better than the last 
            trainingType = "lora" if model.useLora else "full"
            bestModelPath = f"{ROOT_PATH}/models/FlanT5_{trainingType}_lr{lr}"
            model.model.save_pretrained(bestModelPath)
        else:
            epochsWithoutImproving += 1

        if epochsWithoutImproving >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    trainingTime = time() - startTime

    if device.type == "cuda":
        maxVram = torch.cuda.max_memory_allocated()
    else:
        maxVram = 0.0

    # Restore the best checkpoint not the last, to limit the affect of training instability
    if bestModelPath is not None:
        model.load_tuned_model(bestModelPath)
    
    return {"Train Losses": trainLosses,
            "Val Losses": valLosses,
            "Training Time": trainingTime,
            "Max VRAM": maxVram}

def evaluate_rouge(model, valLoader, tokeniser):
    """Evaluates the rouge metrics of the model's generated summaries 

    Also reports the inference throughput of non-special tokens, since both process
    require generating outputs over all validation datapoints.

    Args:
        model: the FLAN-T5 model being trained
        valLoader: a PyTorch DataLoader containing the validation data
        tokeniser: the tokeniser used to decode the generated and reference sequences
    """
    model.eval()
    device = next(model.parameters()).device

    generatedSummaries = []
    originalSummaries = []
    totalGenerationTime = 0.0
    totalTokensGenerated = 0.0

    with torch.no_grad():
        for batch in tqdm(valLoader, desc="Evaluating rouge", unit="batch", ncols=100):
            batch = {key: value.to(device) for key, value in batch.items()}

            startTime = time()
            batchOutput = model.generate(input_ids=batch["input_ids"], 
                                         attention_mask=batch["attention_mask"],
                                         max_new_tokens=256,
                                         no_repeat_ngram_size=5)
            totalGenerationTime += time() - startTime

            # Calculate the total number of tokens the model generates (exluding special tokens)
            specialTokens = set(tokeniser.all_special_ids)
            for seq in batchOutput:
                totalTokensGenerated += sum([token.item() not in specialTokens
                                             for token in seq])

            outputSummaries = tokeniser.batch_decode(batchOutput, skip_special_tokens=True)

            # Readds the padding tokens that were ignored 
            labels = batch["labels"].clone()
            labels[labels == -100] = tokeniser.pad_token_id
            labelSummaries = tokeniser.batch_decode(labels, skip_special_tokens=True)

            generatedSummaries.extend(outputSummaries)
            originalSummaries.extend(labelSummaries)

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=generatedSummaries, 
                           references=originalSummaries, 
                           use_stemmer=True)

    inferenceThroughput = totalTokensGenerated / totalGenerationTime if totalGenerationTime > 0 else 0.0

    return {**scores, "Inference Throughput": inferenceThroughput}

def test(model, testLoader, tokeniser, prefix, seed=12, numCases=5):
    """Returns the original report and generated summary of a sample of the test set
    
    This is used for the manual safetyaudit that is done on 5 datapoints of the test set

    Args:
        model: the FLAN-T5 model being trained
        testLoader: a PyTorch DataLoader containing the test data
        tokeniser: the tokeniser used to decode the generated and reference sequences
        prefix: the LLM instruction prompt used when generating the summaries
        seed: seed to allow for reproducibility of the random samples chosen
        numCases: the number of datapoints that returned 
    """
    model.eval()
    device = next(model.parameters()).device

    originalReports = []
    generatedSummaries = []

    # Gets the indices of the random samples and then creates a smaller 
    # DataLoader using only the random samples
    rng = random.Random(seed)
    indices = rng.sample(range(len(testLoader.dataset)), numCases)

    testSubset = DataLoader(Subset(testLoader.dataset, indices), 
                           batch_size=1, shuffle=False,
                           collate_fn=testLoader.collate_fn)

    with torch.no_grad():
        for batch in testSubset:
            batch = {key: value.to(device) for key, value in batch.items()}

            batchOutput = model.generate(input_ids=batch["input_ids"], 
                                         attention_mask=batch["attention_mask"],
                                         max_new_tokens=256,
                                         no_repeat_ngram_size=5)

            # Decodes the original report, the instruction prompt is then removed from the input
            originalReport = tokeniser.decode(batch["input_ids"][0], skip_special_tokens=True)
            originalReport = originalReport.removeprefix(prefix).strip()

            outputSummary = tokeniser.decode(batchOutput[0], skip_special_tokens=True)

            originalReports.append(originalReport)
            generatedSummaries.append(outputSummary)

    print(f"{'-'*10} Beginning Safety Audit {'-'*10}")
    for i in range(numCases):
        print(f"\nSample {i+1}")
        print(f"Original Report: {originalReports[i]}")
        print(f"Generated Summary: {generatedSummaries[i]}")

    return {"Original Reports": originalReports, 
            "Generated Summaries": generatedSummaries}

def plot_training_loss(trainLosses, valLosses, modelName):
    """Plots the training and validation losses per epoch and saves the plot
    
    Args:
        trainLosses: a list of the train losses over the trained epochs
        valLosses: a list of the validation losses over the trained epochs
        modelName: the type of fine-tuning used (Full or LoRA)
    """
    plt.figure(figsize=(8,6))

    epochs = range(1, len(trainLosses)+1)

    plt.plot(epochs, trainLosses, label="Training Loss")
    plt.plot(epochs, valLosses, label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss for {modelName} Fine-Tuned Model")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{ROOT_PATH}/results/loss_plot_{modelName}.png")
    plt.close()

def plot_rouge_comparison():
    """Plots a grouped bar chart comparing ROUGE on the three FLAN-T5 models

    Requires the evaluation results from the experiments to be saved at
    ./results/results.csv
    """
    results = pd.read_csv(f"{ROOT_PATH}/results/results.csv")
    results = results.set_index("Model")

    metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    models = ["untuned", "full", "lora"]

    rougeScores = {model: results.loc[model, metrics].round(4).tolist() for model in models}

    fig, ax = plt.subplots(figsize=(10,6), layout="constrained")

    bars = ax.grouped_bar(rougeScores, tick_labels=metrics, group_spacing=1)
    for container in bars.bar_containers:
        ax.bar_label(container, padding=3)

    ax.set_ylabel("ROUGE Score")
    ax.set_title("ROUGE Comparison for Each FLAN-T5 Model")
    ax.legend()

    plt.savefig(f"{ROOT_PATH}/results/rouge_plot.png")
    plt.close()

def main_experiment():
    """Runs the main experiments of the project
    
    Trains the FLAN-T5 model using both full fine-tuning and LoRA, and saves the
    training statistics/results for later comparision. Then runs and compares 
    inference on the two models plus the baseline model and saves the results. 
    
    """
    print(f"{'-'*10} Beginning Experiments {'-'*10}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(12)

    os.makedirs(f"{ROOT_PATH}/results", exist_ok=True)
    os.makedirs(f"{ROOT_PATH}/models", exist_ok=True)

    # Loads and preprocesses the data
    print(f"\n{'-'*10} Loading Data {'-'*10}")
    dataset = BioLaySummData()
    dataset.setup_default()
    tokeniser = dataset.tokeniser

    trainLoader = dataset.get_dataloader("train", 16)
    valLoader = dataset.get_dataloader("validation", 16)
    testLoader = dataset.get_dataloader("test", 16)

    learningRates = {"full": 5e-5, "lora": 1e-4}
    results = {}
    safetySamples = {}

    # Training loop
    for modelType in ["full", "lora"]:
        print(f"\n{'-'*10} Training {modelType} Fine-Tuned Model {'-'*10}")

        model = FLANT5(useLora=(modelType=="lora")).to(device)
        lr = learningRates[modelType]

        # Train the model
        trainingResults = train(model, trainLoader, valLoader, epochs=4, lr=lr, patience=2)

        # Get the epoch that had the best validation loss for documentation
        bestEpoch = trainingResults["Val Losses"].index(min(trainingResults["Val Losses"]))

        # Generates and saves the training loss plot
        plot_training_loss(trainingResults["Train Losses"], 
                           trainingResults["Val Losses"], 
                           modelName=modelType)

        results[modelType] = {"Best Epoch": bestEpoch + 1,
                              "Best Train Loss": trainingResults["Train Losses"][bestEpoch],
                              "Best Val Loss": trainingResults["Val Losses"][bestEpoch],
                              "Training Time": trainingResults["Training Time"],
                              "Peak VRAM (GB)": trainingResults["Max VRAM"] / (1024**3)}

        save_data(results, safetySamples)

        # Frees up memory before loading and training the next model
        del model

    # Evaluation Loop
    for modelType in ["untuned", "full", "lora"]:
        print(f"\n{'-'*10} Evaluating {modelType} Model {'-'*10}")

        model = FLANT5(useLora=(modelType=="lora")).to(device)

        # Restores the saved checkpoints for the two models that were fine-tuned
        if modelType != "untuned": 
            lr = learningRates[modelType]
            model.load_tuned_model(f"{ROOT_PATH}/models/FlanT5_{modelType}_lr{lr}")

        # Gather evaluation information
        rougeScores = evaluate_rouge(model, valLoader, tokeniser)
        paramCounts = model.display_parameter_counts()
        sampleGenerations = test(model, testLoader, tokeniser, dataset.prefix)

        results.setdefault(modelType, {})
        results[modelType].update(rougeScores)
        results[modelType].update(paramCounts)

        safetySamples[modelType] = sampleGenerations

        save_data(results, safetySamples)

        # Frees up memory before loading and training the next model
        del model

    # Create a save plot comparing ROUGE across models
    plot_rouge_comparison()

def save_data(results, safetySamples):
    """Saves the training results and sample of generated summaries"""
    results = pd.DataFrame.from_dict(results, orient="index")
    results.to_csv(f"{ROOT_PATH}/results/results.csv", index_label="Model")

    safetySamples = pd.DataFrame.from_dict(safetySamples, orient="index")
    safetySamples.to_csv(f"{ROOT_PATH}/results/safetySamples.csv", index_label="Model")


if __name__ == "__main__":
    main_experiment()