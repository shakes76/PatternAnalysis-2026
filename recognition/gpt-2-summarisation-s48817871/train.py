from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from dataset import BioLaySummDataset, BLSSynthetic
import modules
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import shutil
import os
import argparse
import numpy as np
import json
device = "cuda" if torch.cuda.is_available() else "cpu"

EMUDHRA_PRIVATE_KEY = 4
torch.manual_seed(EMUDHRA_PRIVATE_KEY)

parser = argparse.ArgumentParser(add_help=True)

parser.add_argument("--output-folder", default="./modelout", type=str, help='Where to store output of training (default is a folder in the current working directory called "modelout")')
parser.add_argument("--batch-size", default=6, type=int, help="Batch size in training (default is 6)")
parser.add_argument("--save-every", default=-250, type=int, help="Save every n steps, also does evaluation of metrics (val loss, train loss, ROUGE). If negative, it saves every max-steps/n (default is 250)")
# TODO: change to -1 when not using for testing
parser.add_argument("--max-steps", default=-1, type=int, help="Max number of steps to train for (independent of epoch amount, takes priority over epoch amount), -1 to train for n epochs * number of batches in training dataset (default is -1)")
parser.add_argument("--n-epochs", default=3, type=int, help="Number of epochs to train for (default is 3)")
parser.add_argument("--model-name", default="openai-community/gpt2", type=str, help="Huggingface model to train from (default is 'openai-community/gpt2')")
parser.add_argument("--num-rows", default=None, type=int, help="Maximum number of rows to use in training dataset (default is None)")
parser.add_argument("--max-eval-steps", default=50, type=int, help="Maximum number of steps to evaluate for when calculating ROUGE and validation loss. -1 to evaluate for all batches in validation dataset (default is 50)")

parser.add_argument("--lora", action='store_true', help='Train a LoRA instead of a finetuned model (disabled by default)')

parser.add_argument("--input-format", default="InstructionFormat", help=f"Format for inputting report and summary to model. Options include {str(list(modules.formats))[1:-1]} (default is InstructionFormat)")

parser.add_argument("--synthetic", action='store_true', help='Enable usage of synthetic data during training')
parser.add_argument("--synthetic-only", action='store_true', help='Only use synthetic data in training. Only works if synthetic data is being used')
parser.add_argument("--synthetic-csv", default="synthetic.csv", type=str, help='CSV file of synthetic data (medical report and summary pairs). Only works if synthetic data is being used')

args = parser.parse_args()

if os.path.exists(args.output_folder):
    print(f"Previous folder {args.output_folder} exists, removing...")
    shutil.rmtree(args.output_folder)

if args.lora:
    print("Training with LoRA...")
else:
    print("Finetuning Model Weights...")

inputfmt = modules.get_format(args.input_format)

tokeniser, model, _ = modules.get_pretrained_model_tokeniser_lora(args, device, inputfmt)

data_collator = DataCollatorWithPadding(tokeniser)

if args.synthetic:
    train_dataset = BLSSynthetic(tokeniser, inputfmt.fmt, args.synthetic_csv, num_rows= args.num_rows, synthetic_only=args.synthetic_only)
else:
    train_dataset = BioLaySummDataset("train", tokeniser, inputfmt.fmt, num_rows = args.num_rows)
valid_dataset = BioLaySummDataset("validation", tokeniser, inputfmt.fmt)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)

n_training_steps = len(train_dataloader) * args.n_epochs if args.max_steps == -1 else args.max_steps
optimiser = optim.AdamW(model.parameters(), lr=5e-5)
scheduler = None
scheduler = lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.n_epochs)

if args.save_every < 0:
    divr = abs(args.save_every)
    save_every = n_training_steps // divr
    print(f"Saving and evaluating every {save_every} steps")
else:
    save_every = args.save_every

pbar = tqdm(range(n_training_steps))
model.to(device)
steps = 0
model.train()
best_val_loss = float('inf')
curr_train_losses = []
val_losses = {}
train_losses = {}

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"--TRAINING {n_parameters} PARAMETERS--")

for e in range(args.n_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            outputs = model(**batch)
            # model uses its own loss function that is bundled with outputs, but i've decided to go here with ForCausalLMLoss
            loss = outputs.loss
            # equivalent to `loss = ForCausalLMLoss(outputs.logits, batch['labels'], tokeniser.vocab_size)`
            # under the hood, it uses cross entropy loss
            curr_train_losses.append(loss.item())
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        pbar.update()
        if steps % save_every == 0:
            model.eval()
            # CALL EVAL FUNC HERE
            avg_val_loss, std_val_loss = modules.val_loss(model, valid_dataloader, device, max_eval_steps=args.max_eval_steps)
            avg_train_loss = np.mean(curr_train_losses)
            std_train_loss = np.std(curr_train_losses)
            curr_train_losses = []
            print(f"Validation Loss: {avg_val_loss:.5f} +- {std_val_loss:.5f}")
            print(f"Training Loss: {avg_train_loss:.5f} +- {std_train_loss:.5f}")
            if avg_val_loss <= best_val_loss:
                print(f"Early Stopping: Saving {'lora+model' if args.lora else 'model'} at {args.output_folder}")
                best_val_loss = avg_val_loss
                model.save_pretrained(args.output_folder)
            else:
                print(f"Not eligible for early stopping")
            val_losses[steps] = (avg_val_loss, std_val_loss)
            train_losses[steps] = (avg_train_loss, std_train_loss)
            model.train()
        if args.max_steps != -1 and steps >= args.max_steps:
            break
        steps += 1
    if args.max_steps != -1 and steps >= args.max_steps:
        break
    if scheduler:
        scheduler.step()
model.eval()
print("TRAINING FINISHED")
print(f"Saving {'lora+model' if args.lora else 'model'} at {args.output_folder}")
memused = torch.cuda.max_memory_allocated()
to_output = {
        'max_vram_bytes_used': memused,
        'num_train_params': n_parameters,
        'validation_loss': val_losses,
        'training_losses': train_losses
        }
with open(os.path.join(args.output_folder, "to_plot.json"), "w") as f:
    json.dump(to_output, f)

model.save_pretrained(args.output_folder)
