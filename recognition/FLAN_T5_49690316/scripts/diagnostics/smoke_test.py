import torch
from transformers import AutoTokenizer
from modules import RadiologySummarizer
# Import loaders for encoder-decoder text-generation model and its tokenizer

from config import INSTRUCTION
from dataset import load_audited_data

from config import MODEL_ID
# Use checkpoint id from config.py


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = RadiologySummarizer(MODEL_ID)

model = model.to(device)
model.eval() # Select evaluation mode

parameter_count = sum(
    parameter.numel() for parameter in model.parameters()
) # Counts numerical values in all model parameters. .numel() returns number of elements in each parameter tensor
# For a matrix with 100 rows and 200 columns, parameter.nume() would return 20,000

print("Device:", device)
print("Model:", MODEL_ID)
print("Parameters:", f"{parameter_count:,}")



"""
Now test the complete path from an instructed report to a generated plain-language output. 
Use one training example to check that tokenisation, GPU execution, generation, and decoding all work together.
"""

dataset, clean_train = load_audited_data()
example = clean_train[0] # Loads our prepared data and selects the first training example

model_input = INSTRUCTION + example["radiology_report"] # Combines task instruction with layman report

inputs = tokenizer(
    model_input,
    return_tensors="pt",
).to(device) # Tokenises the input into pytorch tensors, rather than python lists, and moves them to same devide as the model

with torch.inference_mode(): # Disable gradient tracking for the generation block
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False, # Disables random sampling
    ) 
# The ** means that the inputs dictionary is unpacked as input.
"""
inputs = {
    "input_ids": tensor([[465, 16, 8027, 1]]),
    "attention_mask": tensor([[1, 1, 1, 1]]),
}

Therefore, output_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
)

is equivalent to saying 
output_ids = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=128,
    do_sample=False,
)

"""

summary = tokenizer.decode(
    output_ids[0], # Selects first and only output sequence, converts to readable text
    skip_special_tokens=True,
)

print("\nReport:", example["radiology_report"])
print("Reference:", example["layman_report"])
print("Generated summary:", summary)