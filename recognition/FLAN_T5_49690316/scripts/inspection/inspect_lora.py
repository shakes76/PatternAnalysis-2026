from config import MODEL_ID
from modules import RadiologySummarizer


model = RadiologySummarizer(MODEL_ID, use_lora=True)
# Loads model with adapters

total = sum(parameter.numel() for parameter in model.parameters())
# Counts all parameters


trainable = sum( # Counts only parameters with requires_grad=True, 
    #meaning they are eligible for gradient based training
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Total parameters:", f"{total:,}")
print("Trainable parameters:", f"{trainable:,}")
print("Trainable percentage:", f"{100 * trainable / total:.2f}%")

assert 0 < trainable < total, "Expected partially frozen model"