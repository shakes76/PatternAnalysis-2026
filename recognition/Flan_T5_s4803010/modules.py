"""
COMP3710 - Pattern Recognition - modules.py

Author: Nicholas Ardika
Student ID: 48030102

This file implements a PyTorch wrapper for the pretrained FLAN-T5 model.
It allows for the optional use of LoRA fine tuning and implements functionality
to document the number of parameters, and their types, for each model. 
"""

import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model

class FLANT5(nn.Module):
    """A PyTorch wrapper for the pretrained FLAN-T5 model
    
    Allows for both full fine-tuning and LoRA fine-tuning 
    """
    def __init__(self, modelName="google/flan-t5-base", useLora=False):
        """Initialises the pretrained FLAN-T5 model
        
        If useLora is true, LoRA training configuration is applied to the model
        """
        super().__init__()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(modelName)

        self.useLora = useLora
        if useLora:
            self.apply_lora()

    def forward(self, **kwargs):
        """Performs a forward pass of the FLAN-T5 model
        
        Provided arguments should correspond to the arguments needed to be
        passed to the FLAN-T5 model
        """
        return self.model(**kwargs)

    def generate(self, **kwargs):
        """Generates an output token sequence using the FLAN-T5 model
                
        Provided arguments should correspond to the arguments needed to be
        passed to the FLAN-T5 model
        """
        return self.model.generate(**kwargs)

    def apply_lora(self):
        """Applies LoRA adaptors to th pretrained FLAN-T5 model
        
        LoRA freezes the parameters of the base FLAN-T5 model and creates
        new trainable parameters for the model's linear layers to reduce
        the amount of parameters updated during training.
        """
        loraConfig = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, 
                                target_modules="all-linear", 
                                bias="none", task_type="SEQ_2_SEQ_LM")

        self.model = get_peft_model(self.model, loraConfig)

    def display_parameter_counts(self):
        """Display and return the total and trainable parameter counts of the model"""
        total = sum(param.numel() for param in self.model.parameters())
        trainable = sum(param.numel() for param in self.model.parameters() if param.requires_grad)

        print(f"Total parameters: {total} ~ Trainable parameters: {trainable}")

        return {"Total Parameters": total, "Trainable Parameters": trainable}

    def load_tuned_model(self, path):
        """Loads the saved weights from after LoRA or full fine-tuning the model
        
        Reloads on the neccessary weights such that weights that weren't updated
        during training aren't touched, in order to reduce computational resources
        """
        if self.useLora:
            self.model.load_adapter(path, adapter_name="fine_tuned")
            self.model.set_adapter("fine_tuned")
        else:
            device = next(self.model.parameters()).device
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)

if __name__ == "__main__":
    # Testing that the parameter counts are reasonably correct when using LoRA
    model = FLANT5(useLora=True)
    model.display_parameter_counts()

    model = FLANT5()
    model.display_parameter_counts()