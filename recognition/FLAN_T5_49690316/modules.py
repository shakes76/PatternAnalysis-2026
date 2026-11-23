"""FLAN-T5 wrapper using Transformers and PEFT APIs (not a transformer reimplementation).

References: https://huggingface.co/google/flan-t5-small and
https://huggingface.co/docs/peft/en/quicktour (LoraConfig/get_peft_model and
PeftModel.from_pretrained patterns). T5, FLAN and LoRA papers are in README.
"""

import torch
from transformers import AutoModelForSeq2SeqLM
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from pathlib import Path


class RadiologySummarizer(torch.nn.Module): 
    # Defines class as a pytorch module, gives it standard features such as .to(device), .train(), .eval(), etc
    """Wrap pretrained FLAN-T5 for radiology summarisation."""

    def __init__(self, model_id, use_lora=False):
        super().__init__()

        adapter_config_path = Path(model_id) / "adapter_config.json"
        # Checks for config file identifying a local adapter checkpoint
        
        if adapter_config_path.is_file():
            if use_lora:
                raise ValueError(
                    "Cannot create new adapters while loading saved adapters"
                )

            adapter_config = PeftConfig.from_pretrained(model_id)

            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                adapter_config.base_model_name_or_path
            )

            self.model = PeftModel.from_pretrained(
                base_model,
                model_id,
                is_trainable=False,
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        
        # loads FLAN-T5 and stores it inside the wrapper

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, # encoder-decoder text generation
                r=8,# rank of smaller update matrices
                lora_alpha=16, # controls adapter update scaling
                lora_dropout=0.05, # applies dropoout within the adapter path during training
                target_modules=["q", "v"], # adapts t5 attention's query and value projection layers
                bias="none", # Doesn't additionally train bias parameters
            )

            self.model = get_peft_model(self.model, lora_config)
            # Wraps targeted layers with LoRA adapters and freezes base model parameters
        
        

    def forward(self, input_ids, attention_mask=None, labels=None): # Defines forward computation
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )



    def generate(self, input_ids, attention_mask=None, **generation_options):
        """
        Generate output token IDs using the wrapped FLAN-T5 model.
        **generation_options collects extra arguments into a dictionary. For example:
        "max_new_tokens=128, do_sample=False" becomes {"max_new_tokens": 128, "do_sample": False}
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_options,
        )




