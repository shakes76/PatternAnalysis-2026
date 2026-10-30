"""
COMP3710 Pattern Analysis - Final Assessment Project
Title: Clinical Radiology Report Summarisation with FLAN-T5 and Custom LoRA
Author: Aidan Fassy
Student ID: 44368133
File: modules.py
Description: Pure PyTorch LoRALinear implementation and ClinicalFlanT5 wrapper
             with support for zero-shot, LoRA, and full fine-tuning.
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Attention
from typing import Optional, Tuple

# ----------------- Constants -----------------

MODEL: str = "google/flan-t5-base"

# ------------------ Classes ------------------
class TrainingConstants:
    ZERO_SHOT: str = "zero_shot"
    FINETUNE: str = "full_finetune"
    LORA: str = "lora"

class LoRALinear(nn.Module):
    """"LoRA wrapper for a frozen linear layer"""

    def __init__(self, base_layer: nn.Linear, r: int = 8,
                 alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()

        # Freezes the initial base layer, no gradient update
        self._base_layer = base_layer
        self._base_layer.weight.requires_grad = False

        # Hyperparameters
        self._r = r
        self._alpha = alpha
        self._dropout = dropout
        self._in_features: int = self._base_layer.in_features
        self._out_features: int = self._base_layer.out_features
        self._scaling: float = alpha / r

        # Create the dropout layer
        self._lora_dropout = (nn.Dropout(p=self._dropout) if dropout
                            > 0.0 else nn.Identity())

        # LoRA linear layers
        self._lora_A: nn.Linear = nn.Linear(self._in_features, self._r, bias=False)
        self._lora_B: nn.Linear = nn.Linear(self._r, self._out_features, bias=False)

        # Produces the Gaussian noise 
        nn.init.normal_(self._lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self._lora_B.weight)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        b_out = self._base_layer(t)

        # dropout -> LoRA A -> LoRA B
        l_out = self._lora_B(self._lora_A(self._lora_dropout(t))) * self._scaling
        return b_out + l_out

class ClinicalFlanT5(nn.Module):
    """Wrapper for FLAN-T5, to support all training modes"""
    def __init__(self, model: str = MODEL, mode: str = TrainingConstants.LORA,
                 r: int = 8, alpha: float = 32.0, dropout: float = 0.05):

        super().__init__()
        self._mode: str = mode
        self._r: int  = r
        self._alpha: float = alpha
        self._dropout: float = dropout

        self._model: T5ForConditionalGeneration = (
            T5ForConditionalGeneration.from_pretrained(
                model
            )
        )

        # Configuring the mode selection
        match mode:
            case TrainingConstants.ZERO_SHOT:
                for param in self._model.parameters():
                    param.requires_grad = False
            case TrainingConstants.FINETUNE:
                for param in self._model.parameters():
                    param.requires_grad = True 
            case TrainingConstants.LORA:
                for param in self._model.parameters():
                    param.requires_grad = False 
                self._inject_lora(r=self._r, alpha=self._alpha,
                                  dropout=self._dropout)

    def _inject_lora(self, r: int, alpha: float, dropout: float) -> None:
        """
        Wraps the query (q) and value (v) linear projections of every
        T5Attention module with trainable LoRALinear adapters.
        """
        for module in self._model.modules():
            if isinstance(module, T5Attention):
                module.q = LoRALinear(module.q, r=r, alpha=alpha,
                                    dropout=dropout)
                module.v = LoRALinear(module.v, r=r, alpha=alpha,
                                    dropout=dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                max_new_tokens: int = 128, num_beams: int = 2):
        return self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True
        )

    def count_parameters(self) -> Tuple[int, int, float]:
        """
        Returns (trainable_params, total_params, trainable_percentage).
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        pct = 100.0 * trainable / total if total > 0 else 0.0
        return trainable, total, pct