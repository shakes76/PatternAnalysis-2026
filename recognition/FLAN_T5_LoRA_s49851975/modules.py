"""
Model components for radiology report lay summarisation.

FLAN-T5 is trained in one of two ways:
- "full": every weight is updated (full fine-tuning).
- "lora": the original weights are frozen and small low-rank matrices are
  added next to chosen linear layers (LoRA, Hu et al., 2022).

LoRA is written by hand here instead of using the PEFT library.
Run `python modules.py` to run the two LoRA checks.
"""

import math

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google/flan-t5-base"

# LoRA on the attention query and value layers only (Hu et al., 2022).
QUERY_VALUE_TARGETS = ("q", "v")

# LoRA on every linear layer in the attention and feed-forward blocks (Biderman et al., 2024).
ALL_LINEAR_TARGETS = ("q", "k", "v", "o", "wi_0", "wi_1", "wo")


class LoRALinear(nn.Module):
    """A frozen linear layer W0 plus a trainable low-rank update B A scaled by alpha / r (Hu et al., 2022)."""

    def __init__(self, base_layer, r, alpha):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        # Freeze the original weight W0.
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        # A maps the input down to r numbers, B maps them back up to the output size.
        self.lora_A = nn.Parameter(torch.empty(r, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))

        # A starts random like nn.Linear and PEFT do, and B starts at zero, so the layer begins unchanged.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        """Return W0, because the T5 feed-forward code reads `wo.weight` directly."""
        return self.base_layer.weight

    def forward(self, inputs):
        output = self.base_layer(inputs)
        if not self.merged:
            # Low-rank path: the inputs go through A and then B, and the result is scaled by alpha / r.
            lora_output = inputs @ self.lora_A.t() @ self.lora_B.t()
            output = output + self.scaling * lora_output
        return output

    def merge(self):
        """Add the scaled update B A into W0 so inference is as fast as the original model."""
        if not self.merged:
            with torch.no_grad():
                self.base_layer.weight += self.scaling * (self.lora_B @ self.lora_A)
            self.merged = True


def apply_lora(model, targets, r, alpha):
    """Freeze the model and wrap every linear layer whose name is in targets with LoRALinear."""
    # Freeze all original weights; only the new LoRA matrices will be trained.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Find the layers first, because the model should not change while we loop over it.
    names_to_replace = []
    for name, module in model.named_modules():
        last_name = name.split(".")[-1]
        if last_name in targets and isinstance(module, nn.Linear):
            names_to_replace.append(name)

    # Swap each layer for its LoRA version, e.g. "encoder.block.0.layer.0.SelfAttention.q".
    for name in names_to_replace:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        original_layer = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(original_layer, r, alpha))

    return model


def build_model(mode, model_name=MODEL_NAME, r=8, alpha=16, targets=QUERY_VALUE_TARGETS):
    """Load FLAN-T5 for full fine-tuning (mode "full") or with LoRA layers (mode "lora")."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if mode == "full":
        return model
    if mode == "lora":
        return apply_lora(model, targets, r, alpha)
    raise ValueError(f"Unknown mode: {mode}. Use 'full' or 'lora'.")


def count_parameters(model):
    """Return the total and the trainable number of parameters in a model."""
    total = 0
    trainable = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return total, trainable


def count_lora_layers(model):
    """Return how many layers of the model are LoRALinear."""
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            count += 1
    return count


def check_initial_output(model_name=MODEL_NAME):
    """Check 1: before training (B = 0) the LoRA model gives the same output as the original model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    report = "Simplify this radiology report for a patient: Small left pleural effusion. No pneumothorax."
    summary = "There is a little fluid around the left lung. There is no collapsed lung."
    inputs = tokenizer(report, return_tensors="pt")
    labels = tokenizer(text_target=summary, return_tensors="pt")["input_ids"]

    # eval() turns off dropout so both models are deterministic.
    original_model = build_model("full", model_name).eval()
    with torch.no_grad():
        original_logits = original_model(input_ids=inputs["input_ids"],
                                         attention_mask=inputs["attention_mask"],
                                         labels=labels).logits

    for targets in [QUERY_VALUE_TARGETS, ALL_LINEAR_TARGETS]:
        lora_model = build_model("lora", model_name, targets=targets).eval()
        with torch.no_grad():
            lora_logits = lora_model(input_ids=inputs["input_ids"],
                                     attention_mask=inputs["attention_mask"],
                                     labels=labels).logits
        largest_difference = (original_logits - lora_logits).abs().max().item()
        print(f"Check 1, targets {targets}: {count_lora_layers(lora_model)} LoRA layers, "
              f"largest output difference = {largest_difference}")
        assert torch.allclose(original_logits, lora_logits), "LoRA output differs from the original model"


def check_trainable_parameters(model_name=MODEL_NAME):
    """Check 2: LoRA r=8 on q and v of FLAN-T5-base has 884,736 trainable parameters."""
    # 72 q/v layers (24 in the encoder, 48 in the decoder) x r=8 x (768 + 768) = 884,736.
    expected = 884736

    original_model = build_model("full", model_name)
    original_total, original_trainable = count_parameters(original_model)
    print(f"Check 2, full fine-tuning: {original_total:,} total, {original_trainable:,} trainable")

    lora_model = build_model("lora", model_name, r=8, alpha=16, targets=QUERY_VALUE_TARGETS)
    total, trainable = count_parameters(lora_model)
    print(f"Check 2, LoRA r=8 q/v: {total:,} total, {trainable:,} trainable "
          f"({100 * trainable / total:.2f}%), expected {expected:,}")
    assert trainable == expected, "Wrong number of trainable LoRA parameters"


if __name__ == "__main__":
    check_initial_output()
    check_trainable_parameters()
    print("All checks passed.")
