#!/bin/fish
# This script contains the commands used for training models
uv run train.py --output-folder gpt2_ft_instform
uv run train.py --lora --output-folder gpt2_lora_instform
uv run train.py --output-folder gpt2_ft_synth --synthetic
uv run train.py --lora --output-folder gpt2_lora_synth --synthetic
