#!/bin/fish
# This script contains commands used for evaluating models
uv run predict.py --input-folder openai-community/gpt2 --input-format SpecialFreeInstruction > outputs/basegpt2
uv run predict.py --input-folder ./gpt2_ft_instform/ > outputs/ftgpt2
uv run predict.py --input-folder ./gpt2_lora_instform/ > outputs/loragpt2
uv run predict.py --input-folder ./gpt2_ft_synth/ > outputs/ftgpt2synth
uv run predict.py --input-folder ./gpt2_lora_synth/ > outputs/loragpt2synth
