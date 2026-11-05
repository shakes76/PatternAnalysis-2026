- GPT2 sizes (at bf16)
    - small works at 7.3GB vram, batch size of 16
        - takes 18 minutes to train through 5 epochs with this config on horoscope dataset
    - medium works at 7.5GB vram, batch size of 2
        - takes 1 hour, 5 minutes to train through 5 epochs on horoscope dataset
    - large and XL does not work

- normal finetuning took 29 seconds to get through 100 steps at 3.34 it/s
- lora (rank 8) took 23 seconds to get through 100 steps at 4.35 it/s
