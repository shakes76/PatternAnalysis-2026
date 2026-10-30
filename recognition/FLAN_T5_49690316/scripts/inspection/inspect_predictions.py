import json
import random
from pathlib import Path


project_dir = Path(__file__).resolve().parents[2]
# Locate project directory

def load_predictions(folder):
    # Reads one experiment's prediction file
    path = project_dir / "results" / folder / "predictions.jsonl"

    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


lora_records = load_predictions("lora_dev_10k_eval")
full_records = load_predictions("full_dev_10k_eval")
# Loads lora results and full model results. 

assert len(lora_records) == len(full_records) 
# Check that both lora and full model contain same number of records

indices = random.Random(42).sample(range(len(lora_records)), 5)
# Choose 5 distinct positions randomly

for index in indices:
    lora = lora_records[index]
    full = full_records[index]

    assert (
        lora["source"],
        lora["images_path"],
        lora["report"],
        lora["reference"],
    ) == (
        full["source"],
        full["images_path"],
        full["report"],
        full["reference"],
    ), "Prediction files are not aligned"

    print(f"\n{'=' * 60}")
    print(f"Example index: {index} | Source: {full['source']}")
    print("\nREPORT:", full["report"])
    print("\nREFERENCE:", full["reference"])
    print("\nBASELINE:", full["baseline"])
    print("\nLORA:", lora["trained"])
    print("\nFULL:", full["trained"])