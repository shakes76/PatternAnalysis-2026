import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from config import MODEL_ID
from dataset import load_audited_data, tokenize_example
from modules import RadiologySummarizer

from pathlib import Path


def main():
    torch.manual_seed(42) # Sets a random seed to reproduce the environment later

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    ) # Sets device

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) # Loads tokenizer
    model = RadiologySummarizer(MODEL_ID).to(device) # Loads model to summarise reports

    _, clean_train = load_audited_data() # Keeps the filtered training data only

    features = [
        tokenize_example(example, tokenizer)
        for example in clean_train.select(range(2))
    ] 

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    loader = DataLoader(
        features,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
    ) 

    """Above 3 blocks (features, collator, DataLoader) prepare two examples using the established preprocessing"""

    batch = next(iter(loader)) # Selecrs first batch
    batch = {
        name: tensor.to(device)
        for name, tensor in batch.items()
    } # Moves tensors to MPS

    model.eval() # Eval mode
    with torch.no_grad(): # No gradient tracking for now
        initial_loss = model(**batch).loss.item() # Calculate initial loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Initialises AdamW optimiser

    model.train() # Activate train mode

    for step in range(1, 101): # Perform 101 updates
        optimizer.zero_grad(set_to_none=True) # Clear old gradients

        outputs = model(**batch)
        assert torch.isfinite(outputs.loss).item(), "Nonfinite training loss"

        outputs.loss.backward() # Calculates loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        # Limits combined gradient norm before the update, reducing effects of unnaturally large gradients

        optimizer.step() # Changes weights

        if step == 1 or step % 10 == 0: # Print progress at step 1 and every 5th step
            print(f"Step {step}: training loss = {outputs.loss.item():.4f}")

    model.eval() # Eval mode again
    with torch.no_grad(): # Clear old weights
        final_loss = model(**batch).loss.item() #  Calculate loss again

    print(f"\nInitial evaluation-mode loss: {initial_loss:.4f}")
    print(f"Final evaluation-mode loss: {final_loss:.4f}")


    """Now compare the model's generated summaries with the references it just trained on."""

    for example_number, feature in enumerate(features, start=1): # Visits two tokenized examples and numbers them
        generation_inputs = {
            name: torch.tensor([feature[name]], device=device)
            for name in ["input_ids", "attention_mask"]
        } # Creates input tensors for one example. Only include input IDs and attention mask, not reference labels

        with torch.inference_mode(): # Generates without tracking gradients
            generated_ids = model.generate(
                **generation_inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        generated_text = tokenizer.decode( # Converts output IDs into readable text
            generated_ids[0],
            skip_special_tokens=True,
        )

        reference_text = tokenizer.decode( # Decodes unpadded reference IDs
            feature["labels"],
            skip_special_tokens=True,
        )

        print(f"\nExample {example_number}")
        print("Reference:", reference_text)
        print("Generated:", generated_text)

        """Determine whether each summary generated an end-of-sequence token or exhausted
        the output budget. """

        sequence = generated_ids[0] # Select generated sequence
        new_token_count = sequence.numel() - 1 # Count generated tokens, minus start-token
        ended_with_eos = sequence[-1].item() == tokenizer.eos_token_id

        print("Generated tokens:", new_token_count)
        print("Ended with EOS:", ended_with_eos)


        """Preserve trained weights so another script can load them later."""

        checkpoint_dir = (
            Path(__file__).resolve().parents[2] / "checkpoints" / "tiny_overfit"
        ) # Builds a path relative to train.py's location. .resolve() makes the path absolute
          # Non-absolute path would look like src/data/config.json
          # Absolute path would look like /home/user/project/src/data/config.json
          # .parent() selects the containing folder of the file. 


        model.model.save_pretrained(
            checkpoint_dir,
            safe_serialization=True, # Uses safetensors format for the weights
        ) # Saves current flan-t5 weights and config in checkpoint format. 

        tokenizer.save_pretrained(checkpoint_dir)
        # Save tokenizer files alongside the weights so that the prediction can use the matching tokenizer

        print("\nSaved diagnostic checkpoint to:", checkpoint_dir)


if __name__ == "__main__":
    main()