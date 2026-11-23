from transformers import AutoTokenizer

from dataset import load_audited_data

import numpy as np
from config import MODEL_ID, INSTRUCTION


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Reads checkpoint name from config.py file

dataset, clean_train = load_audited_data()

report = clean_train[0]["radiology_report"] # Selects first retained training report

encoded = tokenizer(report)
# Converts report into a dictionary containing token IDS and and attention mask

print("Report:", report)
print("Token IDs:", encoded["input_ids"])
print("Token count:", len(encoded["input_ids"]))


tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
# Converts each numerical ID into its token string

print("Tokens:", tokens)
print("Attention mask:", encoded["attention_mask"])
print("Decoded report:", tokenizer.decode(
    encoded["input_ids"],
    skip_special_tokens=True,
))


 # Creates one instruction string. The purpose of this is to add an instruction for the model before the report

model_input = INSTRUCTION + report
# Joins instruction and original report into one string

prompt_encoded = tokenizer(model_input)
# Converts complete input into token IDs and an attention mask

print("\nModel input:", model_input)
print("Input token count with instruction:", len(prompt_encoded["input_ids"]))



"""
Now tokenise the plain-language referecne separately from the report.
FLAN-T5 is an encoder-decoder model. Encoder reads the report, and decoder learns
to produce the reference summary. Target therefore stays separate from the input.
"""

target = clean_train[0]["layman_report"]
# Gets reference summary from the same row as our input report. 

target_encoded = tokenizer(text_target=target)
# Tokenises the expected output. 

print("\nTarget summary:", target)
print("Target token IDs:", target_encoded["input_ids"])
print("Target token count:", len(target_encoded["input_ids"]))


"""
Find how long our examples are before choosing length limits.
Converts the instruction + jargon report into tokens, converts the layman report into tokens,
and then checks the median length and maximum length. 

The prints showed that most of the data is fairly shorrt, at about 40-60 tokens long, but the longest
examples are over 1000 long. 
"""


input_lengths = []
target_lengths = []
batch_size = 256

for start in range(0, len(clean_train), batch_size):
    # Visits positions 0, 256, 512, etc until the dataset ends. 

    batch = clean_train[start:start + batch_size]
    # Selects up to 256 rows

    inputs = [
        INSTRUCTION + report
        for report in batch["radiology_report"]
    ] # Adds our instruction to each report in the batch

    input_tokens = tokenizer(inputs, truncation=False) # Converts the inputs into tokens without shortening them
    target_tokens = tokenizer(
        text_target=batch["layman_report"],
        truncation=False,
    ) # Tokenises the corresponding summaries separately

    input_lengths.extend(
        len(ids) for ids in input_tokens["input_ids"]
    )
    target_lengths.extend(
        len(ids) for ids in target_tokens["input_ids"]
    )

for name, lengths in [("Input", input_lengths),("Target", target_lengths)]:
    print(f"\n{name} token lengths:")
    print("50th, 95th, 99th percentiles:", np.percentile(lengths, [50, 95, 99]))
    print("Maximum:", max(lengths))


"""
Quantify the truncation tradeoff before choosing the ideal input and target length limits.


Candidate limit: 256 tokens
Input: 2193 examples exceed the limit (1.46%)
Target: 2039 examples exceed the limit (1.36%)

Candidate limit: 512 tokens
Input: 359 examples exceed the limit (0.24%)
Target: 365 examples exceed the limit (0.24%)
"""


for limit in [512, 1024, 2048]:
    print(f"\nCandidate limit: {limit} tokens")

    for name, lengths in [
        ("Input", input_lengths),
        ("Target", target_lengths),
    ]:
        exceeding_count = sum(
            1 for length in lengths if length > limit
        )
        exceeding_percent = 100 * exceeding_count / len(lengths)

        print(
            f"{name}: {exceeding_count} examples exceed the limit "
            f"({exceeding_percent:.2f}%)" # displays to 2 significant figures
        )


"""
512-token limit seems to be reasonable from the previous section.
We need to check what actually gets cut if we make this the limit. 
"""

long_index = next(index for index, length in enumerate(input_lengths) if length > 512)
# Selects the first position who's length exceeds 512 token length. 

long_report = clean_train[long_index]["radiology_report"]
long_input = INSTRUCTION + long_report
# Retrieves that report and rebuilds the constructed input, which includes the report + instruction

long_ids = tokenizer(long_input)["input_ids"]
# Gets the full token sequence without truncation.

print("\nSource:", clean_train[long_index]["source"])
print("Full input token count:", len(long_ids))
print("Full report:", long_report)
print("\nText beyond token 512:")
print(tokenizer.decode(long_ids[512:], skip_special_tokens=True))
 # Special tokens include things like </s> which mark the end of a sequence