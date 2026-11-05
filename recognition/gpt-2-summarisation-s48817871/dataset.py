from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import pandas as pd

"""NOTE ON TEST SPLIT
In the BioLaySumm dataset, the layman reports for the test data are missing.

I simulate a test split by splitting the validation set (in its original order) at a specific point 
(around 80%), then treat that as the test split.

In training, I can then adjust hyperparameters around the validation split, use it for early stopping, etc.
This prevents me from overfitting (through hyperparameters) on all validation data.
"""
SPLIT_POINT = 0.8

class BioLaySummDataset(Dataset):
    def __init__(self, split: str, tokeniser, fmt: str, max_length=512, num_rows: int | None = None) -> None:
        assert split in ['train', 'validation', 'test'], "Unsupported Split!"
        self.split = split
        self.bls = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track",
            split="validation" if split in ['validation', 'test'] else split)
        self.tokeniser = tokeniser
        self.max_length = max_length
        self.fmt = fmt
        if self.split == "test":
            self.length = int(self.bls.num_rows * (1-SPLIT_POINT))
            self.split_offset = int(self.bls.num_rows * SPLIT_POINT)
        elif self.split == "validation":
            self.split_offset = 0
            self.length = int(self.bls.num_rows * SPLIT_POINT)
        else:
            self.split_offset = 0
            self.length = self.bls.num_rows
        if num_rows:
            assert num_rows < self.length and num_rows > 0, "Number of rows is out of range!"
            self.length = num_rows
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        radreport = str(self.bls['radiology_report'][index+self.split_offset])
        stupidreport = str(self.bls['layman_report'][index+self.split_offset])
        new = self.fmt.format(radreport, stupidreport)
        encoded = self.tokeniser(new, truncation=True, max_length=self.max_length, padding="max_length")
        return {
                "input_ids": torch.tensor(encoded['input_ids']),
                "attention_mask": torch.tensor(encoded['attention_mask']),
                "labels": torch.tensor(encoded['input_ids']),
                }

class BLSSynthetic(Dataset):
    def __init__(self, tokeniser, fmt: str, csv: str, max_length=512, num_rows: int | None = None, synthetic_only: bool = False) -> None:
        self.length = 0
        self.synthetic_only = synthetic_only
        self.bls = None
        if not synthetic_only:
            self.bls = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="train")
            self.length += self.bls.num_rows
        self.synth = pd.read_csv(csv)
        self.length += len(self.synth.index)
        self.tokeniser = tokeniser
        self.max_length = max_length
        self.fmt = fmt
        if num_rows != None:
            assert num_rows < self.length and num_rows > 0, "Number of rows is out of range!"
            self.length = num_rows
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        if not self.synthetic_only and self.bls != None:
            if index < self.bls.num_rows:
                radreport = str(self.bls['radiology_report'][index])
                stupidreport = str(self.bls['layman_report'][index])
            else:
                radreport = str(self.synth.iloc[index-self.bls.num_rows]['radiology_report'])
                stupidreport = str(self.synth.iloc[index-self.bls.num_rows]['layman_report'])
        else:
            radreport = str(self.synth.iloc[index]['radiology_report'])
            stupidreport = str(self.synth.iloc[index]['layman_report'])
        new = self.fmt.format(radreport, stupidreport)
        encoded = self.tokeniser(new, truncation=True, max_length=self.max_length, padding="max_length")
        return {
                "input_ids": torch.tensor(encoded['input_ids']),
                "attention_mask": torch.tensor(encoded['attention_mask']),
                "labels": torch.tensor(encoded['input_ids']),
                }

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorWithPadding
    from tqdm import tqdm
    tokeniser = AutoTokenizer.from_pretrained(
        "openai-community/gpt2", padding_side='left',
    )
    tokeniser.pad_token = tokeniser.eos_token
    data_collator = DataCollatorWithPadding(tokeniser)
    train_dataset = BLSSynthetic(tokeniser, "{}, {}", "synthetic.csv", synthetic_only=True)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=128, collate_fn=data_collator)
    for batch in tqdm(train_dataloader):
        pass
