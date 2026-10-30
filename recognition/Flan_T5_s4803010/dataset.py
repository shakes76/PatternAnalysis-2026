"""
COMP3710 - Pattern Recognition - dataset.py

Author: Nicholas Ardika
Student ID: 48030102

This file loads and processes the BioLaySumm 2025 dataset. The file provides
implementations for tokenisation, dynamic batch padding and PyTorch DataLoaders 
for training, validation and testing.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

class BioLaySummData():
    """Data class used to perform loading and preprocessing of the BioLaySumm dataset from Hugging Face"""
    def __init__(self):
        self.dataPath = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
        self.data = None

        # The maximum length of each sequences tokenisation representation before truncation is performed
        self.maxInputTokens = 512
        self.maxLabelTokens = 256

        # LLM instruction that are added to the input texts
        self.prefix = ("Summarise the medical report into lay language for a general "
                       "audience, the summary should be simpler and easier to understand"
                       "whilst still preserving all important clinical findings: ")

        self.tokeniser = None
        self.collator = None

    def load_data(self):
        """Downloads the dataset from Hugging Face"""
        self.data = load_dataset(self.dataPath)

    def load_tokeniser(self, tokeniserName="google/flan-t5-base"):
        """Initialises the tokeniser for the specified model"""
        self.tokeniser = AutoTokenizer.from_pretrained(tokeniserName)

    def load_collator(self):
        """Initialises the collator used for data batch processing
        
        Input and label sequences are dynamically padded to the max length in 
        each batch. Label padding is converted to -100 to ensure padded tokens
        don't affect the loss.
        """
        if self.tokeniser is None:
            raise RuntimeError("Tokeniser not initialised. Run load_tokeniser() first.")
        
        self.collator = DataCollatorForSeq2Seq(tokenizer=self.tokeniser, padding=True, label_pad_token_id=-100)

    def setup_default(self):
        """Initialises the default tokeniser, collator and dataset for general use"""
        self.load_tokeniser()
        self.load_collator()
        self.load_data()   

    def apply_prefix(self, data):
        """Adds the summarisation instruction to the start of each radiology report"""
        return [self.prefix + report for report in data["radiology_report"]]

    def tokenise_data(self, data):
        """Tokenises the radiology reports and corresponding layman summaries
        
        Input and label token sequences are truncated to the specified max length
        """
        if self.tokeniser is None:
            raise RuntimeError("Tokeniser not initialised. Run load_tokeniser() first.")
        
        inputs = self.apply_prefix(data)
        inputs = self.tokeniser(inputs, truncation=True, max_length=self.maxInputTokens)

        labels = self.tokeniser(text_target=data["layman_report"], truncation=True, max_length=self.maxLabelTokens)
        inputs["labels"] = labels["input_ids"]

        return inputs

    def process_split(self, split, batchSize):
        """Gathers, tokenises and returns the specified dataset split
        
        The argument 'split' should be 'train', 'validation' or 'test'
        """
        if self.data is None:
            raise RuntimeError("Data not initialised. Run load_data() first.")
        
        if split not in self.data:
            raise ValueError(f"Invalid dataset split: {split}")
        
        data = self.data.get(split)

        data = data.map(self.tokenise_data, batched=True, batch_size=batchSize, remove_columns=data.column_names)

        return data

    def get_dataloader(self, split, batchSize):
        """Creates a PyTorch dataloader for the specified dataset split

        The argument 'split' should be 'train', 'validation' or 'test'. 
        Data within the train split is shuffled, whereas validation and test
        splits remain unshuffled
        """
        if self.data is None or self.tokeniser is None or self.collator is None:
            raise RuntimeError("Dataset not initialised. Run setup_default() first.")

        if split not in self.data:
            raise ValueError(f"Invalid dataset split: {split}")
        
        data = self.process_split(split, 512)

        return DataLoader(data, batch_size=batchSize, shuffle=(split=="train"), 
                          collate_fn=self.collator, num_workers=4, pin_memory=True) 

if __name__ == "__main__":
    dataSet = BioLaySummData()
    dataSet.setup_default()

    trainData = dataSet.get_dataloader("train", 16)
    valData = dataSet.get_dataloader("validation", 16)
    testData = dataSet.get_dataloader("test", 16)

