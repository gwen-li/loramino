import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset

class OrcaMath(Dataset):
    def __init__(self, parquet_file, tokenizer):
        data = pd.read_parquet(parquet_file)

        # FIX
        # Without data.head(1000)
        # UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
        # I think this is due to dataset being too large to tokenize in memory
        data = data.head(1000)

        self.questions = data["question"]
        self.answers = data["answer"]

        self.tokenizer = tokenizer
        
    
    def tokenize(self, text_list):
        return self.tokenizer(text_list,
                              truncation=True,
                              padding=True,
                              return_tensors='pt')
    

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        text = self.questions[idx] + "\n" + self.answers[idx]

        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        return {
            "input_ids" : tokenized["input_ids"].squeeze(0),
            "attention_mask" : tokenized["attention_mask"].squeeze(0),
            "labels" : tokenized["input_ids"].squeeze(0)
        }