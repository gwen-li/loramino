from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class OrcaMath(Dataset):
    def __init__(self, parquet_file, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(parquet_file)

    def _load_data(self, parquet_file):
        parquet_path = Path(parquet_file)
        if parquet_file and parquet_path.exists():
            data = pd.read_parquet(parquet_path)

            if {"question", "answer"}.issubset(data.columns):
                return data[["question", "answer"]].to_dict("records")

            if {"questions", "answers"}.issubset(data.columns):
                renamed = data.rename(columns={"questions": "question", "answers": "answer"})
                return renamed[["question", "answer"]].to_dict("records")

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Install the `datasets` package or provide a local OrcaMath parquet file."
            ) from exc

        return load_dataset("microsoft/orca-math-word-problems-200k")["train"]

    def tokenize(self, text):
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example["question"]
        answer = example["answer"]
        prompt = f"Question: {question}\nAnswer: {answer}"
        encoded = self.tokenize(prompt)
        encoded["labels"] = encoded["input_ids"].clone()
        return encoded
