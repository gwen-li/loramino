from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .tokenization import tokenize_fixed_length


class OrcaMath(Dataset):
    def __init__(self, parquet_file, tokenizer, max_length: int = 256, max_samples: int | None = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(parquet_file, max_samples=max_samples)

    def _load_data(self, parquet_file, *, max_samples: int | None):
        parquet_path = Path(parquet_file)
        if parquet_file and parquet_path.exists():
            data = pd.read_parquet(parquet_path)

            if {"question", "answer"}.issubset(data.columns):
                records = data[["question", "answer"]].to_dict("records")
                return records if max_samples is None else records[:max_samples]

            if {"questions", "answers"}.issubset(data.columns):
                renamed = data.rename(columns={"questions": "question", "answers": "answer"})
                records = renamed[["question", "answer"]].to_dict("records")
                return records if max_samples is None else records[:max_samples]

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Install the `datasets` package or provide a local OrcaMath parquet file."
            ) from exc

        dataset = load_dataset("microsoft/orca-math-word-problems-200k")["train"]
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset

    def tokenize(self, text):
        return tokenize_fixed_length(self.tokenizer, text, self.max_length)

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
