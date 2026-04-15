from torch.utils.data import Dataset


TINY_ORCA_MATH_EXAMPLES = [
    {"question": "What is 1 + 1?", "answer": "2"},
    {"question": "What is 2 + 3?", "answer": "5"},
    {"question": "What is 4 - 1?", "answer": "3"},
    {"question": "What is 3 * 2?", "answer": "6"},
    {"question": "What is 8 / 2?", "answer": "4"},
    {"question": "What is 5 + 4?", "answer": "9"},
    {"question": "What is 7 - 3?", "answer": "4"},
    {"question": "What is 6 + 1?", "answer": "7"},
]


class TinyOrcaMath(Dataset):
    """A tiny built-in SFT dataset for smoke tests and correctness checks."""

    def __init__(self, _dataset_path, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = list(TINY_ORCA_MATH_EXAMPLES)

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
        prompt = f"Question: {example['question']}\nAnswer: {example['answer']}"
        encoded = self.tokenize(prompt)
        encoded["labels"] = encoded["input_ids"].clone()
        return encoded
