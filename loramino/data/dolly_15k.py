import json
from pathlib import Path

from torch.utils.data import Dataset

from .tokenization import tokenize_fixed_length


class Dolly15k(Dataset):
    """Instruction-tuning dataset backed by Databricks Dolly 15k."""

    dataset_name = "databricks/databricks-dolly-15k"

    def __init__(
        self,
        dataset_path,
        tokenizer,
        max_length: int = 256,
        category: str | None = None,
        categories: list[str] | None = None,
        max_samples: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(
            dataset_path,
            category=category,
            categories=categories,
            max_samples=max_samples,
        )

    def _load_data(
        self,
        dataset_path,
        *,
        category: str | None,
        categories: list[str] | None,
        max_samples: int | None,
    ):
        requested_categories = set(categories or ([] if category is None else [category]))

        local_path = Path(dataset_path)
        if dataset_path and local_path.exists():
            data = self._load_local_records(local_path)
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "Install the `datasets` package or provide a local Dolly JSON/JSONL file."
                ) from exc

            data = load_dataset(self.dataset_name, split="train")
            if requested_categories:
                data = data.filter(lambda row: row["category"] in requested_categories)
            if max_samples is not None:
                data = data.select(range(min(max_samples, len(data))))
            return data

        if requested_categories:
            data = [row for row in data if row.get("category") in requested_categories]
        if max_samples is not None:
            data = data[:max_samples]
        return data

    @staticmethod
    def _load_local_records(path: Path) -> list[dict]:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
                return payload["data"]
        raise ValueError("Local Dolly dataset must be a .json or .jsonl file.")

    def tokenize(self, text):
        return tokenize_fixed_length(self.tokenizer, text, self.max_length)

    @staticmethod
    def _format_prompt(example: dict) -> str:
        parts = [f"Instruction: {example['instruction']}"]
        context = example.get("context", "")
        if context:
            parts.append(f"Context: {context}")
        parts.append(f"Response: {example['response']}")
        return "\n".join(parts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        encoded = self.tokenize(self._format_prompt(example))
        encoded["labels"] = encoded["input_ids"].clone()
        return encoded
