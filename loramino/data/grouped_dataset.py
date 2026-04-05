from torch.utils.data import Dataset
import torch
from ...utils import dataset_dict

class GroupedDataset(Dataset):
    def __init__(self, datasets: Dataset | list[Dataset]):
        self.datasets = datasets if isinstance(datasets, list) else [datasets]
        self.length = min(len(dataset) for dataset in self.datasets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        return torch.stack([dataset[idx] for dataset in self.datasets])
    
    
def load_dataset(datasets: tuple[str] | list[tuple[str]], tokenizer):
    dataset_objects = []
    if isinstance(datasets, tuple):
        datasets = [datasets]
    for dataset_name, dataset_path in datasets:
        if dataset_name in dataset_dict:
            dataset_objects.append(dataset_dict[dataset_name](dataset_path, tokenizer))
        else:
            raise ValueError(f"Dataset {dataset_name} not found in dataset_dict")
    return GroupedDataset(dataset_objects)