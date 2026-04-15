"""Job-oriented dataset helpers for single-job and grouped LoRA training.

The important boundary is:
- each job keeps its own dataset identity and adapter id
- sampling/grouping stays lightweight here
- the final grouped batch is assembled late and carries ``adapter_ids`` for the runtime

This keeps the data layer simple while preserving the information the runtime
needs to route each example to the right adapter.
"""

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .orca_math import OrcaMath
from .tiny_orca_math import TinyOrcaMath


DATASET_REGISTRY = {
    "default": OrcaMath,
    "orca_math": OrcaMath,
    "orca-math": OrcaMath,
    "tiny_orca_math": TinyOrcaMath,
    "tiny-orca-math": TinyOrcaMath,
}


@dataclass(frozen=True)
class DatasetJobSpec:
    """Normalized dataset config for one logical training job."""

    dataset_type: str
    dataset_path: str
    adapter_id: int
    max_length: int


class JobDataset(Dataset):
    """Wrap one dataset with the adapter id that should consume its examples."""

    def __init__(self, dataset: Dataset, adapter_id: int):
        self.dataset = dataset
        self.adapter_id = adapter_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = dict(self.dataset[idx])
        item["adapter_id"] = self.adapter_id
        return item


class GroupedDataset(Dataset):
    """Simple round-robin view across multiple job-local datasets."""

    def __init__(self, job_datasets: list[JobDataset]):
        if not job_datasets:
            raise ValueError("GroupedDataset requires at least one job dataset.")

        self.job_datasets = job_datasets
        self.index_map = self._build_round_robin_index(job_datasets)

    @staticmethod
    def _build_round_robin_index(job_datasets: list[JobDataset]) -> list[tuple[int, int]]:
        index_map = []
        max_length = max(len(dataset) for dataset in job_datasets)
        for local_index in range(max_length):
            for job_index, dataset in enumerate(job_datasets):
                if local_index < len(dataset):
                    index_map.append((job_index, local_index))
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        job_index, local_index = self.index_map[idx]
        return self.job_datasets[job_index][local_index]


def _coerce_job_spec(
    job_config,
    *,
    adapter_id: int,
    default_dataset_type: str,
    default_max_length: int,
) -> DatasetJobSpec:
    if isinstance(job_config, str):
        return DatasetJobSpec(
            dataset_type=default_dataset_type,
            dataset_path=job_config,
            adapter_id=adapter_id,
            max_length=default_max_length,
        )

    if isinstance(job_config, tuple) and len(job_config) == 2:
        dataset_type, dataset_path = job_config
        return DatasetJobSpec(
            dataset_type=dataset_type,
            dataset_path=dataset_path,
            adapter_id=adapter_id,
            max_length=default_max_length,
        )

    if isinstance(job_config, dict):
        dataset_config = job_config.get("dataset")
        if isinstance(dataset_config, dict):
            dataset_type = dataset_config.get("type", job_config.get("dataset_type", default_dataset_type))
            dataset_path = dataset_config.get("path", job_config.get("path", ""))
            max_length = dataset_config.get("max_length", job_config.get("max_length", default_max_length))
        else:
            dataset_type = job_config.get("dataset_type", job_config.get("name", default_dataset_type))
            dataset_path = dataset_config if isinstance(dataset_config, str) else job_config.get("path", "")
            max_length = job_config.get("max_length", default_max_length)

        return DatasetJobSpec(
            dataset_type=dataset_type,
            dataset_path=dataset_path,
            adapter_id=job_config.get("adapter_id", adapter_id),
            max_length=max_length,
        )

    raise TypeError(
        "Each dataset job must be a path string, a (dataset_type, path) tuple, or a config dict."
    )


def build_dataset_job_specs(config_options: dict) -> list[DatasetJobSpec]:
    # ``jobs`` is the preferred config surface. ``dataset_jobs`` and bare
    # ``dataset`` are supported so older configs still work.
    default_dataset_type = config_options.get("dataset_type", "orca_math")
    default_max_length = config_options.get("max_length", 256)
    raw_jobs = config_options.get("jobs")

    if raw_jobs is None:
        raw_jobs = config_options.get("dataset_jobs")

    if raw_jobs is None:
        raw_jobs = [config_options.get("dataset", "")]

    if not isinstance(raw_jobs, list):
        raw_jobs = [raw_jobs]

    job_specs = [
        _coerce_job_spec(
            job_config,
            adapter_id=adapter_id,
            default_dataset_type=default_dataset_type,
            default_max_length=default_max_length,
        )
        for adapter_id, job_config in enumerate(raw_jobs)
    ]

    adapter_ids = [job_spec.adapter_id for job_spec in job_specs]
    if len(adapter_ids) != len(set(adapter_ids)):
        raise ValueError("Each job must map to a unique adapter_id.")

    return job_specs


def _build_base_dataset(job_spec: DatasetJobSpec, tokenizer) -> Dataset:
    dataset_class = DATASET_REGISTRY.get(job_spec.dataset_type)
    if dataset_class is None:
        valid_types = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unknown dataset_type '{job_spec.dataset_type}'. Expected one of: {valid_types}"
        )
    return dataset_class(
        job_spec.dataset_path,
        tokenizer,
        max_length=job_spec.max_length,
    )


def build_training_dataset(config_options: dict, tokenizer) -> Dataset:
    # Grouping is intentionally lightweight here: we normalize job config,
    # build one dataset per job, and only interleave them if there are multiple.
    job_specs = build_dataset_job_specs(config_options)
    if len(job_specs) > 1:
        max_lengths = {job_spec.max_length for job_spec in job_specs}
        if len(max_lengths) != 1:
            raise ValueError(
                "Grouped training currently requires all jobs to use the same max_length."
            )

    job_datasets = [
        JobDataset(_build_base_dataset(job_spec, tokenizer), adapter_id=job_spec.adapter_id)
        for job_spec in job_specs
    ]

    if len(job_datasets) == 1 and config_options.get("jobs") is None and config_options.get("dataset_jobs") is None:
        return job_datasets[0].dataset

    if len(job_datasets) == 1:
        return job_datasets[0]

    return GroupedDataset(job_datasets)


def grouped_batch_collator(examples: list[dict]) -> dict[str, torch.Tensor]:
    # The runtime expects normal model tensors plus optional ``adapter_ids``.
    # That keeps the rest of the stack mostly unaware of how examples were sourced.
    if not examples:
        raise ValueError("Cannot collate an empty batch.")

    batch = {}
    for key in examples[0]:
        if key == "adapter_id":
            continue

        values = [example[key] for example in examples]
        first_value = values[0]
        if isinstance(first_value, torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = torch.as_tensor(values)

    if "adapter_id" in examples[0]:
        batch["adapter_ids"] = torch.tensor(
            [int(example["adapter_id"]) for example in examples],
            dtype=torch.long,
        )

    return batch
