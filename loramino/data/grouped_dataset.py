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

from .dolly_15k import Dolly15k
from .orca_math import OrcaMath
from .tiny_dolly_15k import TinyDolly15k
from .tiny_orca_math import TinyOrcaMath


DATASET_REGISTRY = {
    "default": OrcaMath,
    "dolly_15k": Dolly15k,
    "dolly-15k": Dolly15k,
    "orca_math": OrcaMath,
    "orca-math": OrcaMath,
    "tiny_dolly_15k": TinyDolly15k,
    "tiny-dolly-15k": TinyDolly15k,
    "tiny_orca_math": TinyOrcaMath,
    "tiny-orca-math": TinyOrcaMath,
}


@dataclass(frozen=True)
class DatasetJobSpec:
    """Normalized dataset config for one logical training job."""

    dataset_type: str
    dataset_path: str
    dataset_options: dict
    adapter_id: int
    max_length: int


@dataclass(frozen=True)
class TrainingJob:
    """Runtime-ready job description used by the data layer and scheduler."""

    name: str
    adapter_id: int
    dataset_type: str
    dataset_path: str
    dataset_options: dict
    max_length: int
    rank: int
    alpha: float


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
    """Grouped view across multiple job-local datasets.

    By default we emit contiguous per-adapter chunks sized to fit the configured
    grouped batch. That makes each training step less fragmented than strict
    one-example round robin, which helps the grouped LoRA runtime see larger
    same-adapter slices inside a batch.

    The older strict round-robin order is still available for debugging by
    setting ``strategy="round_robin"``.
    """

    def __init__(
        self,
        job_datasets: list[JobDataset],
        job_groups: list[list[int]] | None = None,
        *,
        batch_size: int = 1,
        strategy: str = "chunked",
    ):
        if not job_datasets:
            raise ValueError("GroupedDataset requires at least one job dataset.")
        if strategy not in {"chunked", "round_robin"}:
            raise ValueError("GroupedDataset strategy must be 'chunked' or 'round_robin'.")

        self.job_datasets = job_datasets
        self.index_map = self._build_index_map(
            job_datasets,
            job_groups,
            batch_size=max(1, int(batch_size)),
            strategy=strategy,
        )

    @staticmethod
    def _build_index_map(
        job_datasets: list[JobDataset],
        job_groups: list[list[int]] | None,
        *,
        batch_size: int,
        strategy: str,
    ) -> list[tuple[int, int]]:
        if job_groups is None:
            job_groups = [list(range(len(job_datasets)))]

        index_map = []
        for job_group in job_groups:
            if strategy == "round_robin":
                index_map.extend(GroupedDataset._build_round_robin_index(job_datasets, job_group))
                continue
            index_map.extend(
                GroupedDataset._build_chunked_group_index(
                    job_datasets,
                    job_group,
                    batch_size=batch_size,
                )
            )
        return index_map

    @staticmethod
    def _build_round_robin_index(
        job_datasets: list[JobDataset],
        job_group: list[int],
    ) -> list[tuple[int, int]]:
        max_length = max(len(job_datasets[job_index]) for job_index in job_group)
        index_map = []
        for local_index in range(max_length):
            for job_index in job_group:
                dataset = job_datasets[job_index]
                if local_index < len(dataset):
                    index_map.append((job_index, local_index))
        return index_map

    @staticmethod
    def _per_job_chunk_sizes(job_group: list[int], batch_size: int) -> dict[int, int]:
        group_size = len(job_group)
        base_chunk = max(1, batch_size // group_size)
        remainder = max(0, batch_size - (base_chunk * group_size))
        return {
            job_index: base_chunk + (1 if offset < remainder else 0)
            for offset, job_index in enumerate(job_group)
        }

    @staticmethod
    def _build_chunked_group_index(
        job_datasets: list[JobDataset],
        job_group: list[int],
        *,
        batch_size: int,
    ) -> list[tuple[int, int]]:
        max_length = max(len(job_datasets[job_index]) for job_index in job_group)
        chunk_sizes = GroupedDataset._per_job_chunk_sizes(job_group, batch_size)
        index_map = []
        local_offsets = {job_index: 0 for job_index in job_group}

        while True:
            emitted_any = False
            for job_index in job_group:
                dataset = job_datasets[job_index]
                start = local_offsets[job_index]
                if start >= len(dataset):
                    continue
                stop = min(start + chunk_sizes[job_index], len(dataset), max_length)
                index_map.extend((job_index, local_index) for local_index in range(start, stop))
                local_offsets[job_index] = stop
                emitted_any = True
            if not emitted_any:
                break
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
            dataset_options={},
            adapter_id=adapter_id,
            max_length=default_max_length,
        )

    if isinstance(job_config, tuple) and len(job_config) == 2:
        dataset_type, dataset_path = job_config
        return DatasetJobSpec(
            dataset_type=dataset_type,
            dataset_path=dataset_path,
            dataset_options={},
            adapter_id=adapter_id,
            max_length=default_max_length,
        )

    if isinstance(job_config, dict):
        dataset_config = job_config.get("dataset")
        if isinstance(dataset_config, dict):
            dataset_type = dataset_config.get("type", job_config.get("dataset_type", default_dataset_type))
            dataset_path = dataset_config.get("path", job_config.get("path", ""))
            max_length = dataset_config.get("max_length", job_config.get("max_length", default_max_length))
            dataset_options = dict(job_config.get("dataset_options", {}))
            dataset_options.update(dataset_config.get("options", {}))
            dataset_options.update(
                {
                    key: value
                    for key, value in dataset_config.items()
                    if key not in {"type", "path", "max_length", "options"}
                }
            )
        else:
            dataset_type = job_config.get("dataset_type", job_config.get("name", default_dataset_type))
            dataset_path = dataset_config if isinstance(dataset_config, str) else job_config.get("path", "")
            max_length = job_config.get("max_length", default_max_length)
            dataset_options = dict(job_config.get("dataset_options", {}))

        return DatasetJobSpec(
            dataset_type=dataset_type,
            dataset_path=dataset_path,
            dataset_options=dataset_options,
            adapter_id=job_config.get("adapter_id", adapter_id),
            max_length=max_length,
        )

    raise TypeError(
        "Each dataset job must be a path string, a (dataset_type, path) tuple, or a config dict."
    )


def _get_num_adapters(config_options: dict) -> int:
    lora_num_adapters = config_options.get("lora_config", {}).get("num_adapters")
    if lora_num_adapters is not None:
        return lora_num_adapters
    return config_options.get("num_adaptors", 1)


def _resolve_adapter_value(
    value,
    *,
    adapter_id: int,
    num_adapters: int,
    field_name: str,
):
    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        if len(value) != num_adapters:
            raise ValueError(
                f"Expected {num_adapters} values for lora_config.{field_name}, got {len(value)}."
            )
        return value[adapter_id]

    return value


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


def build_training_jobs(config_options: dict) -> list[TrainingJob]:
    num_adapters = _get_num_adapters(config_options)
    lora_config = config_options.get("lora_config", {})
    default_rank = lora_config.get("rank", 1)
    default_alpha = lora_config.get("alpha", 1.0)

    jobs = []
    for index, job_spec in enumerate(build_dataset_job_specs(config_options)):
        raw_jobs = config_options.get("jobs") or config_options.get("dataset_jobs")
        raw_job_config = raw_jobs[index] if isinstance(raw_jobs, list) and index < len(raw_jobs) else None

        job_name = f"job_{job_spec.adapter_id}"
        job_rank = _resolve_adapter_value(
            default_rank,
            adapter_id=job_spec.adapter_id,
            num_adapters=num_adapters,
            field_name="rank",
        )
        job_alpha = _resolve_adapter_value(
            default_alpha,
            adapter_id=job_spec.adapter_id,
            num_adapters=num_adapters,
            field_name="alpha",
        )

        if isinstance(raw_job_config, dict):
            job_name = raw_job_config.get("name", job_name)
            job_rank = raw_job_config.get("rank", job_rank)
            job_alpha = raw_job_config.get("alpha", job_alpha)

        jobs.append(
            TrainingJob(
                name=job_name,
                adapter_id=job_spec.adapter_id,
                dataset_type=job_spec.dataset_type,
                dataset_path=job_spec.dataset_path,
                dataset_options=dict(job_spec.dataset_options),
                max_length=job_spec.max_length,
                rank=int(job_rank),
                alpha=float(job_alpha),
            )
        )

    return jobs


def _build_base_dataset(job: TrainingJob, tokenizer) -> Dataset:
    dataset_class = DATASET_REGISTRY.get(job.dataset_type)
    if dataset_class is None:
        valid_types = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unknown dataset_type '{job.dataset_type}'. Expected one of: {valid_types}"
        )
    return dataset_class(
        job.dataset_path,
        tokenizer,
        max_length=job.max_length,
        **job.dataset_options,
    )


def build_training_dataset(
    config_options: dict,
    tokenizer,
    *,
    jobs: list[TrainingJob] | None = None,
    job_groups: list[list[int]] | None = None,
) -> Dataset:
    # Grouping is intentionally lightweight here: we normalize job config,
    # build one dataset per job, and only interleave them if there are multiple.
    jobs = build_training_jobs(config_options) if jobs is None else list(jobs)
    if len(jobs) > 1:
        max_lengths = {job.max_length for job in jobs}
        if len(max_lengths) != 1:
            raise ValueError(
                "Grouped training currently requires all jobs to use the same max_length."
            )

    job_datasets = [
        JobDataset(_build_base_dataset(job, tokenizer), adapter_id=job.adapter_id)
        for job in jobs
    ]

    if len(job_datasets) == 1 and config_options.get("jobs") is None and config_options.get("dataset_jobs") is None:
        return job_datasets[0].dataset

    if len(job_datasets) == 1:
        return job_datasets[0]

    return GroupedDataset(
        job_datasets,
        job_groups=job_groups,
        batch_size=config_options.get("batch_size", 1),
        strategy=config_options.get("grouped_batching_strategy", "chunked"),
    )


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
