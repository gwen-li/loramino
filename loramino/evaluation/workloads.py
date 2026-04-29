"""Named workload presets for end-to-end grouped LoRA benchmarks."""

from copy import deepcopy
import re


DATASET_FAMILIES = {
    "math": {
        "dataset_type": "orca_math",
        "tiny_dataset_type": "tiny_orca_math",
        "dataset_options": {},
    },
    "classification": {
        "dataset_type": "dolly_15k",
        "tiny_dataset_type": "tiny_dolly_15k",
        "dataset_options": {"category": "classification"},
    },
    "closed_qa": {
        "dataset_type": "dolly_15k",
        "tiny_dataset_type": "tiny_dolly_15k",
        "dataset_options": {"category": "closed_qa"},
    },
    "information_extraction": {
        "dataset_type": "dolly_15k",
        "tiny_dataset_type": "tiny_dolly_15k",
        "dataset_options": {"category": "information_extraction"},
    },
    "open_qa": {
        "dataset_type": "dolly_15k",
        "tiny_dataset_type": "tiny_dolly_15k",
        "dataset_options": {"category": "open_qa"},
    },
    "summarization": {
        "dataset_type": "dolly_15k",
        "tiny_dataset_type": "tiny_dolly_15k",
        "dataset_options": {"category": "summarization"},
    },
}

_DYNAMIC_MIXED_TASKS_PATTERN = re.compile(
    r"^mixed_tasks_(?P<num_jobs>\d+)way_(?P<variant>dense|rank_skew)$"
)
_FAMILY_ROTATION = [
    "math",
    "open_qa",
    "closed_qa",
    "classification",
    "information_extraction",
    "summarization",
]
_HIGH_CONCURRENCY_DENSE_BATCH_SIZE_BY_MODEL = {
    "pythia-14m": 4,
    "pythia-70m": 4,
    "pythia-160m": 2,
    "pythia-410m": 2,
    "pythia-1b": 1,
    "pythia-2.8b": 1,
}
_HIGH_CONCURRENCY_RANK_SKEW_BATCH_SIZE_BY_MODEL = {
    "pythia-14m": 4,
    "pythia-70m": 2,
    "pythia-160m": 2,
    "pythia-410m": 1,
    "pythia-1b": 1,
    "pythia-2.8b": 1,
}
_HIGH_CONCURRENCY_DENSE_RANK = 4
_HIGH_CONCURRENCY_RANK_SKEW_PATTERN = [4, 4, 8, 8, 16, 16]


WORKLOAD_PRESETS = {
    "math_pair": {
        "description": "Two same-rank math adapters over OrcaMath.",
        "batch_size": 8,
        "jobs": [
            {"name": "math_a", "family": "math", "rank": 8},
            {"name": "math_b", "family": "math", "rank": 8},
        ],
    },
    "mixed_tasks": {
        "description": "Three equal-rank adapters across math, open QA, and summarization.",
        "batch_size": 12,
        "jobs": [
            {"name": "math_reasoning", "family": "math", "rank": 8},
            {"name": "open_qa", "family": "open_qa", "rank": 8},
            {"name": "summarization", "family": "summarization", "rank": 8},
        ],
    },
    "mixed_tasks_rank_skew": {
        "description": "Mixed task families with skewed ranks to exercise scheduler grouping.",
        "batch_size": 12,
        "job_scheduler": {"max_rank_difference": 4},
        "jobs": [
            {"name": "math_reasoning", "family": "math", "rank": 4},
            {"name": "open_qa", "family": "open_qa", "rank": 8},
            {"name": "summarization", "family": "summarization", "rank": 16},
        ],
    },
    "mixed_tasks_6way": {
        "description": "Six equal-rank adapters across diverse instruction-tuning task families in a throughput-oriented grouped regime.",
        "batch_size": 24,
        "batch_size_by_model": {
            "pythia-14m": 48,
            "pythia-70m": 32,
            "pythia-160m": 32,
            "pythia-410m": 16,
            "pythia-1b": 12,
            "pythia-2.8b": 8,
        },
        "jobs": [
            {"name": "math_reasoning", "family": "math", "rank": 8},
            {"name": "open_qa", "family": "open_qa", "rank": 8},
            {"name": "closed_qa", "family": "closed_qa", "rank": 8},
            {"name": "classification", "family": "classification", "rank": 8},
            {"name": "information_extraction", "family": "information_extraction", "rank": 8},
            {"name": "summarization", "family": "summarization", "rank": 8},
        ],
    },
    "mixed_tasks_6way_rank_skew": {
        "description": "Six diverse adapters with skewed ranks in a memory-aware grouped regime that caps co-scheduled group size.",
        "batch_size": 8,
        "batch_size_by_model": {
            "pythia-14m": 16,
            "pythia-70m": 12,
            "pythia-160m": 8,
            "pythia-410m": 8,
            "pythia-1b": 4,
            "pythia-2.8b": 4,
        },
        "job_scheduler": {"max_rank_difference": 8, "max_group_size": 2},
        "jobs": [
            {"name": "math_reasoning", "family": "math", "rank": 4},
            {"name": "open_qa", "family": "open_qa", "rank": 8},
            {"name": "closed_qa", "family": "closed_qa", "rank": 8},
            {"name": "classification", "family": "classification", "rank": 16},
            {"name": "information_extraction", "family": "information_extraction", "rank": 16},
            {"name": "summarization", "family": "summarization", "rank": 32},
        ],
    },
    "mixed_tasks_8way_heavy": {
        "description": "Eight adapters with a larger grouped batch to stress shared-backbone throughput at more realistic multi-job scale.",
        "batch_size": 32,
        "batch_size_by_model": {
            "pythia-14m": 96,
            "pythia-70m": 64,
            "pythia-160m": 48,
            "pythia-410m": 32,
            "pythia-1b": 24,
            "pythia-2.8b": 16,
        },
        "jobs": [
            {"name": "math_reasoning_a", "family": "math", "rank": 8},
            {"name": "math_reasoning_b", "family": "math", "rank": 8},
            {"name": "open_qa_a", "family": "open_qa", "rank": 8},
            {"name": "open_qa_b", "family": "open_qa", "rank": 8},
            {"name": "closed_qa", "family": "closed_qa", "rank": 8},
            {"name": "classification", "family": "classification", "rank": 8},
            {"name": "information_extraction", "family": "information_extraction", "rank": 8},
            {"name": "summarization", "family": "summarization", "rank": 8},
        ],
    },
    "mixed_tasks_8way_rank_skew_heavy": {
        "description": "Eight adapters with larger grouped batches and skewed ranks to stress both scheduling and grouped execution at higher concurrency.",
        "batch_size": 16,
        "batch_size_by_model": {
            "pythia-14m": 48,
            "pythia-70m": 32,
            "pythia-160m": 24,
            "pythia-410m": 16,
            "pythia-1b": 12,
            "pythia-2.8b": 8,
        },
        "job_scheduler": {"max_rank_difference": 8, "max_group_size": 4},
        "jobs": [
            {"name": "math_reasoning_a", "family": "math", "rank": 4},
            {"name": "math_reasoning_b", "family": "math", "rank": 4},
            {"name": "open_qa_a", "family": "open_qa", "rank": 8},
            {"name": "open_qa_b", "family": "open_qa", "rank": 8},
            {"name": "closed_qa", "family": "closed_qa", "rank": 8},
            {"name": "classification", "family": "classification", "rank": 16},
            {"name": "information_extraction", "family": "information_extraction", "rank": 16},
            {"name": "summarization", "family": "summarization", "rank": 32},
        ],
    },
    "mixed_tasks_32way_dense_round_robin": {
        "description": "32 equal-rank adapters with round-robin grouped batching to ablate the effect of contiguous chunked adapter slices.",
        "batch_size": 1,
        "batch_size_by_model": deepcopy(_HIGH_CONCURRENCY_DENSE_BATCH_SIZE_BY_MODEL),
        "grouped_batching_strategy": "round_robin",
        "jobs": [
            {
                "name": f"{_FAMILY_ROTATION[job_index % len(_FAMILY_ROTATION)]}_{(job_index // len(_FAMILY_ROTATION)) + 1:02d}",
                "family": _FAMILY_ROTATION[job_index % len(_FAMILY_ROTATION)],
                "rank": _HIGH_CONCURRENCY_DENSE_RANK,
            }
            for job_index in range(32)
        ],
    },
    "mixed_tasks_32way_rank_skew_no_scheduler": {
        "description": "32 mixed-task adapters with moderate rank skew but a disabled scheduler, forcing one unconstrained grouped execution pool.",
        "batch_size": 1,
        "batch_size_by_model": deepcopy(_HIGH_CONCURRENCY_RANK_SKEW_BATCH_SIZE_BY_MODEL),
        "job_scheduler": {"max_rank_difference": 10_000, "max_group_size": 32},
        "jobs": [
            {
                "name": f"{_FAMILY_ROTATION[job_index % len(_FAMILY_ROTATION)]}_{(job_index // len(_FAMILY_ROTATION)) + 1:02d}",
                "family": _FAMILY_ROTATION[job_index % len(_FAMILY_ROTATION)],
                "rank": _HIGH_CONCURRENCY_RANK_SKEW_PATTERN[job_index % len(_HIGH_CONCURRENCY_RANK_SKEW_PATTERN)],
            }
            for job_index in range(32)
        ],
    },
}


def _build_scaled_mixed_tasks_preset(num_jobs: int, variant: str) -> dict:
    if num_jobs < 1:
        raise ValueError("Dynamic mixed-task workloads require at least one job.")

    jobs = []
    for job_index in range(num_jobs):
        family = _FAMILY_ROTATION[job_index % len(_FAMILY_ROTATION)]
        family_instance = (job_index // len(_FAMILY_ROTATION)) + 1
        job_name = f"{family}_{family_instance:02d}"
        if variant == "dense":
            rank = _HIGH_CONCURRENCY_DENSE_RANK
        else:
            rank = _HIGH_CONCURRENCY_RANK_SKEW_PATTERN[job_index % len(_HIGH_CONCURRENCY_RANK_SKEW_PATTERN)]
        jobs.append({"name": job_name, "family": family, "rank": rank})

    if variant == "dense":
        return {
            "description": (
                f"{num_jobs} equal-rank adapters across repeated instruction-tuning task families, "
                "designed to stress high-concurrency shared-backbone throughput."
            ),
            "batch_size": 1,
            "batch_size_by_model": deepcopy(_HIGH_CONCURRENCY_DENSE_BATCH_SIZE_BY_MODEL),
            "jobs": jobs,
        }

    return {
        "description": (
            f"{num_jobs} adapters with repeated mixed-task families and moderate rank skew, "
            "designed to stress scheduler quality under high local concurrency."
        ),
        "batch_size": 1,
        "batch_size_by_model": deepcopy(_HIGH_CONCURRENCY_RANK_SKEW_BATCH_SIZE_BY_MODEL),
        "job_scheduler": {"max_rank_difference": 8, "max_group_size": 8},
        "jobs": jobs,
    }


def _resolve_workload_preset(workload_name: str) -> dict:
    preset = WORKLOAD_PRESETS.get(workload_name)
    if preset is not None:
        return preset

    dynamic_match = _DYNAMIC_MIXED_TASKS_PATTERN.fullmatch(workload_name)
    if dynamic_match is None:
        valid_names = ", ".join(sorted(WORKLOAD_PRESETS))
        raise ValueError(
            "Unknown workload "
            f"'{workload_name}'. Expected one of: {valid_names}, or a scalable pattern like "
            "'mixed_tasks_<N>way_dense' / 'mixed_tasks_<N>way_rank_skew'."
        )

    num_jobs = int(dynamic_match.group("num_jobs"))
    variant = dynamic_match.group("variant")
    return _build_scaled_mixed_tasks_preset(num_jobs, variant)


def _resolve_batch_size(preset: dict, base_model: str) -> int:
    batch_size_by_model = preset.get("batch_size_by_model", {})
    if base_model in batch_size_by_model:
        return int(batch_size_by_model[base_model])
    return int(preset["batch_size"])


def build_workload_config(
    workload_name: str,
    *,
    base_model: str = "pythia-14m",
    max_length: int = 128,
    samples_per_job: int = 128,
    optimizer_lr: float = 2e-5,
    max_grad_norm: float | None = 1.0,
    seed: int = 0,
    use_tiny_datasets: bool = False,
) -> dict:
    preset = _resolve_workload_preset(workload_name)
    jobs = []
    ranks = []
    alphas = []
    for adapter_id, job in enumerate(preset["jobs"]):
        family = DATASET_FAMILIES[job["family"]]
        dataset_type = family["tiny_dataset_type"] if use_tiny_datasets else family["dataset_type"]
        dataset_options = deepcopy(family["dataset_options"])
        dataset_options["max_samples"] = samples_per_job

        rank = int(job["rank"])
        alpha = float(job.get("alpha", 1.0))
        ranks.append(rank)
        alphas.append(alpha)
        jobs.append(
            {
                "name": job["name"],
                "adapter_id": adapter_id,
                "rank": rank,
                "alpha": alpha,
                "dataset": {
                    "type": dataset_type,
                    "path": job.get("dataset_path", ""),
                    **dataset_options,
                },
            }
        )

    config = {
        "base_model": base_model,
        "jobs": jobs,
        "batch_size": _resolve_batch_size(preset, base_model),
        "grouped_batching_strategy": preset.get("grouped_batching_strategy", "chunked"),
        "num_epochs": 1,
        "num_adaptors": len(jobs),
        "max_length": max_length,
        "adapter_type": "batched_lora",
        "lora_config": {
            "num_adapters": len(jobs),
            "rank": ranks,
            "alpha": alphas,
            "kernel_backend": "auto",
        },
        "optimizer": "adamw",
        "optimizer_params": {"lr": optimizer_lr},
        "max_grad_norm": max_grad_norm,
        "loss_function": "cross_entropy",
        "verbose": False,
        "seed": seed,
    }

    if "job_scheduler" in preset:
        config["job_scheduler"] = deepcopy(preset["job_scheduler"])

    return config


def summarize_workload(
    workload_name: str,
    *,
    base_model: str,
    samples_per_job: int,
    use_tiny_datasets: bool,
) -> dict:
    preset = _resolve_workload_preset(workload_name)
    summary_jobs = []
    for job in preset["jobs"]:
        family = DATASET_FAMILIES[job["family"]]
        summary_jobs.append(
            {
                "name": job["name"],
                "dataset_type": family["tiny_dataset_type"] if use_tiny_datasets else family["dataset_type"],
                "family": job["family"],
                "rank": job["rank"],
                "samples": samples_per_job,
            }
        )

    return {
        "description": preset["description"],
        "jobs": summary_jobs,
        "batch_size": _resolve_batch_size(preset, base_model),
        "grouped_batching_strategy": preset.get("grouped_batching_strategy", "chunked"),
        "job_scheduler": deepcopy(preset.get("job_scheduler", {})),
    }
