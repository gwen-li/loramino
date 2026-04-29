"""Shared training runtime for both baseline and grouped LoRA experiments.

The main idea is:
- config describes jobs, adapters, and optimizer settings
- the data layer emits normal model tensors plus optional ``adapter_ids``
- the runtime decides whether to train as:
  1. one normal LoRA adapter
  2. a multi-adapter baseline reference path
  3. a batched/grouped LoRA path over one shared backbone

This keeps the user-facing API small while letting training and benchmarking
reuse the same execution code.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from time import perf_counter

import torch
from torch.utils.data import DataLoader, Sampler

from loramino.adapters.baseline_lora import BaselineLoRA
from loramino.adapters.batched_lora import BatchedLoRA, build_adapter_routing_layout
from loramino.data import (
    TrainingJob,
    build_training_jobs,
    build_training_dataset,
    grouped_batch_collator,
)
from loramino.models import loader as load_model
from .distributed import ddp_enabled, env_world_size, get_local_rank, get_rank, get_world_size, initialize_distributed, is_distributed
from .scheduler import RankAwareScheduler, ScheduledJobGroup


ADAPTER_REGISTRY = {
    "baseline": BaselineLoRA,
    "baseline_lora": BaselineLoRA,
    "batched": BatchedLoRA,
    "batched_lora": BatchedLoRA,
}


class DistributedBatchPreservingSampler(Sampler[int]):
    """Shard dataset indices across ranks without breaking precomputed local batches."""

    def __init__(
        self,
        dataset_length: int,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
    ):
        if num_replicas < 1:
            raise ValueError("num_replicas must be at least 1.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}.")

        self.dataset_length = int(dataset_length)
        self.batch_size = max(1, int(batch_size))
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.num_batches = max(1, math.ceil(self.dataset_length / self.batch_size))
        self.num_batches_per_replica = math.ceil(self.num_batches / self.num_replicas)
        self.total_batches = self.num_batches_per_replica * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_batch_blocks(self) -> list[list[int]]:
        if self.dataset_length <= 0:
            return [list(range(self.batch_size))]

        blocks = []
        for start in range(0, self.dataset_length, self.batch_size):
            block = list(range(start, min(start + self.batch_size, self.dataset_length)))
            if len(block) < self.batch_size:
                repeats = math.ceil(self.batch_size / len(block))
                block = (block * repeats)[: self.batch_size]
            blocks.append(block)

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(blocks), generator=generator).tolist()
            blocks = [blocks[index] for index in order]

        if len(blocks) < self.total_batches:
            repeats = math.ceil(self.total_batches / len(blocks))
            blocks = (blocks * repeats)[: self.total_batches]

        return blocks

    def __iter__(self):
        blocks = self._build_batch_blocks()
        rank_blocks = blocks[self.rank : self.total_batches : self.num_replicas]
        return iter(index for block in rank_blocks for index in block)

    def __len__(self) -> int:
        return self.num_batches_per_replica * self.batch_size


@dataclass
class TrainingState:
    """Mutable runtime state for one training client or benchmark case."""

    model: torch.nn.Module
    device: torch.device
    adapter_type: str
    num_adapters: int
    jobs: list[TrainingJob]
    job_groups: list[ScheduledJobGroup]
    multi_adapter_baseline: bool
    optimizer: torch.optim.Optimizer | None = None
    baseline_optimizers: list[torch.optim.Optimizer] | None = None
    baseline_adapter_states: list[dict[str, dict[str, torch.Tensor]]] | None = None
    pending_baseline_grads: list[dict[str, torch.Tensor | None] | None] | None = None
    has_pending_step: bool = False
    max_grad_norm: float | None = None
    ddp_enabled: bool = False


def get_num_adapters(config_options: dict) -> int:
    lora_num_adapters = config_options.get("lora_config", {}).get("num_adapters")
    if lora_num_adapters is not None:
        return lora_num_adapters
    return config_options.get("num_adaptors", 1)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config_options: dict) -> torch.device:
    if "device" in config_options:
        requested = str(config_options["device"])
        if requested == "cuda" and env_world_size() > 1:
            return torch.device(f"cuda:{get_local_rank()}")
        return torch.device(requested)
    if torch.cuda.is_available():
        if env_world_size() > 1:
            return torch.device(f"cuda:{get_local_rank()}")
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def normalize_adapter_type(config_options: dict) -> str:
    adapter_type = config_options.get("adapter_type", "batched_lora")
    if adapter_type not in ADAPTER_REGISTRY:
        valid_types = ", ".join(sorted(ADAPTER_REGISTRY))
        raise ValueError(f"Unknown adapter_type '{adapter_type}'. Expected one of: {valid_types}")
    return adapter_type


def _reload_model_tokenizer(model: torch.nn.Module):
    tokenizer_builder = getattr(model, "build_tokenizer", None)
    if callable(tokenizer_builder):
        tokenizer = tokenizer_builder()
        model.tokenizer = tokenizer
        return tokenizer

    tokenizer_factory = getattr(model, "tokenizer_factory", None)
    if callable(tokenizer_factory):
        tokenizer = tokenizer_factory()
        model.tokenizer = tokenizer
        return tokenizer

    return None


def _build_adapter(
    linear_layer: torch.nn.Linear,
    adapter_type: str,
    lora_config: dict,
    device: torch.device,
):
    adapter_class = ADAPTER_REGISTRY[adapter_type]
    adapter_kwargs = dict(lora_config)
    adapter_kwargs["device"] = device

    if adapter_class is BaselineLoRA:
        adapter_kwargs.pop("num_adapters", None)
        adapter_kwargs.pop("rank_groups", None)
        adapter_kwargs.pop("kernel_backend", None)

    return adapter_class(linear_layer, **adapter_kwargs)


def _replace_linear_layers(
    module: torch.nn.Module,
    adapter_type: str,
    lora_config: dict,
    device: torch.device,
) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, _build_adapter(child, adapter_type, lora_config, device))
            replaced += 1
            continue
        replaced += _replace_linear_layers(child, adapter_type, lora_config, device)
    return replaced


def _expand_per_adapter_value(value, num_adapters: int, field_name: str) -> list:
    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        if len(value) != num_adapters:
            raise ValueError(
                f"Expected {num_adapters} values for lora_config.{field_name}, got {len(value)}."
            )
        return list(value)

    return [value] * num_adapters


def build_training_jobs_and_groups(
    config_options: dict,
) -> tuple[list[TrainingJob], list[ScheduledJobGroup]]:
    jobs = build_training_jobs(config_options)
    scheduler = RankAwareScheduler.from_config(config_options)
    return jobs, scheduler.group_jobs(jobs)


def normalize_lora_config_for_jobs(
    config_options: dict,
    jobs: list[TrainingJob],
    job_groups: list[ScheduledJobGroup],
    *,
    adapter_type: str,
    num_adapters: int,
) -> dict:
    lora_config = dict(config_options["lora_config"])
    rank_values = _expand_per_adapter_value(lora_config.get("rank", 1), num_adapters, "rank")
    alpha_values = _expand_per_adapter_value(lora_config.get("alpha", 1.0), num_adapters, "alpha")

    for job in jobs:
        rank_values[job.adapter_id] = job.rank
        alpha_values[job.adapter_id] = job.alpha

    if adapter_type in {"baseline", "baseline_lora"}:
        unique_ranks = set(rank_values)
        unique_alphas = set(alpha_values)
        if num_adapters > 1 and (len(unique_ranks) > 1 or len(unique_alphas) > 1):
            raise ValueError(
                "Multi-adapter baseline currently requires all jobs to share the same rank and alpha."
            )
        lora_config["rank"] = rank_values[0]
        lora_config["alpha"] = alpha_values[0]
        return lora_config

    lora_config["rank"] = rank_values if num_adapters > 1 else rank_values[0]
    lora_config["alpha"] = alpha_values if num_adapters > 1 else alpha_values[0]
    configured_groups = [list(group.adapter_ids) for group in job_groups]
    covered_adapter_ids = {adapter_id for group in configured_groups for adapter_id in group}
    for adapter_id in range(num_adapters):
        if adapter_id not in covered_adapter_ids:
            configured_groups.append([adapter_id])
    lora_config["rank_groups"] = configured_groups
    return lora_config


def build_dataloader(
    config_options: dict,
    tokenizer,
    shuffle: bool = True,
    *,
    jobs: list[TrainingJob] | None = None,
    job_groups: list[ScheduledJobGroup] | None = None,
) -> DataLoader:
    # The data layer owns job-local datasets and batch collation. The runtime
    # only needs a standard DataLoader that may already carry adapter routing.
    if jobs is None or job_groups is None:
        jobs, job_groups = build_training_jobs_and_groups(config_options)

    dataset = build_training_dataset(
        config_options,
        tokenizer,
        jobs=jobs,
        job_groups=[list(group.job_indices) for group in job_groups],
    )
    generator = None
    if shuffle and config_options.get("seed") is not None:
        generator = torch.Generator()
        generator.manual_seed(config_options["seed"])

    sampler = None
    if ddp_enabled(config_options) and is_distributed():
        sampler = DistributedBatchPreservingSampler(
            len(dataset),
            batch_size=config_options["batch_size"],
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            seed=config_options.get("seed", 0) or 0,
        )

    return DataLoader(
        dataset,
        batch_size=config_options["batch_size"],
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        generator=generator,
        collate_fn=grouped_batch_collator,
    )


def validate_job_adapter_ids(jobs: list[TrainingJob], num_adapters: int) -> None:
    # Jobs are allowed to choose adapter ids explicitly, but they must still fit
    # inside the configured adapter bank for the current run.
    for job in jobs:
        if job.adapter_id < 0 or job.adapter_id >= num_adapters:
            raise ValueError(
                f"Job adapter_id {job.adapter_id} is out of range for num_adapters={num_adapters}."
            )


def build_optimizer(model: torch.nn.Module, config_options: dict):
    optimizer_dict = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
    }
    optimizer_class = optimizer_dict[config_options["optimizer"]]
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return optimizer_class(trainable_parameters, **config_options["optimizer_params"])


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _has_non_finite_trainable_grads(model: torch.nn.Module) -> bool:
    for parameter in _trainable_parameters(model):
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            return True
    return False


def _clip_trainable_grads(model: torch.nn.Module, max_grad_norm: float | None) -> None:
    if max_grad_norm is None:
        return
    torch.nn.utils.clip_grad_norm_(_trainable_parameters(model), max_grad_norm)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def batch_size(batch: dict) -> int:
    return next(iter(batch.values())).shape[0]


def token_count(batch: dict) -> int:
    if "input_ids" not in batch:
        return 0
    return batch["input_ids"].numel()


def infer_tokens_per_example(batch: dict, adapter_ids: torch.Tensor | None) -> int | None:
    if adapter_ids is None:
        return None

    examples = int(adapter_ids.shape[0])
    if examples == 0:
        return 0

    for key in ("input_ids", "attention_mask", "labels"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == examples:
            return value.numel() // examples

    for key, value in batch.items():
        if key == "adapter_ids":
            continue
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == examples:
            return value.numel() // examples

    return None


def build_adapter_ids(batch: dict, num_adapters: int, device: torch.device) -> torch.Tensor:
    return torch.arange(batch_size(batch), device=device) % num_adapters


def split_batch_for_adapters(batch: dict, num_adapters: int) -> list[dict]:
    adapter_ids = batch.get("adapter_ids")
    if adapter_ids is None:
        adapter_ids = build_adapter_ids(batch, num_adapters, next(iter(batch.values())).device)
    return [
        {key: value[adapter_ids == adapter_idx] for key, value in batch.items()}
        for adapter_idx in range(num_adapters)
    ]


def _unwrap_parallel_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def get_model_tokenizer(model: torch.nn.Module):
    model = _unwrap_parallel_model(model)
    tokenizer = getattr(model, "tokenizer", None)
    if callable(tokenizer):
        return tokenizer

    tokenizer = _reload_model_tokenizer(model)
    if callable(tokenizer):
        return tokenizer

    raise TypeError(f"Model {type(model).__name__} does not expose a callable tokenizer.")


def collect_baseline_lora_modules(model: torch.nn.Module) -> dict[str, BaselineLoRA]:
    model = _unwrap_parallel_model(model)
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, BaselineLoRA)
    }


def collect_batched_lora_modules(model: torch.nn.Module) -> dict[str, BatchedLoRA]:
    model = _unwrap_parallel_model(model)
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, BatchedLoRA)
    }


def set_batched_adapter_ids(
    model: torch.nn.Module,
    adapter_ids: torch.Tensor | None,
    *,
    tokens_per_example: int | None = None,
) -> None:
    shared_adapter_ids = None if adapter_ids is None else adapter_ids.detach()
    shared_layout = None
    if shared_adapter_ids is not None and tokens_per_example is not None:
        shared_layout = build_adapter_routing_layout(
            shared_adapter_ids,
            tokens_per_example=tokens_per_example,
        )

    for module in collect_batched_lora_modules(model).values():
        module.active_adapter_ids = shared_adapter_ids
        module.active_routing_layout = shared_layout


def extract_baseline_adapter_state(model: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            "A": module.A.detach().clone(),
            "B": module.B.detach().clone(),
            "alpha": module.alpha.detach().clone(),
        }
        for name, module in collect_baseline_lora_modules(model).items()
    }


def load_baseline_adapter_state(model: torch.nn.Module, state: dict[str, dict[str, torch.Tensor]]) -> None:
    with torch.no_grad():
        for name, module in collect_baseline_lora_modules(model).items():
            module.A.copy_(state[name]["A"].to(module.A.device, dtype=module.A.dtype))
            module.B.copy_(state[name]["B"].to(module.B.device, dtype=module.B.dtype))
            module.alpha.copy_(state[name]["alpha"].to(module.alpha.device, dtype=module.alpha.dtype))


def extract_batched_adapter_state(model: torch.nn.Module, adapter_index: int) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            "A": module.A[adapter_index].detach().clone(),
            "B": module.B[adapter_index].detach().clone(),
            "alpha": module.alpha_tensor[adapter_index].detach().clone(),
        }
        for name, module in collect_batched_lora_modules(model).items()
    }


def extract_all_batched_adapter_states(
    model: torch.nn.Module,
    num_adapters: int,
) -> list[dict[str, dict[str, torch.Tensor]]]:
    return [
        extract_batched_adapter_state(model, adapter_index)
        for adapter_index in range(num_adapters)
    ]


def extract_batched_lora_state(model: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            "A": module.A.detach().clone(),
            "B": module.B.detach().clone(),
            "alpha": module.alpha_tensor.detach().clone(),
        }
        for name, module in collect_batched_lora_modules(model).items()
    }


def build_matched_baseline_adapter_states(config_options: dict, num_adapters: int):
    reference_config = deepcopy(config_options)
    reference_config["adapter_type"] = "batched_lora"
    reference_config["lora_config"] = dict(config_options["lora_config"])
    reference_config["lora_config"]["num_adapters"] = num_adapters
    reference_model = setup_model(reference_config)
    return extract_all_batched_adapter_states(reference_model, num_adapters)


def extract_trainable_grads(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    gradients = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradients[name] = None if parameter.grad is None else parameter.grad.detach().clone()
    return gradients


def load_trainable_grads(model: torch.nn.Module, gradients: dict[str, torch.Tensor | None]) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = gradients[name]
        if grad is None:
            parameter.grad = None
            continue
        parameter.grad = grad.to(parameter.device, dtype=parameter.dtype).clone()


def _run_forward_backward(model: torch.nn.Module, batch: dict, optimizer=None):
    if optimizer is not None:
        optimizer.zero_grad()

    adapter_ids = batch.get("adapter_ids")
    model_batch = {key: value for key, value in batch.items() if key != "adapter_ids"}
    set_batched_adapter_ids(
        model,
        adapter_ids,
        tokens_per_example=infer_tokens_per_example(model_batch, adapter_ids),
    )
    try:
        outputs = model(**model_batch)
        loss = outputs.loss
        loss.backward()
    finally:
        set_batched_adapter_ids(model, None)
    return outputs, loss


def run_training_step(model: torch.nn.Module, batch: dict, optimizer):
    outputs, loss = _run_forward_backward(model, batch, optimizer)
    optimizer.step()
    return outputs, loss


def setup_model(config_options: dict):
    device = resolve_device(config_options)
    model = load_model.load_model(config_options["base_model"])
    lora_config = dict(config_options["lora_config"])
    adapter_type = normalize_adapter_type(config_options)

    if isinstance(lora_config["alpha"], float):
        lora_config["alpha"] = torch.tensor(lora_config["alpha"])

    model.requires_grad_(False)
    num_lora_layers = _replace_linear_layers(model, adapter_type, lora_config, device)
    model.to(device)
    if config_options["verbose"]:
        print(f"Replaced {num_lora_layers} linear layers with {adapter_type} modules.")
    return model


def create_training_state(config_options: dict) -> TrainingState:
    runtime_config = deepcopy(config_options)
    initialize_distributed()
    seed = config_options.get("seed")
    set_seed(seed)

    adapter_type = normalize_adapter_type(runtime_config)
    num_adapters = get_num_adapters(runtime_config)
    jobs, job_groups = build_training_jobs_and_groups(runtime_config)
    validate_job_adapter_ids(jobs, num_adapters)
    runtime_config["lora_config"] = normalize_lora_config_for_jobs(
        runtime_config,
        jobs,
        job_groups,
        adapter_type=adapter_type,
        num_adapters=num_adapters,
    )
    multi_adapter_baseline = adapter_type == "baseline_lora" and num_adapters > 1

    baseline_adapter_states = None
    if multi_adapter_baseline:
        # Reference baseline mode keeps one model in memory, but stores private
        # adapter/optimizer state per logical job so correctness can be compared
        # against the batched path.
        baseline_adapter_states = build_matched_baseline_adapter_states(runtime_config, num_adapters)
        set_seed(seed)

    model = setup_model(runtime_config)
    device = next(model.parameters()).device

    use_ddp = ddp_enabled(runtime_config) and is_distributed()
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel

        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [device.index]
            ddp_kwargs["output_device"] = device.index
        model = DistributedDataParallel(model, **ddp_kwargs)

    optimizer = None if multi_adapter_baseline else build_optimizer(model, runtime_config)
    baseline_optimizers = None
    if baseline_adapter_states is not None:
        load_baseline_adapter_state(model, baseline_adapter_states[0])
        baseline_optimizers = [build_optimizer(model, runtime_config) for _ in range(num_adapters)]

    return TrainingState(
        model=model,
        device=device,
        adapter_type=adapter_type,
        num_adapters=num_adapters,
        jobs=jobs,
        job_groups=job_groups,
        multi_adapter_baseline=multi_adapter_baseline,
        optimizer=optimizer,
        baseline_optimizers=baseline_optimizers,
        baseline_adapter_states=baseline_adapter_states,
        max_grad_norm=runtime_config.get("max_grad_norm"),
        ddp_enabled=use_ddp,
    )


def prepare_batch(state: TrainingState, batch: dict) -> dict:
    prepared_batch = move_batch_to_device(batch, state.device)
    if not state.multi_adapter_baseline and "adapter_ids" not in prepared_batch:
        # Older configs may not provide job routing metadata yet. In that case
        # we fall back to deterministic round-robin assignment.
        prepared_batch["adapter_ids"] = build_adapter_ids(
            prepared_batch,
            state.num_adapters,
            state.device,
        )
    return prepared_batch


def forward_backward(state: TrainingState, batch: dict) -> dict:
    if state.has_pending_step:
        raise RuntimeError("Call optim_step() before starting another forward_backward() call.")

    prepared_batch = prepare_batch(state, batch)
    result = {
        "outputs": None,
        "loss": None,
        "examples_seen": batch_size(prepared_batch),
        "tokens_seen": token_count(prepared_batch),
        "macro_steps": 1,
        "optimizer_steps": 0,
    }

    if state.multi_adapter_baseline:
        # The baseline reference path replays the same macro-batch as separate
        # per-adapter steps, then applies their optimizer updates independently.
        state.pending_baseline_grads = [None] * state.num_adapters
        for adapter_idx, sub_batch in enumerate(split_batch_for_adapters(prepared_batch, state.num_adapters)):
            if batch_size(sub_batch) == 0:
                continue
            load_baseline_adapter_state(state.model, state.baseline_adapter_states[adapter_idx])
            outputs, loss = _run_forward_backward(
                state.model,
                sub_batch,
                optimizer=state.baseline_optimizers[adapter_idx],
            )
            state.pending_baseline_grads[adapter_idx] = extract_trainable_grads(state.model)
            result["outputs"] = outputs
            result["loss"] = loss
    else:
        outputs, loss = _run_forward_backward(state.model, prepared_batch, optimizer=state.optimizer)
        result["outputs"] = outputs
        result["loss"] = loss

    state.has_pending_step = True
    return result


def optim_step(state: TrainingState) -> int:
    if not state.has_pending_step:
        return 0

    if state.multi_adapter_baseline:
        optimizer_steps = 0
        for adapter_idx, gradients in enumerate(state.pending_baseline_grads or []):
            if gradients is None:
                continue
            load_baseline_adapter_state(state.model, state.baseline_adapter_states[adapter_idx])
            optimizer = state.baseline_optimizers[adapter_idx]
            optimizer.zero_grad()
            load_trainable_grads(state.model, gradients)
            if _has_non_finite_trainable_grads(state.model):
                optimizer.zero_grad(set_to_none=True)
                continue
            _clip_trainable_grads(state.model, state.max_grad_norm)
            optimizer.step()
            state.baseline_adapter_states[adapter_idx] = extract_baseline_adapter_state(state.model)
            optimizer_steps += 1
        state.pending_baseline_grads = None
    else:
        if _has_non_finite_trainable_grads(state.model):
            if state.optimizer is not None:
                state.optimizer.zero_grad(set_to_none=True)
            state.has_pending_step = False
            return 0
        _clip_trainable_grads(state.model, state.max_grad_norm)
        state.optimizer.step()
        optimizer_steps = 1

    state.has_pending_step = False
    return optimizer_steps


def clear_pending_step(state: TrainingState) -> None:
    if state.multi_adapter_baseline:
        state.pending_baseline_grads = None
        for optimizer in state.baseline_optimizers or []:
            optimizer.zero_grad(set_to_none=True)
    elif state.optimizer is not None:
        state.optimizer.zero_grad(set_to_none=True)

    for parameter in state.model.parameters():
        if parameter.requires_grad:
            parameter.grad = None

    state.has_pending_step = False


def train_epoch(state: TrainingState, dataloader) -> dict:
    step_times = []
    examples_seen = 0
    tokens_seen = 0
    optimizer_steps = 0
    last_loss = None

    for batch in dataloader:
        synchronize_device(state.device)
        step_start = perf_counter()
        step_result = forward_backward(state, batch)
        optimizer_steps += optim_step(state)
        synchronize_device(state.device)

        step_times.append(perf_counter() - step_start)
        examples_seen += step_result["examples_seen"]
        tokens_seen += step_result["tokens_seen"]
        last_loss = step_result["loss"]

    return {
        "step_times": step_times,
        "examples_seen": examples_seen,
        "tokens_seen": tokens_seen,
        "last_loss": last_loss,
        "macro_steps": len(step_times),
        "optimizer_steps": optimizer_steps,
    }


def summarize_training(
    *,
    adapter_type: str,
    num_adapters: int,
    step_times: list[float],
    total_time: float,
    examples_seen: int,
    tokens_seen: int,
    macro_steps: int,
    optimizer_steps: int,
    last_loss,
) -> dict:
    return {
        "final_loss": last_loss.item() if last_loss is not None else None,
        "num_steps": len(step_times),
        "total_time_s": total_time,
        "avg_step_time_s": (sum(step_times) / len(step_times)) if step_times else 0.0,
        "examples_seen": examples_seen,
        "tokens_seen": tokens_seen,
        "macro_steps": macro_steps,
        "optimizer_steps": optimizer_steps,
        "examples_per_second": (examples_seen / total_time) if total_time else 0.0,
        "tokens_per_second": (tokens_seen / total_time) if total_time else 0.0,
        "adapter_type": adapter_type,
        "num_adapters": num_adapters,
    }


def build_lora_checkpoint(state: TrainingState, config_options: dict, metrics: dict | None = None) -> dict:
    checkpoint = {
        "base_model": config_options["base_model"],
        "adapter_type": state.adapter_type,
        "num_adapters": state.num_adapters,
        "jobs": [asdict(job) for job in state.jobs],
        "job_groups": [asdict(group) for group in state.job_groups],
        "lora_config": deepcopy(config_options["lora_config"]),
        "config": deepcopy(config_options),
    }
    if metrics is not None:
        checkpoint["metrics"] = deepcopy(metrics)

    if state.multi_adapter_baseline:
        checkpoint["adapter_states"] = state.baseline_adapter_states
        return checkpoint

    if state.adapter_type in {"baseline", "baseline_lora"}:
        checkpoint["adapter_states"] = extract_baseline_adapter_state(state.model)
        return checkpoint

    checkpoint["adapter_states"] = extract_batched_lora_state(state.model)
    return checkpoint
