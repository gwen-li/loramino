from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from loramino.adapters.baseline_lora import BaselineLoRA
from loramino.adapters.batched_lora import BatchedLoRA
from loramino.data.orca_math import OrcaMath
from loramino.models import loader as load_model


ADAPTER_REGISTRY = {
    "baseline": BaselineLoRA,
    "baseline_lora": BaselineLoRA,
    "batched": BatchedLoRA,
    "batched_lora": BatchedLoRA,
}


@dataclass
class TrainingState:
    model: torch.nn.Module
    device: torch.device
    adapter_type: str
    num_adapters: int
    multi_adapter_baseline: bool
    optimizer: torch.optim.Optimizer | None = None
    baseline_optimizers: list[torch.optim.Optimizer] | None = None
    baseline_adapter_states: list[dict[str, dict[str, torch.Tensor]]] | None = None
    pending_baseline_grads: list[dict[str, torch.Tensor | None] | None] | None = None
    has_pending_step: bool = False


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
        return torch.device(config_options["device"])
    if torch.cuda.is_available():
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


def build_dataloader(config_options: dict, tokenizer, shuffle: bool = True) -> DataLoader:
    dataset = OrcaMath(
        config_options.get("dataset", ""),
        tokenizer,
        max_length=config_options.get("max_length", 256),
    )
    generator = None
    if shuffle and config_options.get("seed") is not None:
        generator = torch.Generator()
        generator.manual_seed(config_options["seed"])

    return DataLoader(
        dataset,
        batch_size=config_options["batch_size"],
        shuffle=shuffle,
        generator=generator,
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


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def batch_size(batch: dict) -> int:
    return next(iter(batch.values())).shape[0]


def token_count(batch: dict) -> int:
    if "input_ids" not in batch:
        return 0
    return batch["input_ids"].numel()


def build_adapter_ids(batch: dict, num_adapters: int, device: torch.device) -> torch.Tensor:
    return torch.arange(batch_size(batch), device=device) % num_adapters


def split_batch_for_adapters(batch: dict, num_adapters: int) -> list[dict]:
    adapter_ids = build_adapter_ids(batch, num_adapters, next(iter(batch.values())).device)
    return [
        {key: value[adapter_ids == adapter_idx] for key, value in batch.items()}
        for adapter_idx in range(num_adapters)
    ]


def collect_baseline_lora_modules(model: torch.nn.Module) -> dict[str, BaselineLoRA]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, BaselineLoRA)
    }


def collect_batched_lora_modules(model: torch.nn.Module) -> dict[str, BatchedLoRA]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, BatchedLoRA)
    }


def set_batched_adapter_ids(model: torch.nn.Module, adapter_ids: torch.Tensor | None) -> None:
    for module in collect_batched_lora_modules(model).values():
        module.active_adapter_ids = None if adapter_ids is None else adapter_ids.detach()


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
            "alpha": module.alpha[adapter_index].detach().clone(),
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
            "alpha": module.alpha.detach().clone(),
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
    set_batched_adapter_ids(model, adapter_ids)
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
    seed = config_options.get("seed")
    set_seed(seed)

    adapter_type = normalize_adapter_type(config_options)
    num_adapters = get_num_adapters(config_options)
    multi_adapter_baseline = adapter_type == "baseline_lora" and num_adapters > 1

    baseline_adapter_states = None
    if multi_adapter_baseline:
        baseline_adapter_states = build_matched_baseline_adapter_states(config_options, num_adapters)
        set_seed(seed)

    model = setup_model(config_options)
    device = next(model.parameters()).device

    optimizer = None if multi_adapter_baseline else build_optimizer(model, config_options)
    baseline_optimizers = None
    if baseline_adapter_states is not None:
        load_baseline_adapter_state(model, baseline_adapter_states[0])
        baseline_optimizers = [build_optimizer(model, config_options) for _ in range(num_adapters)]

    return TrainingState(
        model=model,
        device=device,
        adapter_type=adapter_type,
        num_adapters=num_adapters,
        multi_adapter_baseline=multi_adapter_baseline,
        optimizer=optimizer,
        baseline_optimizers=baseline_optimizers,
        baseline_adapter_states=baseline_adapter_states,
    )


def prepare_batch(state: TrainingState, batch: dict) -> dict:
    prepared_batch = move_batch_to_device(batch, state.device)
    if not state.multi_adapter_baseline:
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
            optimizer.step()
            state.baseline_adapter_states[adapter_idx] = extract_baseline_adapter_state(state.model)
            optimizer_steps += 1
        state.pending_baseline_grads = None
    else:
        state.optimizer.step()
        optimizer_steps = 1

    state.has_pending_step = False
    return optimizer_steps


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
