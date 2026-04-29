"""Small distributed helpers for multi-GPU evaluation and training."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return env_world_size()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def ddp_enabled(config_options: dict | None = None) -> bool:
    if env_world_size() <= 1:
        return False
    if config_options is None:
        return True
    return config_options.get("distributed_mode", "ddp") == "ddp"


def is_primary_rank() -> bool:
    return get_rank() == 0


def initialize_distributed() -> None:
    if env_world_size() <= 1:
        return
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this environment.")
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())


def finalize_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_gather_objects(value):
    if not is_distributed():
        return [value]
    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def _reduce_scalar(value: float, *, op: dist.ReduceOp, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def reduce_sum(value: float, *, device: torch.device) -> float:
    if not is_distributed():
        return float(value)
    return _reduce_scalar(value, op=dist.ReduceOp.SUM, device=device)


def reduce_max(value: float, *, device: torch.device) -> float:
    if not is_distributed():
        return float(value)
    return _reduce_scalar(value, op=dist.ReduceOp.MAX, device=device)


def reduce_optional_mean(value: float | None, *, device: torch.device) -> float | None:
    if not is_distributed():
        return value

    is_valid = value is not None and torch.isfinite(torch.tensor(value))
    payload = torch.tensor(
        [float(value) if is_valid else 0.0, 1.0 if is_valid else 0.0],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    if payload[1].item() == 0:
        return None
    return float((payload[0] / payload[1]).item())
