"""Batched LoRA module for shared-backbone multi-adapter execution."""

import torch

from .lora_kernel import grouped_lora_forward
from .scheduler import compute_rank_groups


class BatchedLoRA(torch.nn.Module):
    def __init__(self,
                linear_layer: torch.nn.Linear,
                num_adapters: int = 1,
                rank: int | list[int] = 1,
                alpha: float | torch.Tensor = 1.0,
                rank_groups: list[list[int]] | None = None,
                kernel_backend: str = "auto",
                device: torch.device = torch.device('cpu')):
        super().__init__()
        self.linear_layer = linear_layer
        self.num_adapters = num_adapters
        self.kernel_backend = kernel_backend
        
        if isinstance(rank, int):
            self.ranks = [rank] * num_adapters
        else:
            if len(rank) != num_adapters:
                raise ValueError(
                    f"Expected {num_adapters} ranks, got {len(rank)}"
                )
        
            self.ranks = list(rank)

        self.max_rank = max(self.ranks)

        self.rank_groups = list(map(list, rank_groups)) if rank_groups is not None else compute_rank_groups(self.ranks)
        execution_order = [adapter_id for group in self.rank_groups for adapter_id in group]
        adapter_group_ids = torch.empty(num_adapters, dtype=torch.long)
        for group_id, group in enumerate(self.rank_groups):
            for adapter_id in group:
                adapter_group_ids[adapter_id] = group_id
        self.register_buffer("adapter_execution_order", torch.tensor(execution_order, dtype=torch.long))
        self.register_buffer("adapter_group_ids", adapter_group_ids)

        self.linear_layer.requires_grad_(False)

        alpha_tensor = torch.as_tensor(alpha, dtype=linear_layer.weight.dtype, device=device)
        if alpha_tensor.numel() == 1:
            alpha_tensor = alpha_tensor.repeat(num_adapters)
        elif alpha_tensor.shape != (num_adapters,):
            raise ValueError(
                f"Alpha must be a scalar or a tensor of shape ({num_adapters},), got {tuple(alpha_tensor.shape)}"
            )
        self.register_buffer("alpha_tensor", alpha_tensor)

        rank_tensor = torch.as_tensor(self.ranks, dtype=torch.long, device=device)
        self.register_buffer("rank_tensor", rank_tensor)

        parameter_kwargs = {
            "device": device,
            "dtype": linear_layer.weight.dtype,
        }

        # Gaussian noise
        self.A = torch.nn.Parameter(
            torch.randn(num_adapters, self.max_rank, linear_layer.in_features, **parameter_kwargs) * 0.01
        )
        # Zero
        self.B = torch.nn.Parameter(
            torch.zeros(num_adapters, linear_layer.out_features, self.max_rank, **parameter_kwargs)
        )
        self.active_adapter_ids = None

    def _resolve_adapter_ids(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.active_adapter_ids is None:
            return torch.arange(batch_size, device=device) % self.num_adapters

        adapter_ids = self.active_adapter_ids.to(device=device)
        if adapter_ids.shape != (batch_size,):
            raise ValueError(
                f"adapter_ids must have shape ({batch_size},), got {tuple(adapter_ids.shape)}"
            )
        return adapter_ids

    def forward(self, x):
        batch_size = x.shape[0]
        hidden_dim = x.shape[-1]
        tokens_per_example = x.numel() // (batch_size * hidden_dim)
        device = x.device

        adapter_ids = self._resolve_adapter_ids(batch_size, device)
        token_adapter_ids = adapter_ids.repeat_interleave(tokens_per_example)
        scales = self.alpha_tensor / self.rank_tensor.to(device=device, dtype=self.alpha_tensor.dtype)

        flat_x = x.reshape(batch_size * tokens_per_example, hidden_dim).to(dtype=self.A.dtype)

        base_output = self.linear_layer(x)
        flat_output = grouped_lora_forward(
            flat_x,
            token_adapter_ids,
            self.A,
            self.B,
            scales,
            self.rank_tensor,
            adapter_order=self.adapter_execution_order,
            adapter_group_ids=self.adapter_group_ids,
            backend=self.kernel_backend,
        )
        lora_output = flat_output.reshape(*x.shape[:-1], self.linear_layer.out_features).to(dtype=base_output.dtype)

        return base_output + lora_output
