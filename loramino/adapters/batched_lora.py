"""Batched LoRA module for shared-backbone multi-adapter execution."""

from dataclasses import dataclass

import torch

from .lora_kernel import grouped_lora_forward
from .scheduler import compute_rank_groups


@dataclass(frozen=True)
class AdapterRoutingLayout:
    """Cached packed-batch routing shared across BatchedLoRA layers."""

    adapter_ids: torch.Tensor
    tokens_per_example: int
    segment_adapter_ids: torch.Tensor
    segment_token_starts: torch.Tensor
    segment_token_counts: torch.Tensor


def build_contiguous_segment_layout(
    adapter_ids: torch.Tensor,
    *,
    tokens_per_example: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if adapter_ids.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=adapter_ids.device)
        return empty, empty, empty

    segment_starts = torch.zeros_like(adapter_ids, dtype=torch.bool)
    segment_starts[0] = True
    if adapter_ids.numel() > 1:
        segment_starts[1:] = adapter_ids[1:] != adapter_ids[:-1]

    segment_indices = segment_starts.nonzero(as_tuple=True)[0]
    next_segment_indices = torch.cat(
        (
            segment_indices[1:],
            torch.tensor([adapter_ids.shape[0]], device=adapter_ids.device, dtype=torch.long),
        )
    )
    segment_adapter_ids = adapter_ids.index_select(0, segment_indices)
    segment_token_starts = segment_indices * tokens_per_example
    segment_token_counts = (next_segment_indices - segment_indices) * tokens_per_example
    return segment_adapter_ids, segment_token_starts, segment_token_counts


def build_adapter_routing_layout(
    adapter_ids: torch.Tensor,
    *,
    tokens_per_example: int,
) -> AdapterRoutingLayout:
    detached_adapter_ids = adapter_ids.detach()
    segment_adapter_ids, segment_token_starts, segment_token_counts = build_contiguous_segment_layout(
        detached_adapter_ids,
        tokens_per_example=tokens_per_example,
    )
    return AdapterRoutingLayout(
        adapter_ids=detached_adapter_ids,
        tokens_per_example=int(tokens_per_example),
        segment_adapter_ids=segment_adapter_ids,
        segment_token_starts=segment_token_starts,
        segment_token_counts=segment_token_counts,
    )


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
        self.register_buffer(
            "scales",
            alpha_tensor / rank_tensor.to(device=device, dtype=alpha_tensor.dtype),
        )

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
        self.active_routing_layout: AdapterRoutingLayout | None = None

    def _resolve_adapter_ids(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.active_adapter_ids is None:
            return torch.arange(batch_size, device=device) % self.num_adapters

        adapter_ids = self.active_adapter_ids.to(device=device)
        if adapter_ids.shape != (batch_size,):
            raise ValueError(
                f"adapter_ids must have shape ({batch_size},), got {tuple(adapter_ids.shape)}"
            )
        return adapter_ids

    def _resolve_routing_layout(
        self,
        *,
        batch_size: int,
        tokens_per_example: int,
        device: torch.device,
    ) -> AdapterRoutingLayout:
        routing_layout = self.active_routing_layout
        if (
            routing_layout is not None
            and routing_layout.tokens_per_example == tokens_per_example
            and routing_layout.adapter_ids.device == device
            and routing_layout.adapter_ids.shape == (batch_size,)
            and (
                self.active_adapter_ids is None
                or routing_layout.adapter_ids.data_ptr() == self.active_adapter_ids.data_ptr()
            )
        ):
            return routing_layout

        adapter_ids = self._resolve_adapter_ids(batch_size, device)
        routing_layout = build_adapter_routing_layout(
            adapter_ids,
            tokens_per_example=tokens_per_example,
        )
        self.active_routing_layout = routing_layout
        return routing_layout

    def forward(self, x):
        batch_size = x.shape[0]
        hidden_dim = x.shape[-1]
        tokens_per_example = x.numel() // (batch_size * hidden_dim)
        device = x.device

        routing_layout = self._resolve_routing_layout(
            batch_size=batch_size,
            tokens_per_example=tokens_per_example,
            device=device,
        )
        flat_x = x.reshape(batch_size * tokens_per_example, hidden_dim).to(dtype=self.A.dtype)

        base_output = self.linear_layer(x)
        flat_output = grouped_lora_forward(
            flat_x,
            None,
            self.A,
            self.B,
            self.scales,
            self.rank_tensor,
            adapter_order=self.adapter_execution_order,
            adapter_group_ids=self.adapter_group_ids,
            segment_adapter_ids=routing_layout.segment_adapter_ids,
            segment_token_starts=routing_layout.segment_token_starts,
            segment_token_counts=routing_layout.segment_token_counts,
            backend=self.kernel_backend,
        )
        lora_output = flat_output.reshape(*x.shape[:-1], self.linear_layer.out_features).to(dtype=base_output.dtype)

        return base_output + lora_output
