"""Batched LoRA module for shared-backbone multi-adapter execution.

Each linear layer keeps one bank of adapter weights, but the forward pass routes
each example to its adapter id. Adapters with similar ranks are processed in
groups so the batched path wastes less work on padding.
"""

import torch
from .scheduler import compute_rank_groups


class BatchedLoRA(torch.nn.Module):
    def __init__(self,
                linear_layer: torch.nn.Linear,
                num_adapters: int = 1,
                rank: int | list[int] = 1,
                alpha: float | torch.Tensor = 1.0,
                device: torch.device = torch.device('cpu')):
        super().__init__()
        self.linear_layer = linear_layer
        self.num_adapters = num_adapters
        
        if isinstance(rank, int):
            self.ranks = [rank] * num_adapters
        else:
            if len(rank) != num_adapters:
                raise ValueError(
                    f"Expected {num_adapters} ranks, got {len(rank)}"
                )
        
            self.ranks = list(rank)
        
        self.max_rank = max(self.ranks)

        self.rank_groups = compute_rank_groups(self.ranks)
        grouped_adapters = torch.empty(num_adapters, dtype=torch.long)
        for i, group in enumerate(self.rank_groups):
            for adapter_id in group:
                grouped_adapters[adapter_id] = i
        self.register_buffer("grouped_adapters_tensor", grouped_adapters)

        self.linear_layer.requires_grad_(False)

        alpha_tensor = torch.as_tensor(alpha, dtype=linear_layer.weight.dtype, device=device)
        if alpha_tensor.numel() == 1:
            alpha_tensor = alpha_tensor.repeat(num_adapters)
        elif alpha_tensor.shape != (num_adapters,):
            raise ValueError(
                f"Alpha must be a scalar or a tensor of shape ({num_adapters},), got {tuple(alpha_tensor.shape)}"
            )
        self.register_buffer("alpha_tensor", alpha_tensor)

        rank_tensor = torch.as_tensor(self.ranks, dtype=linear_layer.weight.dtype, device=device)
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
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device

        adapter_ids = self._resolve_adapter_ids(batch_size, device)
        token_adapter_ids = adapter_ids.repeat_interleave(seq_len)

        flat_x = x.reshape(batch_size * seq_len, hidden_dim).to(dtype=self.A.dtype)

        base_output = self.linear_layer(x)

        flat_output = torch.zeros(flat_x.shape[0], self.linear_layer.out_features, device=device, dtype=self.A.dtype)

        group_ids = self.grouped_adapters_tensor.index_select(0, token_adapter_ids)

        # Tokens are first routed by adapter id, then co-processed by rank group.
        # That keeps per-adapter identity intact while still letting similar
        # adapters share one batched matrix path.
        for group_id in group_ids.unique():
            indices = (group_ids == group_id).nonzero(as_tuple=True)[0]

            group_adapter_ids = token_adapter_ids[indices]
            group_rank = int(self.rank_tensor[group_adapter_ids].max().item())

            # TODO (Gwen): A and B are still being duplicated :(
            A = self.A.index_select(0, group_adapter_ids)[:, :group_rank, :]
            B = self.B.index_select(0, group_adapter_ids)[:, :, :group_rank]

            x_group = flat_x[indices]

            Ax = torch.bmm(A, x_group.unsqueeze(-1)).squeeze(-1)
            BAx = torch.bmm(B, Ax.unsqueeze(-1)).squeeze(-1)

            scales = (self.alpha_tensor.index_select(0, group_adapter_ids) / self.rank_tensor.index_select(0, group_adapter_ids)).unsqueeze(-1)

            flat_output[indices] = BAx * scales

        lora_output = flat_output.reshape(batch_size, seq_len, -1).to(dtype=base_output.dtype)

        return base_output + lora_output
