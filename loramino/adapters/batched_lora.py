import torch


class BatchedLoRA(torch.nn.Module):
    def __init__(self,
                linear_layer: torch.nn.Linear,
                num_adapters: int = 1,
                rank: int = 1,
                alpha: float | torch.Tensor = 1.0,
                device: torch.device = torch.device('cpu')):
        super().__init__()
        self.linear_layer = linear_layer
        self.num_adapters = num_adapters
        self.rank = rank
        self.linear_layer.requires_grad_(False)

        alpha_tensor = torch.as_tensor(alpha, dtype=linear_layer.weight.dtype, device=device)
        if alpha_tensor.numel() == 1:
            alpha_tensor = alpha_tensor.repeat(num_adapters)
        elif alpha_tensor.shape != (num_adapters,):
            raise ValueError(
                f"Alpha must be a scalar or a tensor of shape ({num_adapters},), got {tuple(alpha_tensor.shape)}"
            )
        self.register_buffer("alpha", alpha_tensor)

        parameter_kwargs = {
            "device": device,
            "dtype": linear_layer.weight.dtype,
        }

        # Gaussian noise
        self.A = torch.nn.Parameter(
            torch.randn(num_adapters, rank, linear_layer.in_features, **parameter_kwargs) * 0.01
        )
        # Zero
        self.B = torch.nn.Parameter(
            torch.zeros(num_adapters, linear_layer.out_features, rank, **parameter_kwargs)
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
        A = self.A.index_select(0, token_adapter_ids)
        B = self.B.index_select(0, token_adapter_ids)

        base_output = self.linear_layer(x)
        Ax = torch.bmm(A, flat_x.unsqueeze(-1)).squeeze(-1)
        BAx = torch.bmm(B, Ax.unsqueeze(-1)).squeeze(-1)
        scales = (self.alpha.index_select(0, token_adapter_ids) / self.rank).unsqueeze(-1)
        lora_output = (BAx * scales).reshape(batch_size, seq_len, -1).to(dtype=base_output.dtype)

        return base_output + lora_output
